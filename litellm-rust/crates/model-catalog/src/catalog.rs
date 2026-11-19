use std::borrow::Cow;
use std::collections::HashSet;

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::Value;

use crate::error::{Error, Loaded, Rejects, Repr, pointer_child, pointer_for_key};
use crate::fallback::{FallbackGeneralizations, FallbackRule};
use crate::model::Model;
use crate::store::{Rows, Store};
use crate::wire::{EntryWire, Fields};

const SAMPLE_SPEC: &str = "sample_spec";
const FALLBACK_GENERALIZATIONS: &str = "fallback_generalizations";

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntegrityLimits {
    min_model_count: usize,
    min_reference_ratio: f64,
}

#[derive(Clone, Debug)]
pub struct ModelEntry {
    fields: Fields,
    model: Option<Model>,
}

#[derive(Clone, Copy, Debug)]
pub struct ModelMatch<'a> {
    pub matched_key: &'a str,
    pub canonical_key: &'a str,
    pub entry: &'a ModelEntry,
}

#[derive(Clone, Debug)]
struct Section {
    index: usize,
    key: String,
    value: Value,
}

/// One fetched key kept off the served map because its row failed validation
#[derive(Clone, Debug)]
pub struct Rejection<'a> {
    pub key: String,
    pub errors: Vec<&'a Error>,
    /// The bundled row for the key is served instead. `false` means the key is left out
    pub replaced_by_bundled: bool,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
#[serde(untagged)]
pub enum Written<'a> {
    Entry(&'a Fields),
    Section(&'a Value),
}

#[derive(Clone, Debug)]
pub struct Catalog {
    store: Store<CatalogRows>,
    sections: Vec<Section>,
    fallback_generalizations: Option<FallbackGeneralizations>,
}

/// Models and aliases, an alias served as its owner's row
#[derive(Clone, Debug)]
struct CatalogRows {
    entries: IndexMap<String, ModelEntry>,
    aliases: IndexMap<String, String>,
}

impl CatalogRows {
    fn owner<'a>(&'a self, key: &'a str) -> &'a str {
        self.aliases.get(key).map_or(key, String::as_str)
    }

    fn fields(&self, key: &str) -> Option<&Fields> {
        let fields = self.entries.get(self.owner(key))?.fields();
        (!fields.contains_key("alias_of")).then_some(fields)
    }
}

/// Model names match ignoring case, and keys are grouped by their raw `litellm_provider`
impl Rows for CatalogRows {
    type Row = Fields;
    type Stamp = ();

    fn fold(name: &str) -> String {
        name.to_lowercase()
    }

    fn stamp(&self) -> Self::Stamp {}

    fn contains(&self, key: &str) -> bool {
        self.fields(key).is_some()
    }

    fn row(&self, key: &str) -> Option<Cow<'_, Fields>> {
        self.fields(key).map(Cow::Borrowed)
    }

    fn scan(&self, visit: &mut dyn FnMut(&str, Option<&str>)) {
        let keys = self.entries.keys().chain(
            self.aliases
                .keys()
                .filter(|key| !self.entries.contains_key(*key)),
        );
        for key in keys {
            if self.fields(key).is_none() {
                continue;
            }
            let provider = self
                .fields(key)
                .and_then(|fields| fields.get("litellm_provider")?.as_str());
            visit(key, provider);
        }
    }
}

impl IntegrityLimits {
    pub fn new(min_model_count: usize, min_reference_ratio: f64) -> Option<Self> {
        (0.0..=1.0).contains(&min_reference_ratio).then_some(Self {
            min_model_count,
            min_reference_ratio,
        })
    }
}

impl ModelEntry {
    pub fn field(&self, name: &str) -> Option<&Value> {
        self.fields.get(name)
    }

    pub fn fields(&self) -> &Fields {
        &self.fields
    }

    pub fn model(&self) -> Option<&Model> {
        self.model.as_ref()
    }
}

impl Catalog {
    pub fn parse(body: &[u8]) -> Result<Loaded<Self>, Error> {
        let document = serde_json::from_slice::<Fields>(body)
            .map_err(|error| Error::new("", Repr::Json(error)))?;

        let mut problems = Vec::new();
        let mut sections = Vec::new();
        let mut entries = IndexMap::with_capacity(document.len());
        let mut alias_lists = Vec::new();
        for (index, (name, value)) in document.into_iter().enumerate() {
            let fields = match value {
                Value::Object(fields) if !is_section(&name) => fields,
                value => {
                    if !is_section(&name) {
                        problems.push(Error::from_key(&name, Repr::NotAnObject));
                    }
                    sections.push(Section {
                        index,
                        key: name,
                        value,
                    });
                    continue;
                }
            };
            let (model, aliases, row_problems) = if fields.contains_key("alias_of") {
                (None, None, Vec::new())
            } else {
                read_row(&fields)
            };
            problems.extend(
                row_problems
                    .into_iter()
                    .map(|problem| problem.prepend(&name)),
            );
            if let Some(aliases) = aliases {
                alias_lists.push((name.clone(), aliases));
            }
            entries.insert(name, ModelEntry { fields, model });
        }
        if entries.is_empty() {
            return Err(Error::new("", Repr::Empty));
        }

        let mut aliases: IndexMap<String, String> = IndexMap::new();
        for (model, names) in alias_lists {
            for alias in names {
                if entries.contains_key(&alias) {
                    problems.push(Error::from_key(&model, Repr::AliasShadowsModel { alias }));
                } else if let Some(taken_by) = aliases.get(&alias) {
                    problems.push(Error::from_key(
                        &model,
                        Repr::AliasTaken {
                            alias,
                            taken_by: taken_by.clone(),
                        },
                    ));
                } else {
                    aliases.insert(alias, model.clone());
                }
            }
        }

        for (name, entry) in &entries {
            if !entry.fields.contains_key("alias_of") {
                continue;
            }
            match alias_owner(name, &entries, &aliases) {
                Some(owner) => {
                    aliases.insert(name.clone(), owner);
                }
                None => problems.push(Error::new(
                    pointer_child(&pointer_for_key(name), "alias_of"),
                    Repr::InvalidAlias,
                )),
            }
        }

        let fallback_generalizations = sections
            .iter()
            .find(|section| section.key == FALLBACK_GENERALIZATIONS)
            .and_then(|section| {
                FallbackGeneralizations::deserialize(&section.value)
                    .map_err(|error| {
                        problems.push(Error::from_key(&section.key, Repr::Invalid(error)))
                    })
                    .ok()
            });
        Ok(Loaded {
            value: Self {
                store: Store::new(CatalogRows { entries, aliases }),
                sections,
                fallback_generalizations,
            },
            errors: problems,
        })
    }

    pub fn problems(body: &[u8]) -> Vec<Error> {
        let Loaded {
            value: catalog,
            errors,
        } = match Self::parse(body) {
            Ok(loaded) => loaded,
            Err(fatal) => return vec![fatal],
        };
        let unread = catalog.rows().entries.iter().flat_map(|(model, entry)| {
            let wire = if entry.fields.contains_key("alias_of") {
                EntryWire::default()
            } else {
                EntryWire::read(entry.fields())
            };
            wire.unknown.into_iter().map(move |(key, _)| {
                Error::new(
                    pointer_child(&pointer_for_key(model), &key),
                    Repr::UnreadField,
                )
            })
        });
        let no_fallback_rules = catalog
            .fallback_rules()
            .is_none_or(<[_]>::is_empty)
            .then(|| Error::from_key(FALLBACK_GENERALIZATIONS, Repr::NoFallbackRules));
        errors
            .into_iter()
            .chain(unread)
            .chain(no_fallback_rules)
            .collect()
    }

    pub fn row_errors(model: &str, fields: &Fields) -> Vec<Error> {
        if let Some(target) = fields.get("alias_of") {
            return if fields.len() == 1 && target.is_string() {
                Vec::new()
            } else {
                vec![Error::new(
                    pointer_child(&pointer_for_key(model), "alias_of"),
                    Repr::InvalidAlias,
                )]
            };
        }
        let (_, _, errors) = read_row(fields);
        errors
            .into_iter()
            .map(|error| error.prepend(model))
            .collect()
    }

    pub fn written(&self) -> impl Iterator<Item = (&str, Written<'_>)> {
        let mut sections = self.sections.iter().peekable();
        let mut entries = self.rows().entries.iter();
        (0..self.rows().entries.len() + self.sections.len()).map_while(move |index| match sections
            .next_if(|section| section.index == index)
        {
            Some(section) => Some((section.key.as_str(), Written::Section(&section.value))),
            None => entries
                .next()
                .map(|(name, entry)| (name.as_str(), Written::Entry(&entry.fields))),
        })
    }

    pub fn served(&self) -> impl Iterator<Item = (&str, &Fields)> {
        let rows = self.rows();
        rows.entries
            .iter()
            .filter_map(|(name, _)| Some((name.as_str(), rows.fields(name)?)))
            .chain(
                rows.aliases
                    .iter()
                    .filter(|(alias, _)| !rows.entries.contains_key(*alias))
                    .map(|(alias, owner)| (alias.as_str(), rows.entries[owner].fields())),
            )
    }

    pub fn check_size(&self, reference: usize, limits: IntegrityLimits) -> Result<(), Error> {
        let actual = self.model_count();
        if actual < limits.min_model_count {
            return Err(Error::new(
                "",
                Repr::BelowMinimum {
                    actual,
                    minimum: limits.min_model_count,
                },
            ));
        }
        if (actual as f64) < (reference as f64) * limits.min_reference_ratio {
            return Err(Error::new(
                "",
                Repr::Shrunk {
                    actual,
                    reference,
                    ratio: limits.min_reference_ratio,
                },
            ));
        }
        Ok(())
    }

    pub fn without_rejected(self, errors: &[Error]) -> (Self, Vec<&Error>) {
        let row_names: HashSet<String> = self
            .rows()
            .entries
            .keys()
            .cloned()
            .chain(
                self.sections
                    .iter()
                    .filter(|section| !is_section(&section.key))
                    .map(|section| section.key.clone()),
            )
            .collect();
        let rejected: HashSet<String> = errors
            .iter()
            .filter(|error| error.rejects() == Rejects::Row)
            .filter_map(|error| crate::error::pointer_root(error.path()))
            .filter(|root| row_names.contains(root))
            .collect();
        let removed_errors = errors
            .iter()
            .filter(|error| {
                crate::error::pointer_root(error.path())
                    .is_some_and(|root| rejected.contains(&root))
            })
            .collect();
        let removed_positions: HashSet<usize> = self
            .written()
            .enumerate()
            .filter(|(_, (key, _))| rejected.contains(*key))
            .map(|(index, _)| index)
            .collect();
        let mut catalog = self;
        catalog.store.edit(|rows| {
            rows.entries.retain(|name, _| !rejected.contains(name));
            rows.aliases
                .retain(|alias, owner| !rejected.contains(alias) && !rejected.contains(owner));
        });
        catalog
            .sections
            .retain(|section| is_section(&section.key) || !rejected.contains(&section.key));
        for section in &mut catalog.sections {
            let removed_before = removed_positions
                .iter()
                .filter(|index| **index < section.index)
                .count();
            section.index -= removed_before;
        }
        (catalog, removed_errors)
    }

    pub fn unrecognized(&self) -> impl Iterator<Item = (&str, &str)> {
        self.rows().entries.iter().flat_map(|(name, entry)| {
            entry
                .model
                .iter()
                .flat_map(|model| &model.unrecognized)
                .map(move |key| (name.as_str(), key.as_str()))
        })
    }

    pub fn lookup(&self, key: &str) -> Option<ModelMatch<'_>> {
        let rows = self.rows();
        let found = self.store.find_key(key)?;
        let (canonical_key, entry) = rows.entries.get_key_value(rows.owner(&found))?;
        let matched_key = rows
            .entries
            .get_key_value(found.as_str())
            .map(|(key, _)| key.as_str())
            .or_else(|| {
                rows.aliases
                    .get_key_value(found.as_str())
                    .map(|(key, _)| key.as_str())
            })?;
        Some(ModelMatch {
            matched_key,
            canonical_key,
            entry,
        })
    }

    pub fn model_count(&self) -> usize {
        self.rows().entries.len()
    }

    pub fn model_names(&self) -> impl Iterator<Item = &str> {
        self.rows().entries.keys().map(String::as_str)
    }

    pub fn aliases(&self) -> &IndexMap<String, String> {
        &self.rows().aliases
    }

    fn rows(&self) -> &CatalogRows {
        self.store.rows()
    }

    pub fn sample_spec(&self) -> Option<&Value> {
        self.sections
            .iter()
            .find(|section| section.key == SAMPLE_SPEC)
            .map(|section| &section.value)
    }

    pub fn fallback_rules(&self) -> Option<&[FallbackRule]> {
        Some(self.fallback_generalizations.as_ref()?.rules.as_slice())
    }
}

fn read_row(fields: &Fields) -> (Option<Model>, Option<Vec<String>>, Vec<Error>) {
    let wire = EntryWire::read(fields);
    let aliases = wire.attributes.aliases.clone();
    let timezone_error = unknown_timezone(&wire);
    match Model::try_from(wire) {
        Ok(model) => (Some(model), aliases, timezone_error.into_iter().collect()),
        Err(mut errors) => {
            errors.extend(timezone_error);
            (None, aliases, errors)
        }
    }
}

fn alias_owner(
    name: &str,
    entries: &IndexMap<String, ModelEntry>,
    aliases: &IndexMap<String, String>,
) -> Option<String> {
    let mut current = name;
    let mut visited = HashSet::new();
    loop {
        if !visited.insert(current) {
            return None;
        }
        if let Some(owner) = aliases.get(current) {
            current = owner;
            continue;
        }
        let fields = entries.get(current)?.fields();
        match fields.get("alias_of") {
            None => return Some(current.to_owned()),
            Some(Value::String(target)) if fields.len() == 1 => current = target,
            _ => return None,
        }
    }
}

fn is_section(key: &str) -> bool {
    key == SAMPLE_SPEC || key == FALLBACK_GENERALIZATIONS
}

fn unknown_timezone(wire: &EntryWire) -> Option<Error> {
    let zone = wire.off_peak.as_ref()?.weekday_timezone.as_deref()?.trim();
    (!zone.is_empty() && jiff::tz::TimeZone::get(zone).is_err()).then(|| {
        Error::from_path(
            "off_peak_pricing.weekday_timezone",
            Repr::UnknownTimezone {
                zone: zone.to_owned(),
            },
        )
    })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    const RULES: &str = r#""fallback_generalizations":{"rules":[{"name":"family","pattern":"^new-","model_info":{"mode":"chat"}}]}"#;

    #[rstest]
    #[case::clean(format!(r#"{{{RULES},"m":{{"litellm_provider":"test"}}}}"#), vec![])]
    #[case::not_json("{".to_owned(), vec![("", "json")])]
    #[case::row_error(format!(r#"{{{RULES},"m":{{}}}}"#), vec![("/m/litellm_provider", "missing_provider")])]
    #[case::unread_key(
        format!(r#"{{{RULES},"m":{{"litellm_provider":"test","typo_cost":1}}}}"#),
        vec![("/m/typo_cost", "unread_field")]
    )]
    #[case::empty_model_key(
        format!(r#"{{{RULES},"":{{"litellm_provider":"test","future_field":true}}}}"#),
        vec![("//future_field", "unread_field")]
    )]
    #[case::unread_key_on_invalid_row(
        format!(r#"{{{RULES},"m":{{"typo_cost":1}}}}"#),
        vec![("/m/litellm_provider", "missing_provider"), ("/m/typo_cost", "unread_field")]
    )]
    #[case::escaped_unread_key_on_invalid_row(
        format!(r#"{{{RULES},"open/router~v1":{{"typo/cost~":1}}}}"#),
        vec![("/open~1router~0v1/litellm_provider", "missing_provider"), ("/open~1router~0v1/typo~1cost~0", "unread_field")]
    )]
    #[case::no_rules(
        r#"{"m":{"litellm_provider":"test"}}"#.to_owned(),
        vec![("/fallback_generalizations", "no_fallback_rules")]
    )]
    #[case::empty_rules(
        r#"{"fallback_generalizations":{"rules":[]},"m":{"litellm_provider":"test"}}"#.to_owned(),
        vec![("/fallback_generalizations", "no_fallback_rules")]
    )]
    fn problems_list_what_a_cost_map_change_must_fix(
        #[case] body: String,
        #[case] expected: Vec<(&str, &str)>,
    ) {
        let problems: Vec<_> = Catalog::problems(body.as_bytes())
            .iter()
            .map(|problem| (problem.path().to_owned(), <&str>::from(problem.kind())))
            .collect();
        let expected: Vec<_> = expected
            .into_iter()
            .map(|(path, kind)| (path.to_owned(), kind))
            .collect();
        assert_eq!(problems, expected);
    }

    #[rstest]
    fn runtime_parsing_preserves_custom_fields_and_recovers_valid_rows() {
        let body = br#"{
            "good":{"litellm_provider":"test","custom_metadata":true},
            "bad":{"litellm_provider":"test","input_cost_per_token":"wrong"}
        }"#;
        let loaded = Catalog::parse(body).unwrap();
        assert_eq!(loaded.errors.len(), 1);
        assert_eq!(loaded.errors[0].path(), "/bad/input_cost_per_token");
        let (catalog, rejected) = loaded.value.without_rejected(&loaded.errors);
        assert_eq!(rejected.len(), 1);
        assert!(catalog.lookup("bad").is_none());
        let good = catalog.lookup("good").unwrap();
        assert!(good.entry.model().is_some());
        assert_eq!(
            good.entry.field("custom_metadata"),
            Some(&Value::Bool(true))
        );
        assert!(
            Catalog::problems(body)
                .iter()
                .any(|problem| problem.kind() == crate::ErrorKind::UnreadField)
        );
        assert!(
            Catalog::problems(body)
                .iter()
                .any(|problem| problem.kind() == crate::ErrorKind::NoFallbackRules)
        );
    }

    #[rstest]
    #[case::simple(r#"{"alias_of":"owner"}"#, "")]
    #[case::chain(r#"{"alias_of":"middle"}"#, r#", "middle":{"alias_of":"owner"}"#)]
    #[case::legacy(r#"{"alias_of":"legacy"}"#, "")]
    fn alias_of_serves_the_final_owner_and_preserves_written_rows(
        #[case] alias: &str,
        #[case] extra: &str,
    ) {
        let body = format!(
            r#"{{{RULES},"latest":{alias},"owner":{{"litellm_provider":"test","aliases":["legacy"]}}{extra}}}"#
        );
        let loaded = Catalog::parse(body.as_bytes()).unwrap();
        assert!(Catalog::problems(body.as_bytes()).is_empty());
        let catalog = loaded.value;
        let found = catalog.lookup("LATEST").unwrap();
        let owner = catalog.lookup("owner").unwrap();
        assert_eq!(found.canonical_key, "owner");
        assert!(std::ptr::eq(
            found.entry.model().unwrap(),
            owner.entry.model().unwrap()
        ));
        assert_eq!(
            catalog
                .served()
                .find(|(key, _)| *key == "latest")
                .unwrap()
                .1,
            owner.entry.fields()
        );
        assert_eq!(
            serde_json::to_value(
                catalog
                    .written()
                    .find(|(key, _)| *key == "latest")
                    .unwrap()
                    .1
            )
            .unwrap(),
            serde_json::from_str::<Value>(alias).unwrap()
        );
    }

    #[rstest]
    #[case::missing(r#"{"alias_of":"missing"}"#, "")]
    #[case::cycle(r#"{"alias_of":"middle"}"#, r#", "middle":{"alias_of":"latest"}"#)]
    #[case::self_cycle(r#"{"alias_of":"latest"}"#, "")]
    #[case::mixed(r#"{"alias_of":"owner","mode":"chat"}"#, "")]
    #[case::not_string(r#"{"alias_of":1}"#, "")]
    fn invalid_aliases_are_diagnosed_and_not_served(#[case] alias: &str, #[case] extra: &str) {
        let body =
            format!(r#"{{{RULES},"latest":{alias},"owner":{{"litellm_provider":"test"}}{extra}}}"#);
        let loaded = Catalog::parse(body.as_bytes()).unwrap();
        assert!(
            loaded
                .errors
                .iter()
                .any(|error| error.path() == "/latest/alias_of"
                    && error.kind() == crate::ErrorKind::InvalidAlias)
        );
        assert!(
            Catalog::problems(body.as_bytes())
                .iter()
                .all(|error| error.kind() == crate::ErrorKind::InvalidAlias)
        );
        assert!(loaded.value.lookup("latest").is_none());
        assert!(loaded.value.served().all(|(key, _)| key != "latest"));
        assert!(loaded.value.lookup("owner").is_some());
    }
}
