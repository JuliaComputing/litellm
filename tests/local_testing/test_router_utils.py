#### What this tests ####
# This tests utils used by llm router -> like llmrouter.get_settings()

import sys, os, time
import traceback, asyncio
import pytest

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import litellm
from litellm import Router
from litellm.router import Deployment, LiteLLM_Params
from litellm.types.router import ModelInfo
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dotenv import load_dotenv
from unittest.mock import patch, MagicMock, AsyncMock

load_dotenv()


def test_returned_settings():
    # this tests if the router raises an exception when invalid params are set
    # in this test both deployments have bad keys - Keep this test. It validates if the router raises the most recent exception
    litellm.set_verbose = True
    import openai

    try:
        print("testing if router raises an exception")
        model_list = [
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
                },
                "tpm": 240000,
                "rpm": 1800,
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  #
                    "model": "gpt-3.5-turbo",
                    "api_key": "bad-key",
                },
                "tpm": 240000,
                "rpm": 1800,
            },
        ]
        router = Router(
            model_list=model_list,
            redis_host=os.getenv("REDIS_HOST"),
            redis_password=os.getenv("REDIS_PASSWORD"),
            redis_port=int(os.getenv("REDIS_PORT")),
            routing_strategy="latency-based-routing",
            routing_strategy_args={"ttl": 10},
            set_verbose=False,
            num_retries=3,
            retry_after=5,
            allowed_fails=1,
            cooldown_time=30,
        )  # type: ignore

        settings = router.get_settings()
        print(settings)

        """
        routing_strategy: "simple-shuffle"
        routing_strategy_args: {"ttl": 10} # Average the last 10 calls to compute avg latency per model
        allowed_fails: 1
        num_retries: 3
        retry_after: 5 # seconds to wait before retrying a failed request
        cooldown_time: 30 # seconds to cooldown a deployment after failure
        """
        assert settings["routing_strategy"] == "latency-based-routing"
        assert settings["routing_strategy_args"]["ttl"] == 10
        assert settings["allowed_fails"] == 1
        assert settings["num_retries"] == 3
        assert settings["retry_after"] == 5
        assert settings["cooldown_time"] == 30

    except Exception:
        print(traceback.format_exc())
        pytest.fail("An error occurred - " + traceback.format_exc())


from litellm.types.utils import CallTypes


def test_update_kwargs_before_fallbacks_unit_test():
    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
                },
            }
        ],
    )

    kwargs = {"messages": [{"role": "user", "content": "write 1 sentence poem"}]}

    router._update_kwargs_before_fallbacks(
        model="gpt-3.5-turbo",
        kwargs=kwargs,
    )

    assert kwargs["litellm_trace_id"] is not None


@pytest.mark.parametrize(
    "call_type",
    [
        CallTypes.acompletion,
        CallTypes.atext_completion,
        CallTypes.aembedding,
        CallTypes.arerank,
        CallTypes.atranscription,
    ],
)
@pytest.mark.asyncio
async def test_update_kwargs_before_fallbacks(call_type):

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",
                "litellm_params": {
                    "model": "azure/gpt-4.1-mini",
                    "api_key": "bad-key",
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_AI_API_BASE"),
                },
            }
        ],
    )

    if call_type.value.startswith("a"):
        with patch.object(router, "async_function_with_fallbacks") as mock_client:
            if call_type.value == "acompletion":
                input_kwarg = {
                    "messages": [{"role": "user", "content": "Hello, how are you?"}],
                }
            elif (
                call_type.value == "atext_completion"
                or call_type.value == "aimage_generation"
            ):
                input_kwarg = {
                    "prompt": "Hello, how are you?",
                }
            elif call_type.value == "aembedding" or call_type.value == "arerank":
                input_kwarg = {
                    "input": "Hello, how are you?",
                }
            elif call_type.value == "atranscription":
                input_kwarg = {
                    "file": "path/to/file",
                }
            else:
                input_kwarg = {}

            await getattr(router, call_type.value)(
                model="gpt-3.5-turbo",
                **input_kwarg,
            )

            mock_client.assert_called_once()

            print(mock_client.call_args.kwargs)
            assert mock_client.call_args.kwargs["litellm_trace_id"] is not None


def test_router_get_model_info_wildcard_routes():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )
    model_info = router.get_router_model_info(
        deployment=None, received_model_name="gemini/gemini-1.5-flash", id="1"
    )
    print(model_info)
    assert model_info is not None
    assert model_info["tpm"] is not None
    assert model_info["rpm"] is not None


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_router_get_model_group_usage_wildcard_routes():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )

    resp = await router.acompletion(
        model="gemini/gemini-1.5-flash",
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        mock_response="Hello, I'm good.",
    )
    print(resp)

    await asyncio.sleep(2)

    tpm, rpm = await router.get_model_group_usage(model_group="gemini/gemini-1.5-flash")

    assert tpm is not None, "tpm is None"
    assert rpm is not None, "rpm is None"


@pytest.mark.asyncio
async def test_call_router_callbacks_on_success():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )

    with patch.object(
        router.cache, "async_increment_cache_pipeline", new=AsyncMock()
    ) as mock_callback:
        await router.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            mock_response="Hello, I'm good.",
        )
        await asyncio.sleep(1)
        assert mock_callback.call_count == 1

        increment_list = mock_callback.call_args_list[0].kwargs["increment_list"]
        assert len(increment_list) == 2

        for increment in increment_list:
            if "tpm" in increment["key"]:
                assert increment["key"].startswith(
                    "global_router:1:gemini/gemini-1.5-flash:tpm"
                )
                assert increment["increment_value"] == 30
            elif "rpm" in increment["key"]:
                assert increment["key"].startswith(
                    "global_router:1:gemini/gemini-1.5-flash:rpm"
                )
                assert increment["increment_value"] == 1


@pytest.mark.serial
@pytest.mark.asyncio
async def test_call_router_callbacks_on_failure():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            },
        ]
    )

    with patch.object(
        router.cache, "async_increment_cache", new=AsyncMock()
    ) as mock_callback:
        with pytest.raises(litellm.RateLimitError):
            await router.acompletion(
                model="gemini/gemini-1.5-flash",
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                mock_response="litellm.RateLimitError",
                num_retries=0,
            )
        await asyncio.sleep(3)
        print(mock_callback.call_args_list)
        assert mock_callback.call_count == 1

        assert (
            mock_callback.call_args_list[0]
            .kwargs["key"]
            .startswith("global_router:1:gemini/gemini-1.5-flash:rpm")
        )


@pytest.mark.asyncio
async def test_router_model_group_headers():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    from litellm.types.utils import OPENAI_RESPONSE_HEADERS

    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            }
        ]
    )

    for _ in range(2):
        resp = await router.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            mock_response="Hello, I'm good.",
        )
        await asyncio.sleep(1)

    assert (
        resp._hidden_params["additional_headers"]["x-litellm-model-group"]
        == "gemini/gemini-1.5-flash"
    )

    assert "x-ratelimit-remaining-requests" in resp._hidden_params["additional_headers"]
    assert "x-ratelimit-remaining-tokens" in resp._hidden_params["additional_headers"]


@pytest.mark.asyncio
async def test_get_remaining_model_group_usage():
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    litellm.model_cost = litellm.get_model_cost_map(url="")
    from litellm.types.utils import OPENAI_RESPONSE_HEADERS

    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1},
            }
        ]
    )
    for _ in range(2):
        resp = await router.acompletion(
            model="gemini/gemini-1.5-flash",
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            mock_response="Hello, I'm good.",
        )
        assert (
            "x-ratelimit-remaining-tokens" in resp._hidden_params["additional_headers"]
        )
        assert (
            "x-ratelimit-remaining-requests"
            in resp._hidden_params["additional_headers"]
        )
        await asyncio.sleep(1)

    remaining_usage = await router.get_remaining_model_group_usage(
        model_group="gemini/gemini-1.5-flash"
    )
    assert remaining_usage is not None
    assert "x-ratelimit-remaining-requests" in remaining_usage
    assert "x-ratelimit-remaining-tokens" in remaining_usage


@pytest.mark.parametrize(
    "potential_access_group, expected_result",
    [("gemini-models", True), ("gemini-models-2", False), ("gemini/*", False)],
)
def test_router_get_model_access_groups(potential_access_group, expected_result):
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": 1, "access_groups": ["gemini-models"]},
            },
        ]
    )
    access_groups = router._is_model_access_group_for_wildcard_route(
        model_access_group=potential_access_group
    )
    assert access_groups == expected_result


def test_router_redis_cache():
    router = Router(
        model_list=[{"model_name": "gemini/*", "litellm_params": {"model": "gemini/*"}}]
    )

    redis_cache = MagicMock()

    router._update_redis_cache(cache=redis_cache)

    assert router.cache.redis_cache == redis_cache


def test_router_handle_clientside_credential():
    deployment = {
        "model_name": "gemini/*",
        "litellm_params": {"model": "gemini/*"},
        "model_info": {
            "id": "1",
        },
    }
    router = Router(model_list=[deployment])

    new_deployment = router._handle_clientside_credential(
        deployment=deployment,
        kwargs={
            "api_key": "123",
            "metadata": {"model_group": "gemini/gemini-1.5-flash"},
        },
        function_name="acompletion",
    )

    assert new_deployment.litellm_params.api_key == "123"
    assert len(router.get_model_list()) == 2


def test_router_clientside_credential_per_key_scoped_selection():
    """
    Regression for the clientside-credential synthetic-deployment leak
    (root cause upstream PR #8966; upstream issue #17115; JuliaHub #22864).

    When a request supplies a clientside api_key, the router upserts a synthetic
    per-key deployment into the shared model_list under the public model group
    name. Without scoping, a later request could be load-balanced onto another
    caller's synthetic deployment and reuse their key. The fix scopes selection:
    a no-key request sees only config deployments; a keyed request sees config
    deployments plus only its own matching synthetic.

    Deterministic: asserts on the candidate set, independent of random.choice.
    """
    base = {
        "model_name": "claude-opus-4-8",
        "litellm_params": {
            "model": "anthropic/claude-opus-4-8",
            "api_key": "CONFIGKEY",
        },
        "model_info": {"id": "base-1"},
    }
    router = Router(model_list=[base])

    dep_a = router._handle_clientside_credential(
        deployment=base,
        kwargs={"api_key": "USERKEY_A", "metadata": {"model_group": "claude-opus-4-8"}},
        function_name="acompletion",
    )
    dep_b = router._handle_clientside_credential(
        deployment=base,
        kwargs={"api_key": "USERKEY_B", "metadata": {"model_group": "claude-opus-4-8"}},
        function_name="acompletion",
    )
    assert dep_a.litellm_params.api_key == "USERKEY_A"
    assert dep_b.litellm_params.api_key == "USERKEY_B"
    assert len(router.get_model_list()) == 3  # base + two synthetics persisted

    def candidate_keys(request_kwargs):
        _, healthy = router._common_checks_available_deployment(
            model="claude-opus-4-8", request_kwargs=request_kwargs
        )
        return sorted(d["litellm_params"].get("api_key") for d in healthy)

    # (1) NO-key request: ONLY the config deployment.
    assert candidate_keys({}) == ["CONFIGKEY"]
    # (2) Keyed (USERKEY_A): config + its OWN synthetic, NOT USERKEY_B's.
    assert candidate_keys({"api_key": "USERKEY_A"}) == ["CONFIGKEY", "USERKEY_A"]
    # (3) Symmetric for USERKEY_B.
    assert candidate_keys({"api_key": "USERKEY_B"}) == ["CONFIGKEY", "USERKEY_B"]


def test_router_prunes_stale_clientside_credential_deployments():
    """
    Synthetic per-key clientside-credential deployments unused beyond a TTL are
    evicted, bounding memory growth (JuliaHub #22864; upstream PR #8966 / #17115).
    Deterministic: backdates last-used timestamps rather than sleeping.
    """
    base = {
        "model_name": "claude-opus-4-8",
        "litellm_params": {
            "model": "anthropic/claude-opus-4-8",
            "api_key": "CONFIGKEY",
        },
        "model_info": {"id": "base-1"},
    }
    router = Router(model_list=[base])
    router.clientside_credential_ttl = 3600
    router._clientside_prune_interval = 0  # disable throttle for the test

    def mk(key):
        dep = router._handle_clientside_credential(
            deployment=base,
            kwargs={"api_key": key, "metadata": {"model_group": "claude-opus-4-8"}},
            function_name="acompletion",
        )
        return dep.model_info.id

    id_a, id_b = mk("USERKEY_A"), mk("USERKEY_B")
    assert len(router.get_model_list()) == 3

    router.clientside_credential_last_used[id_a] = 0.0  # epoch -> stale
    router.clientside_credential_last_used[id_b] = time.time()  # fresh
    router._maybe_prune_clientside_deployments()

    ids = {d["model_info"]["id"] for d in router.get_model_list()}
    assert id_a not in ids  # stale synthetic evicted
    assert id_b in ids and "base-1" in ids  # fresh synthetic + config kept
    assert id_a not in router.clientside_credential_last_used  # side dict cleaned


def test_router_clientside_credential_scoping_github_copilot(monkeypatch):
    """
    github_copilot flavor of the per-key scoping regression (JuliaHub #22864):
    per-user Copilot keys are injected as clientside api_key AND api_base, the
    config deployment carries no api_key at all, and two users can share the
    same api_base (same Copilot plan) while holding different keys.
    """
    # Router._add_deployment resolves the provider key at registration time;
    # without this, an unseeded authenticator would start the interactive
    # device-code login (and then fail Router init).
    monkeypatch.setenv("GITHUB_COPILOT_NON_INTERACTIVE", "1")
    base = {
        "model_name": "claude-sonnet-4.6",
        "litellm_params": {
            "model": "github_copilot/claude-sonnet-4.6",
            # no api_key: per-user credentials are injected per request
        },
        "model_info": {"id": "copilot-base-1", "base_model": "claude-sonnet-4-6"},
    }
    router = Router(model_list=[base])
    shared_base = "https://api.individual.githubcopilot.com"

    def mk(key):
        return router._handle_clientside_credential(
            deployment=base,
            kwargs={
                "api_key": key,
                "api_base": shared_base,
                "metadata": {"model_group": "claude-sonnet-4.6"},
            },
            function_name="acompletion",
        )

    dep_a, dep_b = mk("COPILOT_KEY_A"), mk("COPILOT_KEY_B")
    assert len(router.get_model_list()) == 3

    # synthetic deployments keep the config model_info (base_model pricing)
    # alongside the clientside_credential marker
    assert dep_a.model_info.base_model == "claude-sonnet-4-6"
    assert dep_a.model_info.clientside_credential is True

    def candidate_keys(request_kwargs):
        _, healthy = router._common_checks_available_deployment(
            model="claude-sonnet-4.6", request_kwargs=request_kwargs
        )
        return {
            d["litellm_params"].get("api_key") for d in healthy
        }  # config deployment contributes None

    # no-key request (e.g. token counting / health check): config only
    assert candidate_keys({}) == {None}
    # same api_base, different keys: each user sees only their own synthetic
    assert candidate_keys({"api_key": "COPILOT_KEY_A", "api_base": shared_base}) == {
        None,
        "COPILOT_KEY_A",
    }
    assert candidate_keys({"api_key": "COPILOT_KEY_B", "api_base": shared_base}) == {
        None,
        "COPILOT_KEY_B",
    }
    # key rotation (same user, new minted key): old synthetic no longer visible
    mk("COPILOT_KEY_A_ROTATED")
    assert candidate_keys(
        {"api_key": "COPILOT_KEY_A_ROTATED", "api_base": shared_base}
    ) == {None, "COPILOT_KEY_A_ROTATED"}


def test_router_get_async_openai_model_client():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {
                    "model": "gemini/*",
                    "api_base": "https://api.gemini.com",
                },
            }
        ]
    )
    model_client = router._get_async_openai_model_client(
        deployment=MagicMock(), kwargs={}
    )
    assert model_client is None


def test_router_get_deployment_credentials():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*", "api_key": "123"},
                "model_info": {"id": "1"},
            }
        ]
    )
    credentials = router.get_deployment_credentials(model_id="1")
    assert credentials is not None
    assert credentials["api_key"] == "123"


def test_router_get_deployment_credentials_with_provider():
    """
    Test that get_deployment_credentials_with_provider returns credentials with provider info.
    """
    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",
                "litellm_params": {
                    "model": "gpt-4o",
                    "api_key": "sk-test-123",
                    "api_base": "https://api.openai.com/v1",
                },
                "model_info": {"id": "openai-deployment-1"},
            },
            {
                "model_name": "claude-3",
                "litellm_params": {
                    "model": "anthropic/claude-3-sonnet",
                    "api_key": "sk-ant-123",
                },
                "model_info": {"id": "anthropic-deployment-1"},
            },
        ]
    )

    # Test getting credentials by model_id
    credentials = router.get_deployment_credentials_with_provider(
        model_id="openai-deployment-1"
    )
    assert credentials is not None
    assert credentials["api_key"] == "sk-test-123"
    assert credentials["custom_llm_provider"] == "openai"
    assert credentials["api_base"] == "https://api.openai.com/v1"

    # Test getting credentials by model_group_name (model_name)
    credentials2 = router.get_deployment_credentials_with_provider(model_id="claude-3")
    assert credentials2 is not None
    assert credentials2["api_key"] == "sk-ant-123"
    assert credentials2["custom_llm_provider"] == "anthropic"

    # Test with non-existent model
    credentials3 = router.get_deployment_credentials_with_provider(
        model_id="non-existent"
    )
    assert credentials3 is None


def test_router_get_deployment_credentials_with_provider_wildcard():
    """
    Test that get_deployment_credentials_with_provider handles wildcard patterns.

    When a model like openai/gpt-4o is requested and the config has openai/*,
    the method should resolve the wildcard pattern and return credentials.
    """
    router = Router(
        model_list=[
            {
                "model_name": "openai/*",
                "litellm_params": {
                    "model": "openai/*",
                    "api_key": "sk-wildcard-123",
                    "api_base": "https://api.openai.com/v1",
                },
                "model_info": {"id": "openai-wildcard-deployment"},
            },
            {
                "model_name": "anthropic/*",
                "litellm_params": {
                    "model": "anthropic/*",
                    "api_key": "sk-ant-wildcard-456",
                },
                "model_info": {"id": "anthropic-wildcard-deployment"},
            },
        ]
    )

    # Test wildcard pattern matching for OpenAI
    credentials = router.get_deployment_credentials_with_provider(
        model_id="openai/gpt-4o"
    )
    assert credentials is not None
    assert credentials["api_key"] == "sk-wildcard-123"
    assert credentials["custom_llm_provider"] == "openai"
    assert credentials["api_base"] == "https://api.openai.com/v1"

    # Test wildcard pattern matching for Anthropic
    credentials2 = router.get_deployment_credentials_with_provider(
        model_id="anthropic/claude-3-opus"
    )
    assert credentials2 is not None
    assert credentials2["api_key"] == "sk-ant-wildcard-456"
    assert credentials2["custom_llm_provider"] == "anthropic"

    # Test with non-matching model
    credentials3 = router.get_deployment_credentials_with_provider(
        model_id="vertex_ai/gemini-pro"
    )
    assert credentials3 is None


def test_router_get_deployment_model_info():
    router = Router(
        model_list=[
            {
                "model_name": "gemini/*",
                "litellm_params": {"model": "gemini/*"},
                "model_info": {"id": "1"},
            }
        ]
    )
    model_info = router.get_deployment_model_info(
        model_id="1", model_name="gemini/gemini-1.5-flash"
    )
    assert model_info is not None
