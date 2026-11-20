import json
from pathlib import Path
from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest
import respx

import litellm
from litellm.exceptions import AuthenticationError
from litellm.llms.github_copilot.common_utils import DEFAULT_GITHUB_COPILOT_API_BASE, GetAccessTokenError
from litellm.llms.github_copilot.messages.transformation import (
    GithubCopilotAnthropicMessagesConfig,
)
from litellm.llms.github_copilot.per_user_auth import (
    GITHUB_COPILOT_USER_SESSION_KWARG_KEY,
    GithubCopilotUserSession,
)


@pytest.mark.parametrize("api_base", [None, "https://caller.example/", "https://caller.example/v1/messages"])
def test_messages_caller_key_never_uses_stored_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_base: str | None
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_COPILOT_NON_INTERACTIVE", "1")
    (tmp_path / "access-token").write_text("saved-token")
    config: Final = GithubCopilotAnthropicMessagesConfig()
    with respx.mock(assert_all_called=False) as http:
        discovery: Final = http.get("https://api.github.com/copilot_internal/user").respond(
            json={"endpoints": {"api": "https://api.business.githubcopilot.com"}}
        )
        headers, base = config.validate_anthropic_messages_environment(
            headers={},
            model="github_copilot/claude-opus-5.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key="caller-token",
            api_base=api_base,
        )
        url: Final = config.get_complete_url(base, "caller-token", "claude-opus-5.5", {}, {})
        direct_url: Final = config.get_complete_url(api_base, "caller-token", "claude-opus-5.5", {}, {})
    assert headers["Authorization"] == "Bearer caller-token"
    assert (
        url
        == direct_url
        == (
            f"{DEFAULT_GITHUB_COPILOT_API_BASE}/v1/messages"
            if api_base is None
            else "https://caller.example/v1/messages"
        )
    )
    assert not discovery.called


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_messages_blank_key_keeps_stored_credentials_and_ignores_caller_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_key: str | None
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_COPILOT_NON_INTERACTIVE", "1")
    (tmp_path / "access-token").write_text(f"stored-{tmp_path.name}")
    config: Final = GithubCopilotAnthropicMessagesConfig()
    with respx.mock as http:
        http.get("https://api.github.com/copilot_internal/user").respond(
            json={"endpoints": {"api": "https://api.business.githubcopilot.com/"}}
        )
        headers, base = config.validate_anthropic_messages_environment(
            headers={},
            model="github_copilot/claude-opus-5.5",
            messages=[],
            optional_params={},
            litellm_params={},
            api_key=api_key,
            api_base="https://caller.example",
        )
    assert headers["Authorization"] == f"Bearer stored-{tmp_path.name}"
    assert config.get_complete_url(base, api_key, "claude-opus-5.5", {}, {}) == (
        "https://api.business.githubcopilot.com/v1/messages"
    )


@pytest.mark.parametrize("api_key", [None, "", "   "])
def test_messages_without_caller_key_or_login_still_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, api_key: str | None
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_COPILOT_NON_INTERACTIVE", "1")
    with respx.mock:
        with pytest.raises(AuthenticationError, match="GITHUB_COPILOT_NON_INTERACTIVE"):
            GithubCopilotAnthropicMessagesConfig().validate_anthropic_messages_environment(
                headers={},
                model="github_copilot/claude-opus-5.5",
                messages=[],
                optional_params={},
                litellm_params={},
                api_key=api_key,
                api_base=None,
            )


@pytest.mark.asyncio
@pytest.mark.parametrize("use_sdk_wrapper", [False, True])
@pytest.mark.parametrize("stream", [False, True])
async def test_messages_public_requests_use_caller_key_without_login(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, use_sdk_wrapper: bool, stream: bool
) -> None:
    monkeypatch.setenv("GITHUB_COPILOT_TOKEN_DIR", str(tmp_path))
    monkeypatch.setenv("GITHUB_COPILOT_NON_INTERACTIVE", "1")
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    payload: Final = {
        "id": "msg-caller",
        "type": "message",
        "role": "assistant",
        "content": [{"type": "text", "text": "hi"}],
        "model": "claude-opus-5.5",
        "stop_reason": "end_turn",
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }
    events: Final = (
        {"type": "message_start", "message": {**payload, "content": [], "stop_reason": None}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
        {"type": "message_stop"},
    )
    sse: Final = "".join(f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events).encode()
    request: Final = {
        "model": "github_copilot/claude-opus-5.5",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "api_key": "caller-token",
        "api_base": "https://caller.example",
        "stream": stream,
    }
    with respx.mock as http:
        upstream: Final = http.post("https://caller.example/v1/messages").respond(
            content=sse if stream else json.dumps(payload).encode(),
            headers={"content-type": "text/event-stream" if stream else "application/json"},
        )
        result: Final = (
            await litellm.anthropic.messages.acreate(**request)
            if use_sdk_wrapper
            else await litellm.anthropic_messages(**request)
        )
        if stream:
            body: Final = b"".join([chunk async for chunk in result])
            assert b'"text": "hi"' in body
            assert b"message_stop" in body
        else:
            assert result["id"] == payload["id"]
            assert result["content"] == payload["content"]
    assert upstream.calls.last.request.headers["authorization"] == "Bearer caller-token"
    assert json.loads(upstream.calls.last.request.content)["stream"] == stream


def test_github_copilot_anthropic_messages_get_complete_url():
    """Build the /v1/messages URL from the validated base."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_base.return_value = None

    # No api_base supplied and no authenticator base -> default Copilot endpoint.
    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.githubcopilot.com/v1/messages"
    # Falls back to a single authenticator read, not a hard-coded second one.
    config.authenticator.get_api_base.assert_called()

    # The resolved (validated) base passed in is reused verbatim; no extra read.
    config.authenticator.get_api_base.reset_mock()
    url = config.get_complete_url(
        api_base="https://api.business.githubcopilot.com",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.business.githubcopilot.com/v1/messages"
    config.authenticator.get_api_base.assert_not_called()

    # A trailing slash on the base must not produce a double-slash URL.
    url = config.get_complete_url(
        api_base="https://api.business.githubcopilot.com/",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.business.githubcopilot.com/v1/messages"

    # An already-complete /v1/messages base is left untouched.
    url = config.get_complete_url(
        api_base="https://api.githubcopilot.com/v1/messages",
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.githubcopilot.com/v1/messages"


def test_github_copilot_anthropic_messages_get_complete_url_normalizes_authenticator_trailing_slash():
    """A tenant base with a trailing slash from the authenticator fallback must
    not yield a double-slash URL."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_base.return_value = "https://api.business.githubcopilot.com/"

    url = config.get_complete_url(
        api_base=None,
        api_key=None,
        model="github_copilot/claude-haiku-4.5",
        optional_params={},
        litellm_params={},
    )
    assert url == "https://api.business.githubcopilot.com/v1/messages"


def test_github_copilot_anthropic_messages_validate_environment():
    """Test environment validation and header injection."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key-123"
    config.authenticator.get_api_base.return_value = None

    headers = {}
    # Pass a hostile api_base to confirm it is ignored.
    validated_headers, api_base = config.validate_anthropic_messages_environment(
        headers=headers,
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base="https://attacker.example.com",
    )

    assert "copilot-integration-id" in validated_headers
    assert validated_headers["copilot-integration-id"] == "vscode-chat"
    assert "Authorization" in validated_headers
    assert "anthropic-version" in validated_headers
    assert validated_headers["anthropic-version"] == "2023-06-01"
    # /v1/messages must use the messages-proxy intent so the Copilot backend
    # enables Anthropic-native features (context_management, thinking, etc.).
    assert validated_headers["openai-intent"] == "messages-proxy"
    assert validated_headers["x-interaction-type"] == "messages-proxy"
    assert validated_headers["x-github-api-version"] == "2026-06-01"
    assert api_base == "https://api.githubcopilot.com"


def test_github_copilot_anthropic_messages_validate_environment_injects_beta_headers():
    """Anthropic-beta headers must be auto-injected for advanced features
    (context_management, output_format, etc.) — matches the parent
    AnthropicMessagesConfig contract."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = None

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"output_format": {"type": "json_object"}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "anthropic-beta" in validated_headers
    assert "structured-outputs-2025-11-13" in validated_headers["anthropic-beta"]


def test_github_copilot_anthropic_messages_validate_environment_preserves_caller_anthropic_version():
    """Caller-supplied anthropic-version must be forwarded verbatim."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = None

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={"anthropic-version": "2024-10-22"},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["anthropic-version"] == "2024-10-22"


def test_github_copilot_anthropic_messages_validate_environment_injects_context_management_beta():
    """context_management in optional_params must trigger the corresponding
    anthropic-beta header so the Copilot backend accepts the field."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = None

    validated_headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["openai-intent"] == "messages-proxy"
    assert validated_headers["x-interaction-type"] == "messages-proxy"
    assert "anthropic-beta" in validated_headers
    assert "context-management-2025-06-27" in validated_headers["anthropic-beta"]


@pytest.mark.usefixtures("local_beta_headers_config")
def test_github_copilot_safeguards_do_not_derive_the_dangerous_tool_use_beta() -> None:
    session: Final = GithubCopilotUserSession(token="gh.test-key", api_base="https://api.githubcopilot.com")

    validated_headers, _ = GithubCopilotAnthropicMessagesConfig().validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={
            "safeguards": ({"type": "dangerous_tool_use"},),
            "context_management": {"edits": ({"type": "clear_tool_uses_20250919"},)},
        },
        litellm_params={GITHUB_COPILOT_USER_SESSION_KWARG_KEY: session},
        api_key=None,
        api_base=None,
    )

    assert validated_headers["anthropic-beta"] == "context-management-2025-06-27"


def test_github_copilot_anthropic_messages_validate_environment_auth_error():
    """Test error handling when authentication fails."""
    config = GithubCopilotAnthropicMessagesConfig()

    # Mock the authenticator to raise an error
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = GetAccessTokenError(
        status_code=401, message="No valid API key found"
    )

    with pytest.raises(AuthenticationError):
        config.validate_anthropic_messages_environment(
            headers={},
            model="github_copilot/claude-haiku-4.5",
            messages=[{"role": "user", "content": "Hello"}],
            optional_params={},
            litellm_params={},
            api_key=None,
            api_base=None,
        )


def test_github_copilot_anthropic_messages_supported_params():
    """Test supported parameters list."""
    config = GithubCopilotAnthropicMessagesConfig()
    params = config.get_supported_anthropic_messages_params("github_copilot/claude-haiku-4.5")

    # Should inherit from AnthropicMessagesConfig
    assert "messages" in params
    assert "model" in params
    assert "max_tokens" in params
    assert "thinking" in params


def test_provider_config_manager_dispatches_claude_to_copilot_messages_config():
    """ProviderConfigManager must return the Copilot Anthropic Messages config
    for Claude models served via github_copilot."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="github_copilot/claude-haiku-4.5",
        provider=LlmProviders.GITHUB_COPILOT,
    )

    assert isinstance(config, GithubCopilotAnthropicMessagesConfig)


def test_provider_config_manager_skips_non_claude_copilot_models():
    """Non-Claude github_copilot models (e.g. gpt-*) must not be routed through
    the Anthropic Messages dispatch."""
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    config = ProviderConfigManager.get_provider_anthropic_messages_config(
        model="github_copilot/gpt-5-mini",
        provider=LlmProviders.GITHUB_COPILOT,
    )

    assert config is None


def test_github_copilot_anthropic_messages_validate_environment_normalizes_trailing_slash():
    """A tenant base with a trailing slash from the authenticator must be
    normalized so the URL built downstream has no double slash."""
    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = "https://api.business.githubcopilot.com/"

    _, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={},
        litellm_params={},
        api_key=None,
        api_base="https://attacker.example.com",
    )

    assert api_base == "https://api.business.githubcopilot.com"


def test_github_copilot_config_disables_anthropic_beta_filtering():
    """Copilot's /v1/messages is a native Anthropic passthrough, so injected
    anthropic-beta values (context_management, structured outputs, ...) must be
    forwarded verbatim. The default provider-scoped filter would drop them
    because github_copilot has no entry in the beta headers config; a regression
    here would silently disable header-gated Anthropic features for Copilot."""
    from litellm.anthropic_beta_headers_manager import update_headers_with_filtered_beta
    from litellm.llms.azure_ai.anthropic.messages_transformation import AzureAnthropicMessagesConfig

    config = GithubCopilotAnthropicMessagesConfig()
    assert config.should_filter_anthropic_beta_headers() is False
    assert AzureAnthropicMessagesConfig().should_filter_anthropic_beta_headers() is True

    config.authenticator = MagicMock()
    config.authenticator.get_api_key.return_value = "gh.test-key"
    config.authenticator.get_api_base.return_value = None

    headers, _ = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-haiku-4.5",
        messages=[{"role": "user", "content": "Hello"}],
        optional_params={"context_management": {"edits": [{"type": "clear_tool_uses_20250919"}]}},
        litellm_params={},
        api_key=None,
        api_base=None,
    )

    assert "context-management-2025-06-27" in headers["anthropic-beta"]

    # The override is load-bearing: had the config opted into the provider-scoped
    # filter, the handler would have run it and dropped every value, since
    # github_copilot has no mapping. Prove that here so a regression that flips
    # should_filter back on is caught as the silent feature breakage it causes.
    stripped = update_headers_with_filtered_beta(headers=dict(headers), provider="github_copilot")
    assert "anthropic-beta" not in stripped


def test_github_copilot_config_does_not_handle_web_search_natively():
    """Copilot's /v1/messages does not run web_search, so its config must report
    handles_web_search_natively() == False. This is what keeps the web-search
    interception handler short-circuiting Copilot instead of routing to it, even
    though Copilot now has a BaseAnthropicMessagesConfig. The base Anthropic
    config (bedrock/vertex/anthropic path) must report True."""
    from litellm.llms.anthropic.messages.native.transformation import (
        AnthropicMessagesConfig,
    )

    assert GithubCopilotAnthropicMessagesConfig().handles_web_search_natively() is False
    assert AnthropicMessagesConfig().handles_web_search_natively() is True


def test_github_copilot_messages_config_probes_capabilities_under_copilot_namespace():
    """Capability probes in the shared pass-through helpers read
    ``self.custom_llm_provider``; without this override they probed the
    ``anthropic`` namespace and ignored the exact ``github_copilot/claude-*``
    cost-map entries."""
    assert GithubCopilotAnthropicMessagesConfig().custom_llm_provider == "github_copilot"


def test_validate_environment_uses_per_user_session_and_skips_authenticator():
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_key.side_effect = AssertionError("shared authenticator must not run")
    config.authenticator.get_api_base.side_effect = AssertionError("shared authenticator must not run")

    session = GithubCopilotUserSession(token="user-copilot-token", api_base="https://tenant.githubcopilot.com/")
    headers, api_base = config.validate_anthropic_messages_environment(
        headers={},
        model="github_copilot/claude-sonnet-5.5",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
        api_key="caller-token",
        api_base="https://caller.example",
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"
    assert api_base == "https://tenant.githubcopilot.com"
    config.authenticator.get_api_key.assert_not_called()


def test_per_user_session_token_wins_over_caller_authorization():
    from litellm.llms.github_copilot.per_user_auth import GithubCopilotUserSession

    config = GithubCopilotAnthropicMessagesConfig()
    config.authenticator = MagicMock()
    config.authenticator.get_api_base.return_value = "https://api.githubcopilot.com"
    session = GithubCopilotUserSession(token="user-copilot-token", api_base="https://api.githubcopilot.com")
    headers, _ = config.validate_anthropic_messages_environment(
        headers={"authorization": "Bearer caller-token"},
        model="github_copilot/claude-sonnet-4.5",
        messages=[],
        optional_params={},
        litellm_params={"github_copilot_user_session": session},
    )
    assert headers["Authorization"] == "Bearer user-copilot-token"


def test_transform_response_carries_upstream_usage():
    config = GithubCopilotAnthropicMessagesConfig()
    raw = httpx.Response(
        200,
        json={
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "github_copilot/claude-sonnet-5.5",
            "content": [{"type": "text", "text": "hi"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 14, "output_tokens": 3},
        },
    )
    result = config.transform_anthropic_messages_response(
        model="github_copilot/claude-sonnet-5.5",
        raw_response=raw,
        logging_obj=MagicMock(),
    )
    assert result["usage"]["input_tokens"] == 14
    assert result["usage"]["output_tokens"] == 3
