"""
Tests for configurable timeout on pass-through endpoints.
"""

import sys
from unittest.mock import AsyncMock, patch, MagicMock
from typing import Optional

import pytest

from litellm.proxy._types import PassThroughGenericEndpoint


class TestPassThroughGenericEndpointTimeout:
    """Test that PassThroughGenericEndpoint accepts and defaults the timeout field."""

    def test_timeout_defaults_to_none(self):
        endpoint = PassThroughGenericEndpoint(
            path="/test",
            target="https://example.com",
        )
        assert endpoint.timeout is None

    def test_timeout_accepts_custom_value(self):
        endpoint = PassThroughGenericEndpoint(
            path="/test",
            target="https://example.com",
            timeout=1200.0,
        )
        assert endpoint.timeout == 1200.0

    def test_timeout_included_in_model_dump(self):
        endpoint = PassThroughGenericEndpoint(
            path="/test",
            target="https://example.com",
            timeout=900,
        )
        dumped = endpoint.model_dump()
        assert dumped["timeout"] == 900

    def test_timeout_none_in_model_dump(self):
        endpoint = PassThroughGenericEndpoint(
            path="/test",
            target="https://example.com",
        )
        dumped = endpoint.model_dump()
        assert dumped["timeout"] is None


def _make_mock_proxy_server_module():
    """Create a fake proxy_server module with a mock proxy_logging_obj."""
    mock_module = MagicMock()
    mock_module.proxy_logging_obj = MagicMock()
    mock_module.proxy_logging_obj.pre_call_hook = AsyncMock(return_value={"test": True})
    return mock_module


class TestPassThroughRequestTimeout:
    """Test that the timeout value flows through to get_async_httpx_client."""

    @pytest.mark.asyncio
    async def test_custom_timeout_passed_to_httpx_client(self):
        """Verify that a custom timeout is forwarded to get_async_httpx_client."""
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.headers = MagicMock()
        mock_request.headers.items.return_value = []
        mock_request.headers.get.return_value = None
        mock_request.query_params = {}
        mock_request.body = AsyncMock(return_value=b'{"test": true}')

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"
        mock_user_api_key_dict.user_id = None
        mock_user_api_key_dict.team_id = None
        mock_user_api_key_dict.end_user_id = None

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = "{}"
        mock_response.iter_bytes = MagicMock(return_value=iter([b"{}"]))
        mock_client.send = AsyncMock(return_value=mock_response)

        mock_client_obj = MagicMock()
        mock_client_obj.client = mock_client

        mock_proxy_module = _make_mock_proxy_server_module()
        proxy_server_key = "litellm.proxy.proxy_server"
        original_module = sys.modules.get(proxy_server_key)
        sys.modules[proxy_server_key] = mock_proxy_module

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
                return_value=mock_client_obj,
            ) as mock_get_client:
                try:
                    await pass_through_request(
                        request=mock_request,
                        target="https://example.com/api",
                        custom_headers={"Authorization": "Bearer test"},
                        user_api_key_dict=mock_user_api_key_dict,
                        timeout=1200,
                    )
                except Exception:
                    # We only care that get_async_httpx_client was called with the right timeout
                    pass

                mock_get_client.assert_called_once()
                call_kwargs = mock_get_client.call_args
                assert call_kwargs.kwargs["params"]["timeout"] == 1200
        finally:
            if original_module is not None:
                sys.modules[proxy_server_key] = original_module
            else:
                sys.modules.pop(proxy_server_key, None)

    @pytest.mark.asyncio
    async def test_default_timeout_when_none(self):
        """Verify that timeout defaults to 600 when not specified."""
        from litellm.proxy.pass_through_endpoints.pass_through_endpoints import (
            pass_through_request,
        )

        mock_request = MagicMock()
        mock_request.method = "POST"
        mock_request.headers = MagicMock()
        mock_request.headers.items.return_value = []
        mock_request.headers.get.return_value = None
        mock_request.query_params = {}
        mock_request.body = AsyncMock(return_value=b'{"test": true}')

        mock_user_api_key_dict = MagicMock()
        mock_user_api_key_dict.api_key = "test-key"
        mock_user_api_key_dict.user_id = None
        mock_user_api_key_dict.team_id = None
        mock_user_api_key_dict.end_user_id = None

        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.headers = {}
        mock_response.text = "{}"
        mock_client.send = AsyncMock(return_value=mock_response)

        mock_client_obj = MagicMock()
        mock_client_obj.client = mock_client

        mock_proxy_module = _make_mock_proxy_server_module()
        proxy_server_key = "litellm.proxy.proxy_server"
        original_module = sys.modules.get(proxy_server_key)
        sys.modules[proxy_server_key] = mock_proxy_module

        try:
            with patch(
                "litellm.proxy.pass_through_endpoints.pass_through_endpoints.get_async_httpx_client",
                return_value=mock_client_obj,
            ) as mock_get_client:
                try:
                    await pass_through_request(
                        request=mock_request,
                        target="https://example.com/api",
                        custom_headers={"Authorization": "Bearer test"},
                        user_api_key_dict=mock_user_api_key_dict,
                    )
                except Exception:
                    pass

                mock_get_client.assert_called_once()
                call_kwargs = mock_get_client.call_args
                assert call_kwargs.kwargs["params"]["timeout"] == 600
        finally:
            if original_module is not None:
                sys.modules[proxy_server_key] = original_module
            else:
                sys.modules.pop(proxy_server_key, None)
