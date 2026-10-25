"""Automated unit and integration test suite for the Incident Digest API (v0.1).

Covers contract validation, Bearer authentication, tail-preserving 20,000-char
truncation, Markdown fence sanitization, and all documented edge cases.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# Ensure the project root is importable when running `pytest` directly
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import llm_client
from llm_client import LLMProviderError, LLMTimeoutError
from main import app
from schemas import MAX_DESCRIPTION_LENGTH
from tools import get_service_runbook

TEST_BEARER_TOKEN = "test-static-bearer-token-12345"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_BEARER_TOKEN}"}
ENDPOINT = "/api/v1/digest"


@pytest.fixture(autouse=True)
def configure_test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set deterministic environment variables for all tests."""
    monkeypatch.setenv("API_BEARER_TOKEN", TEST_BEARER_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "test-gemini-api-key")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "60")
    monkeypatch.setenv("LLM_MAX_TOKENS", "500")


@pytest_asyncio.fixture
async def api_client() -> AsyncClient:
    """Provide an async HTTP client wired directly to the FastAPI ASGI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        yield client


# ============================================================================
# 1. Contract & Authentication Tests
# ============================================================================


@pytest.mark.asyncio
async def test_valid_payload_returns_200_ok(api_client: AsyncClient) -> None:
    """200 OK: Valid payload returns structured summary and recommended_action."""
    mock_llm_response = json.dumps(
        {
            "summary": "Primary PostgreSQL database connection pool is exhausted due to slow queries.",
            "recommended_action": "Inspect active locks in pg_stat_activity and terminate long-running queries.",
        }
    )

    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(return_value=mock_llm_response),
    ) as mock_generate:
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "High DB Connection Utilization on prod-db-01",
                "severity": "P1",
                "description": "FATAL: remaining connection slots are reserved for non-replication superuser connections",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "Primary PostgreSQL database connection pool is exhausted due to slow queries.",
        "recommended_action": "Inspect active locks in pg_stat_activity and terminate long-running queries.",
    }
    mock_generate.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": ""},
        {"Authorization": "Bearer wrong-token"},
        {"Authorization": "Basic dXNlcjpwYXNz"},
        {"Authorization": "Bearer"},
    ],
)
async def test_missing_or_invalid_bearer_token_returns_401(
    api_client: AsyncClient,
    headers: dict[str, str],
) -> None:
    """401 Unauthorized: Missing, malformed, or invalid bearer token returns 401 (not 403)."""
    response = await api_client.post(
        ENDPOINT,
        headers=headers,
        json={
            "title": "Redis cluster node unreachable",
            "description": "Connection refused on port 6379",
        },
    )

    assert response.status_code == 401


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "invalid_payload",
    [
        # Missing title
        {"description": "Stack trace here"},
        # Blank / whitespace-only title
        {"title": "   ", "description": "Stack trace here"},
        # Missing description
        {"title": "API Gateway 5xx spike"},
        # Invalid field type for title
        {"title": 12345, "description": "Stack trace here"},
        # Invalid field type for description
        {"title": "API Gateway 5xx spike", "description": ["not", "a", "string"]},
        # Invalid field type for severity
        {"title": "API Gateway 5xx spike", "severity": 1, "description": "logs"},
        # Unexpected extra field
        {
            "title": "API Gateway 5xx spike",
            "description": "logs",
            "unexpected_field": "disallowed",
        },
    ],
)
async def test_invalid_schema_returns_400_bad_request(
    api_client: AsyncClient,
    invalid_payload: dict[str, object],
) -> None:
    """400 Bad Request: Missing required fields or invalid field types return 400 (not 422)."""
    response = await api_client.post(
        ENDPOINT,
        headers=AUTH_HEADERS,
        json=invalid_payload,
    )

    assert response.status_code == 400


@pytest.mark.asyncio
async def test_malformed_json_body_returns_400_bad_request(
    api_client: AsyncClient,
) -> None:
    """400 Bad Request: Unparseable raw JSON body returns 400."""
    response = await api_client.post(
        ENDPOINT,
        headers={**AUTH_HEADERS, "Content-Type": "application/json"},
        content="{malformed_json: true",
    )

    assert response.status_code == 400


# ============================================================================
# 2. Functional & Edge Case Tests
# ============================================================================


@pytest.mark.asyncio
async def test_20000_char_tail_preserving_truncation(api_client: AsyncClient) -> None:
    """20,000-Char Truncation: 25,000-char description silently keeps only the last 20,000 chars."""
    prefix_marker = "OLD_LOG_HEADER_"
    dropped_prefix = prefix_marker + ("A" * (5_000 - len(prefix_marker)))
    tail_marker = "_MOST_RECENT_STACK_TRACE_LINE_AT_END"
    preserved_tail = ("B" * (MAX_DESCRIPTION_LENGTH - len(tail_marker))) + tail_marker
    assert len(dropped_prefix) == 5_000
    assert len(preserved_tail) == MAX_DESCRIPTION_LENGTH

    full_description = dropped_prefix + preserved_tail
    assert len(full_description) == 25_000

    mock_llm_response = json.dumps(
        {
            "summary": "Worker crashed with NullPointerException at the end of trace.",
            "recommended_action": "Roll back deployment v2.4.1 and inspect null payload handling.",
        }
    )

    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(return_value=mock_llm_response),
    ) as mock_generate:
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "Order processor worker crash",
                "severity": "SEV-1",
                "description": full_description,
            },
        )

    assert response.status_code == 200
    mock_generate.assert_awaited_once()

    sent_payload = mock_generate.call_args.kwargs["incident_payload"]
    assert len(sent_payload["description"]) == MAX_DESCRIPTION_LENGTH
    assert sent_payload["description"] == preserved_tail
    assert prefix_marker not in sent_payload["description"]
    assert sent_payload["description"].endswith(tail_marker)


@pytest.mark.asyncio
async def test_markdown_code_fence_stripping(api_client: AsyncClient) -> None:
    """Markdown Stripping: LLM output wrapped in ```json ... ``` is sanitized into a clean 200 OK JSON response."""
    fenced_llm_output = (
        "```json\n"
        '{\n  "summary": "TLS certificate expired on ingress-nginx",\n'
        '  "recommended_action": "Renew certificate via cert-manager and reload nginx pods"\n}\n'
        "```"
    )

    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(return_value=fenced_llm_output),
    ):
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "SSL Handshake Failed on api.example.com",
                "description": "x509: certificate has expired or is not yet valid",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "TLS certificate expired on ingress-nginx",
        "recommended_action": "Renew certificate via cert-manager and reload nginx pods",
    }


@pytest.mark.asyncio
async def test_empty_description_edge_case_1(api_client: AsyncClient) -> None:
    """Edge Case 1: Empty description (\"\") with a valid title succeeds with 200 OK."""
    mock_llm_response = json.dumps(
        {
            "summary": "High memory utilization alert triggered on payment-service with no log body attached.",
            "recommended_action": "Check pod memory metrics and recent deployment history for payment-service.",
        }
    )

    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(return_value=mock_llm_response),
    ) as mock_generate:
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "payment-service memory usage > 95%",
                "severity": "P2",
                "description": "",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "High memory utilization alert triggered on payment-service with no log body attached.",
        "recommended_action": "Check pod memory metrics and recent deployment history for payment-service.",
    }
    sent_payload = mock_generate.call_args.kwargs["incident_payload"]
    assert sent_payload["description"] == ""


@pytest.mark.asyncio
async def test_gibberish_input_edge_case_2(api_client: AsyncClient) -> None:
    """Edge Case 2: Gibberish/irrelevant input returns the exact safe fallback structure."""
    fallback_llm_response = json.dumps(
        {
            "summary": "Invalid incident data",
            "recommended_action": "None",
        }
    )

    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(return_value=fallback_llm_response),
    ):
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "asdfghjkl qwerty 12345",
                "description": "What is the best recipe for chocolate chip cookies?",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "summary": "Invalid incident data",
        "recommended_action": "None",
    }


@pytest.mark.asyncio
async def test_upstream_provider_error_returns_502_edge_case_3(
    api_client: AsyncClient,
) -> None:
    """Edge Case 3 (Provider Outage): Upstream LLMProviderError maps to HTTP 502 Bad Gateway."""
    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(side_effect=LLMProviderError("Upstream Gemini 503 Service Unavailable")),
    ):
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "Kafka consumer lag increasing",
                "description": "Consumer group billing-cg lag exceeded 50,000 messages",
            },
        )

    assert response.status_code == 502
    assert "detail" in response.json()


@pytest.mark.asyncio
async def test_unparseable_llm_output_returns_502_edge_case_3(
    api_client: AsyncClient,
) -> None:
    """Edge Case 3 (Format Guarantee): Non-JSON or schema-violating LLM output maps to HTTP 502."""
    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(return_value="I apologize, I cannot process this alert right now."),
    ):
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "Kafka consumer lag increasing",
                "description": "Consumer group billing-cg lag exceeded 50,000 messages",
            },
        )

    assert response.status_code == 502


@pytest.mark.asyncio
async def test_upstream_timeout_returns_504_edge_case_3(
    api_client: AsyncClient,
) -> None:
    """Edge Case 3 (Timeout > 60s): Upstream LLMTimeoutError maps to HTTP 504 Gateway Timeout."""
    with patch(
        "llm_client.generate_digest_raw",
        new=AsyncMock(
            side_effect=LLMTimeoutError("Upstream LLM request timed out after 60.0 seconds.")
        ),
    ):
        response = await api_client.post(
            ENDPOINT,
            headers=AUTH_HEADERS,
            json={
                "title": "Kubernetes node NotReady",
                "description": "kubelet stopped posting node status",
            },
        )

    assert response.status_code == 504
    assert "timed out" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_llm_client_enforces_timeout_and_token_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify llm_client.generate_digest_raw enforces max_output_tokens=500 and raises LLMTimeoutError on slow calls."""
    monkeypatch.setenv("LLM_TIMEOUT_SECONDS", "0.05")
    monkeypatch.setenv("LLM_MAX_TOKENS", "500")

    async def simulated_slow_gemini_call(**kwargs: object) -> SimpleNamespace:
        await asyncio.sleep(0.2)
        return SimpleNamespace(text='{"summary": "late", "recommended_action": "none"}')

    mock_genai_client = MagicMock()
    mock_genai_client.aio.models.generate_content = AsyncMock(
        side_effect=simulated_slow_gemini_call
    )

    await llm_client.init_mcp_client()
    try:
        with patch("llm_client._create_genai_client", return_value=mock_genai_client):
            with pytest.raises(LLMTimeoutError):
                await llm_client.generate_digest_raw(
                    system_prompt="System prompt",
                    incident_payload={"title": "Test", "description": "Test"},
                )
    finally:
        await llm_client.close_mcp_client()

    call_kwargs = mock_genai_client.aio.models.generate_content.call_args.kwargs
    assert call_kwargs["config"].max_output_tokens == 500
    discovered_tool_names = [
        decl.name
        for tool in (call_kwargs["config"].tools or [])
        for decl in (getattr(tool, "function_declarations", None) or [])
    ]
    assert "get_service_runbook" in discovered_tool_names


@pytest.mark.asyncio
async def test_llm_client_handles_tool_call_and_returns_final_response() -> None:
    """Verify llm_client executes get_service_runbook via the persistent MCP client and returns the final response."""
    first_response = SimpleNamespace(
        function_calls=[
            SimpleNamespace(name="get_service_runbook", args={"service_name": "payments"})
        ],
        candidates=[
            SimpleNamespace(
                content=SimpleNamespace(
                    role="model",
                    parts=[
                        SimpleNamespace(
                            function_call=SimpleNamespace(
                                name="get_service_runbook",
                                args={"service_name": "payments"},
                            )
                        )
                    ],
                )
            )
        ],
        text=None,
    )
    second_response = SimpleNamespace(
        function_calls=None,
        candidates=[],
        text=json.dumps(
            {
                "summary": "Payment service gateway pod is failing.",
                "recommended_action": "Restart the payment-gateway pod and check Stripe API status.",
            }
        ),
    )

    mock_genai_client = MagicMock()
    mock_genai_client.aio.models.generate_content = AsyncMock(
        side_effect=[first_response, second_response]
    )

    from main import lifespan

    async with lifespan(app):
        assert llm_client.get_mcp_client() is not None
        assert getattr(app.state, "mcp_session", None) is not None

        with patch("llm_client._create_genai_client", return_value=mock_genai_client):
            result = await llm_client.generate_digest_raw(
                system_prompt="System prompt",
                incident_payload={
                    "title": "Payments service outage",
                    "description": "502 errors from payment-gateway",
                },
            )

    assert llm_client.get_mcp_client() is None
    assert getattr(app.state, "mcp_session", None) is None

    assert json.loads(result) == {
        "summary": "Payment service gateway pod is failing.",
        "recommended_action": "Restart the payment-gateway pod and check Stripe API status.",
    }
    assert mock_genai_client.aio.models.generate_content.await_count == 2

    second_call_kwargs = mock_genai_client.aio.models.generate_content.call_args_list[1].kwargs
    conversation_contents = second_call_kwargs["contents"]
    assert isinstance(conversation_contents, list)
    assert len(conversation_contents) == 3
    tool_response_part = conversation_contents[2].parts[0]
    assert tool_response_part.function_response.name == "get_service_runbook"
    assert (
        tool_response_part.function_response.response["result"]
        == get_service_runbook("payments")
    )
