"""Isolated asynchronous LLM adapter (Layer 4) for Google Gemini.

Encapsulates vendor SDK calls, token limits, retry policy, and strict request
timeouts while exposing a provider-agnostic interface and exception hierarchy
to the API router.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

load_dotenv()

DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
DEFAULT_TIMEOUT_SECONDS: float = 60.0
DEFAULT_MAX_TOKENS: int = 500
MAX_RETRY_ATTEMPTS: int = 2
MAX_TOOL_CALL_TURNS: int = 5
MCP_SERVER_SCRIPT_PATH: Path = Path(__file__).resolve().parent / "mcp_server.py"

_STANDARD_LOG_RECORD_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)


class StructuredJSONFormatter(logging.Formatter):
    """Formats Python log records as single-line structured JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        log_payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _STANDARD_LOG_RECORD_ATTRS and not key.startswith("_"):
                log_payload[key] = value

        if record.exc_info:
            log_payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_payload, default=str, ensure_ascii=False)


def configure_json_logger(name: str) -> logging.Logger:
    """Create or return a module logger configured with structured JSON output."""
    module_logger = logging.getLogger(name)
    if not module_logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(StructuredJSONFormatter())
        module_logger.addHandler(handler)
    module_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
    module_logger.propagate = True
    return module_logger


logger = configure_json_logger(__name__)


class LLMClientError(Exception):
    """Base exception for all LLM adapter errors."""


class LLMTimeoutError(LLMClientError):
    """Raised when the upstream LLM request exceeds the 60-second timeout budget.

    Maps to HTTP 504 Gateway Timeout in the API router.
    """


class LLMProviderError(LLMClientError):
    """Raised on upstream 4xx/5xx errors, network failures, or empty responses.

    Maps to HTTP 502 Bad Gateway in the API router.
    """


def _get_timeout_seconds() -> float:
    raw_timeout = os.getenv("LLM_TIMEOUT_SECONDS")
    if not raw_timeout:
        return DEFAULT_TIMEOUT_SECONDS
    try:
        value = float(raw_timeout)
        return value if value > 0 else DEFAULT_TIMEOUT_SECONDS
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _get_max_tokens() -> int:
    raw_max_tokens = os.getenv("LLM_MAX_TOKENS")
    if not raw_max_tokens:
        return DEFAULT_MAX_TOKENS
    try:
        value = int(raw_max_tokens)
        return value if value > 0 else DEFAULT_MAX_TOKENS
    except ValueError:
        return DEFAULT_MAX_TOKENS


def _create_genai_client(api_key: str) -> genai.Client:
    """Instantiate the Google GenAI client."""
    return genai.Client(api_key=api_key)


_mcp_exit_stack: AsyncExitStack | None = None
_mcp_session: ClientSession | None = None
_mcp_gemini_tools: list[types.Tool] = []


def _get_mcp_server_params() -> StdioServerParameters:
    """Build the stdio server parameters used to launch mcp_server.py as a subprocess."""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(MCP_SERVER_SCRIPT_PATH)],
    )


def _sanitize_json_schema_for_gemini(schema: Any) -> dict[str, Any]:
    """Filter an MCP JSON Schema dictionary to fields supported by google.genai.types.Schema."""
    if not isinstance(schema, dict):
        return {}
    allowed_fields = set(types.Schema.model_fields.keys())
    cleaned: dict[str, Any] = {}
    for key, value in schema.items():
        if key not in allowed_fields:
            continue
        if key == "properties" and isinstance(value, dict):
            cleaned[key] = {
                prop_name: _sanitize_json_schema_for_gemini(prop_schema)
                for prop_name, prop_schema in value.items()
            }
        elif key == "items" and isinstance(value, dict):
            cleaned[key] = _sanitize_json_schema_for_gemini(value)
        else:
            cleaned[key] = value
    return cleaned


async def _discover_mcp_tools(session: ClientSession) -> list[types.Tool]:
    """Dynamically discover tools from the MCP server and convert them for the Gemini SDK."""
    tools_result = await session.list_tools()
    mcp_tools = getattr(tools_result, "tools", None) or []

    declarations: list[types.FunctionDeclaration] = []
    for mcp_tool in mcp_tools:
        raw_schema = (
            getattr(mcp_tool, "inputSchema", None)
            or getattr(mcp_tool, "input_schema", None)
            or {}
        )
        declarations.append(
            types.FunctionDeclaration(
                name=mcp_tool.name,
                description=getattr(mcp_tool, "description", None),
                parameters=_sanitize_json_schema_for_gemini(raw_schema) or None,
            )
        )

    logger.info(
        "Discovered tools from MCP server",
        extra={
            "event": "mcp_tools_discovered",
            "tool_names": [decl.name for decl in declarations],
            "tool_count": len(declarations),
        },
    )
    return [types.Tool(function_declarations=declarations)] if declarations else []


async def init_mcp_client() -> ClientSession:
    """Launch the MCP server subprocess and initialize the global MCP client session."""
    global _mcp_exit_stack, _mcp_session, _mcp_gemini_tools

    if _mcp_session is not None:
        return _mcp_session

    stack = AsyncExitStack()
    try:
        server_params = _get_mcp_server_params()
        read_stream, write_stream = await stack.enter_async_context(
            stdio_client(server_params)
        )
        session = await stack.enter_async_context(
            ClientSession(read_stream, write_stream)
        )
        await session.initialize()
        gemini_tools = await _discover_mcp_tools(session)

        _mcp_exit_stack = stack
        _mcp_session = session
        _mcp_gemini_tools = gemini_tools

        tool_names = [
            decl.name
            for tool in gemini_tools
            for decl in (getattr(tool, "function_declarations", None) or [])
        ]
        logger.info(
            "MCP client connected and initialized successfully",
            extra={
                "event": "mcp_client_connected",
                "server_script": str(MCP_SERVER_SCRIPT_PATH),
                "tool_names": tool_names,
                "tool_count": len(tool_names),
            },
        )
        return session
    except Exception as exc:
        await stack.aclose()
        logger.error(
            "Failed to initialize MCP client",
            extra={
                "event": "mcp_client_init_failed",
                "server_script": str(MCP_SERVER_SCRIPT_PATH),
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise LLMProviderError(f"Failed to initialize MCP client: {exc}") from exc


async def close_mcp_client() -> None:
    """Close the global MCP client session and terminate the MCP server subprocess."""
    global _mcp_exit_stack, _mcp_session, _mcp_gemini_tools

    stack = _mcp_exit_stack
    _mcp_exit_stack = None
    _mcp_session = None
    _mcp_gemini_tools = []

    if stack is not None:
        await stack.aclose()
        logger.info(
            "MCP client connection closed",
            extra={
                "event": "mcp_client_closed",
            },
        )


def get_mcp_client() -> ClientSession | None:
    """Return the globally initialized MCP client session, if active."""
    return _mcp_session


async def _generate_content_with_retries(
    client: genai.Client,
    model_name: str,
    contents: Any,
    config: types.GenerateContentConfig,
    input_payload_length: int,
) -> Any:
    """Execute a single async Gemini generate_content call with lightweight retry on transient errors."""
    retryer = AsyncRetrying(
        stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
        retry=retry_if_exception_type(
            (errors.ServerError, httpx.TransportError, ConnectionError)
        ),
        reraise=True,
    )

    async for attempt in retryer:
        with attempt:
            attempt_number = attempt.retry_state.attempt_number
            if attempt_number > 1:
                logger.warning(
                    "Retrying Gemini API call after transient error",
                    extra={
                        "event": "llm_call_retry",
                        "model": model_name,
                        "attempt": attempt_number,
                        "max_attempts": MAX_RETRY_ATTEMPTS,
                        "input_payload_length": input_payload_length,
                    },
                )
            return await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=config,
            )

    raise LLMProviderError("Upstream LLM call failed after retry attempts.")


def _extract_function_calls(response: Any) -> list[Any]:
    """Extract any tool/function call requests from a Gemini response."""
    direct_calls = getattr(response, "function_calls", None)
    if direct_calls:
        return list(direct_calls)

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return []
    first_content = getattr(candidates[0], "content", None)
    parts = getattr(first_content, "parts", None)
    if not parts:
        return []

    return [
        part.function_call
        for part in parts
        if getattr(part, "function_call", None) is not None
    ]


async def _execute_tool_call(
    session: ClientSession,
    func_call: Any,
) -> types.Part:
    """Route a requested tool call to the MCP server and wrap the result in a FunctionResponse Part."""
    func_name = getattr(func_call, "name", "") or ""
    raw_args = getattr(func_call, "args", None) or {}
    call_args = dict(raw_args) if isinstance(raw_args, dict) else {}

    logger.info(
        "Executing requested tool call via MCP server",
        extra={
            "event": "llm_tool_call_executed",
            "tool_name": func_name,
            "tool_args": call_args,
        },
    )
    try:
        tool_result = await session.call_tool(func_name, arguments=call_args)
        content_blocks = getattr(tool_result, "content", None) or []
        text_parts = [
            getattr(block, "text", str(block)) for block in content_blocks
        ]
        combined_text = "\n".join(text_parts)

        is_error = bool(
            getattr(tool_result, "is_error", False)
            or getattr(tool_result, "isError", False)
        )
        if is_error:
            error_msg = combined_text or f"MCP tool call failed: {func_name}"
            logger.warning(
                "MCP tool call returned an error",
                extra={
                    "event": "llm_tool_call_failed",
                    "tool_name": func_name,
                    "error": error_msg,
                },
            )
            response_payload: dict[str, Any] = {"error": error_msg}
        else:
            response_payload = {"result": combined_text}
    except Exception as exc:
        logger.warning(
            "MCP tool execution raised an error",
            extra={
                "event": "llm_tool_call_failed",
                "tool_name": func_name,
                "error": str(exc),
            },
        )
        response_payload = {"error": str(exc)}

    return types.Part.from_function_response(
        name=func_name,
        response=response_payload,
    )


async def _call_gemini_with_retries(
    client: genai.Client,
    model_name: str,
    user_content: str,
    config: types.GenerateContentConfig,
    session: ClientSession,
) -> str:
    """Execute the async Gemini generation call and route any tool calls through the MCP session."""
    input_payload_length = len(user_content)
    overall_start = time.perf_counter()

    response = await _generate_content_with_retries(
        client=client,
        model_name=model_name,
        contents=user_content,
        config=config,
        input_payload_length=input_payload_length,
    )

    function_calls = _extract_function_calls(response)
    if function_calls:
        conversation: list[Any] = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=user_content)],
            )
        ]

        turns = 0
        while function_calls and turns < MAX_TOOL_CALL_TURNS:
            turns += 1
            candidates = getattr(response, "candidates", None)
            model_content = (
                candidates[0].content
                if candidates and getattr(candidates[0], "content", None) is not None
                else types.Content(
                    role="model",
                    parts=[
                        types.Part.from_function_call(
                            name=getattr(fc, "name", ""),
                            args=dict(getattr(fc, "args", None) or {}),
                        )
                        for fc in function_calls
                    ],
                )
            )
            conversation.append(model_content)

            tool_response_parts = [
                await _execute_tool_call(session, fc) for fc in function_calls
            ]
            conversation.append(
                types.Content(
                    role="user",
                    parts=tool_response_parts,
                )
            )

            response = await _generate_content_with_retries(
                client=client,
                model_name=model_name,
                contents=conversation,
                config=config,
                input_payload_length=input_payload_length,
            )
            function_calls = _extract_function_calls(response)

    attempt_elapsed = time.perf_counter() - overall_start
    text = getattr(response, "text", None)
    if not text or not isinstance(text, str) or not text.strip():
        logger.error(
            "Gemini API returned an empty response",
            extra={
                "event": "llm_empty_response",
                "model": model_name,
                "attempt": 1,
                "elapsed_seconds": round(attempt_elapsed, 4),
                "latency_ms": round(attempt_elapsed * 1000, 2),
                "input_payload_length": input_payload_length,
                "status": "failed",
                "succeeded": False,
                "timed_out": False,
            },
        )
        raise LLMProviderError("Upstream LLM returned an empty response.")
    return text.strip()


async def generate_digest_raw(
    system_prompt: str,
    incident_payload: dict[str, Any],
    mcp_session: ClientSession | None = None,
) -> str:
    """Send the system prompt and incident payload to Gemini and return the raw text output.

    Enforces:
    - `max_output_tokens=500` (or `LLM_MAX_TOKENS` from environment)
    - Strict overall timeout of 60.0 seconds (or `LLM_TIMEOUT_SECONDS` from environment)
    - Lightweight retry (up to 2 attempts) for transient upstream 5xx / transport errors
    - Reuses the globally initialized MCP client session (`_mcp_session`)

    Raises:
        LLMTimeoutError: If the overall operation exceeds the timeout budget (60s).
        LLMProviderError: If configuration or the MCP client is missing, upstream
            returns an error, or the model returns an empty response.
    """
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        logger.error(
            "GEMINI_API_KEY environment variable is not configured",
            extra={
                "event": "llm_config_error",
                "status": "failed",
                "succeeded": False,
                "timed_out": False,
            },
        )
        raise LLMProviderError("GEMINI_API_KEY environment variable is not configured.")

    active_session = mcp_session or _mcp_session
    if active_session is None:
        logger.error(
            "MCP client is not initialized",
            extra={
                "event": "mcp_client_not_initialized",
                "status": "failed",
                "succeeded": False,
                "timed_out": False,
            },
        )
        raise LLMProviderError(
            "MCP client is not initialized. Ensure init_mcp_client() is called during startup."
        )

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    timeout_seconds = _get_timeout_seconds()
    max_tokens = _get_max_tokens()

    user_content = json.dumps(incident_payload, ensure_ascii=False)
    input_payload_length = len(user_content)
    description_length = len(str(incident_payload.get("description", "")))

    logger.info(
        "Starting Gemini API request",
        extra={
            "event": "llm_request_started",
            "model": model_name,
            "input_payload_length": input_payload_length,
            "description_length": description_length,
            "timeout_seconds": timeout_seconds,
            "max_output_tokens": max_tokens,
        },
    )

    start_time = time.perf_counter()
    try:
        client = _create_genai_client(api_key=api_key)
        gemini_tools = (
            _mcp_gemini_tools
            if (mcp_session is None and _mcp_gemini_tools)
            else await _discover_mcp_tools(active_session)
        )

        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=0.1,
            tools=gemini_tools,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )

        raw_output = await asyncio.wait_for(
            _call_gemini_with_retries(
                client=client,
                model_name=model_name,
                user_content=user_content,
                config=config,
                session=active_session,
            ),
            timeout=timeout_seconds,
        )

        elapsed_seconds = time.perf_counter() - start_time
        logger.info(
            "Gemini API request succeeded",
            extra={
                "event": "llm_request_completed",
                "model": model_name,
                "status": "succeeded",
                "succeeded": True,
                "timed_out": False,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "latency_ms": round(elapsed_seconds * 1000, 2),
                "input_payload_length": input_payload_length,
                "description_length": description_length,
                "output_length": len(raw_output),
            },
        )
        return raw_output
    except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException) as exc:
        elapsed_seconds = time.perf_counter() - start_time
        logger.error(
            "Gemini API request timed out",
            extra={
                "event": "llm_request_timed_out",
                "model": model_name,
                "status": "timed_out",
                "succeeded": False,
                "timed_out": True,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "latency_ms": round(elapsed_seconds * 1000, 2),
                "timeout_seconds": timeout_seconds,
                "input_payload_length": input_payload_length,
                "description_length": description_length,
            },
        )
        raise LLMTimeoutError(
            f"Upstream LLM request timed out after {timeout_seconds} seconds."
        ) from exc
    except LLMClientError as exc:
        elapsed_seconds = time.perf_counter() - start_time
        logger.error(
            "Gemini API request failed with client error",
            extra={
                "event": "llm_request_failed",
                "model": model_name,
                "status": "failed",
                "succeeded": False,
                "timed_out": False,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "latency_ms": round(elapsed_seconds * 1000, 2),
                "input_payload_length": input_payload_length,
                "description_length": description_length,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    except errors.APIError as exc:
        elapsed_seconds = time.perf_counter() - start_time
        logger.error(
            "Upstream Gemini API error",
            extra={
                "event": "llm_request_failed",
                "model": model_name,
                "status": "failed",
                "succeeded": False,
                "timed_out": False,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "latency_ms": round(elapsed_seconds * 1000, 2),
                "input_payload_length": input_payload_length,
                "description_length": description_length,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise LLMProviderError(f"Upstream Gemini API error: {exc}") from exc
    except httpx.HTTPError as exc:
        elapsed_seconds = time.perf_counter() - start_time
        logger.error(
            "Upstream HTTP transport error communicating with Gemini API",
            extra={
                "event": "llm_request_failed",
                "model": model_name,
                "status": "failed",
                "succeeded": False,
                "timed_out": False,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "latency_ms": round(elapsed_seconds * 1000, 2),
                "input_payload_length": input_payload_length,
                "description_length": description_length,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise LLMProviderError(f"Upstream HTTP transport error: {exc}") from exc
    except Exception as exc:
        elapsed_seconds = time.perf_counter() - start_time
        logger.error(
            "Unexpected error communicating with LLM provider",
            extra={
                "event": "llm_request_failed",
                "model": model_name,
                "status": "failed",
                "succeeded": False,
                "timed_out": False,
                "elapsed_seconds": round(elapsed_seconds, 4),
                "latency_ms": round(elapsed_seconds * 1000, 2),
                "input_payload_length": input_payload_length,
                "description_length": description_length,
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise LLMProviderError(f"Unexpected error communicating with LLM provider: {exc}") from exc
