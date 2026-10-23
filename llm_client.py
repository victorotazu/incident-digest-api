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
from datetime import datetime, timezone
from typing import Any

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
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


async def _call_gemini_with_retries(
    client: genai.Client,
    model_name: str,
    user_content: str,
    config: types.GenerateContentConfig,
) -> str:
    """Execute the async Gemini generation call with lightweight retry on transient errors."""
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
            attempt_start = time.perf_counter()
            if attempt_number > 1:
                logger.warning(
                    "Retrying Gemini API call after transient error",
                    extra={
                        "event": "llm_call_retry",
                        "model": model_name,
                        "attempt": attempt_number,
                        "max_attempts": MAX_RETRY_ATTEMPTS,
                        "input_payload_length": len(user_content),
                    },
                )
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=user_content,
                config=config,
            )
            attempt_elapsed = time.perf_counter() - attempt_start
            text = getattr(response, "text", None)
            if not text or not isinstance(text, str) or not text.strip():
                logger.error(
                    "Gemini API returned an empty response",
                    extra={
                        "event": "llm_empty_response",
                        "model": model_name,
                        "attempt": attempt_number,
                        "elapsed_seconds": round(attempt_elapsed, 4),
                        "latency_ms": round(attempt_elapsed * 1000, 2),
                        "input_payload_length": len(user_content),
                        "status": "failed",
                        "succeeded": False,
                        "timed_out": False,
                    },
                )
                raise LLMProviderError("Upstream LLM returned an empty response.")
            return text.strip()

    raise LLMProviderError("Upstream LLM call failed after retry attempts.")


async def generate_digest_raw(
    system_prompt: str,
    incident_payload: dict[str, Any],
) -> str:
    """Send the system prompt and incident payload to Gemini and return the raw text output.

    Enforces:
    - `max_output_tokens=500` (or `LLM_MAX_TOKENS` from environment)
    - Strict overall timeout of 60.0 seconds (or `LLM_TIMEOUT_SECONDS` from environment)
    - Lightweight retry (up to 2 attempts) for transient upstream 5xx / transport errors

    Raises:
        LLMTimeoutError: If the overall operation exceeds the timeout budget (60s).
        LLMProviderError: If configuration is missing, upstream returns an error,
            or the model returns an empty response.
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

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    timeout_seconds = _get_timeout_seconds()
    max_tokens = _get_max_tokens()

    user_content = json.dumps(incident_payload, ensure_ascii=False)
    input_payload_length = len(user_content)
    description_length = len(str(incident_payload.get("description", "")))

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.1,
    )

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
        raw_output = await asyncio.wait_for(
            _call_gemini_with_retries(
                client=client,
                model_name=model_name,
                user_content=user_content,
                config=config,
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
