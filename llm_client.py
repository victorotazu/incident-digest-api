"""Isolated asynchronous LLM adapter (Layer 4) for Google Gemini.

Encapsulates vendor SDK calls, token limits, retry policy, and strict request
timeouts while exposing a provider-agnostic interface and exception hierarchy
to the API router.
"""

from __future__ import annotations

import asyncio
import json
import os
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
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=user_content,
                config=config,
            )
            text = getattr(response, "text", None)
            if not text or not isinstance(text, str) or not text.strip():
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
        raise LLMProviderError("GEMINI_API_KEY environment variable is not configured.")

    model_name = os.getenv("GEMINI_MODEL", DEFAULT_GEMINI_MODEL).strip() or DEFAULT_GEMINI_MODEL
    timeout_seconds = _get_timeout_seconds()
    max_tokens = _get_max_tokens()

    user_content = json.dumps(incident_payload, ensure_ascii=False)
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.1,
    )

    try:
        client = _create_genai_client(api_key=api_key)
        return await asyncio.wait_for(
            _call_gemini_with_retries(
                client=client,
                model_name=model_name,
                user_content=user_content,
                config=config,
            ),
            timeout=timeout_seconds,
        )
    except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException) as exc:
        raise LLMTimeoutError(
            f"Upstream LLM request timed out after {timeout_seconds} seconds."
        ) from exc
    except LLMClientError:
        raise
    except errors.APIError as exc:
        raise LLMProviderError(f"Upstream Gemini API error: {exc}") from exc
    except httpx.HTTPError as exc:
        raise LLMProviderError(f"Upstream HTTP transport error: {exc}") from exc
    except Exception as exc:
        raise LLMProviderError(f"Unexpected error communicating with LLM provider: {exc}") from exc
