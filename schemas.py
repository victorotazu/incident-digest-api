"""Validation layer (Layer 2) for the Incident Digest API.

Enforces the OpenAPI contract (`docs/api-spec.yaml`), silent tail-preserving
input truncation (`docs/requirements.md`), and LLM output sanitization/validation.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

MAX_TITLE_LENGTH: int = 255
MAX_SEVERITY_LENGTH: int = 255
MAX_DESCRIPTION_LENGTH: int = 20_000

# Matches optional leading/trailing markdown code fences such as ```json ... ``` or ``` ... ```
_MARKDOWN_FENCE_PATTERN = re.compile(
    r"^\s*```(?:json)?\s*(.*?)\s*```\s*$",
    re.DOTALL | re.IGNORECASE,
)


class LLMOutputValidationError(ValueError):
    """Raised when raw LLM output cannot be parsed or fails schema validation.

    The API router translates this error into an HTTP 502 Bad Gateway response.
    """


class IncidentDigestRequest(BaseModel):
    """Incoming request payload for `POST /api/v1/digest`."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(
        ...,
        max_length=MAX_TITLE_LENGTH,
        description="The short title or subject of the incident alert.",
    )
    severity: Optional[str] = Field(
        default=None,
        max_length=MAX_SEVERITY_LENGTH,
        description="The priority level (e.g., P1, P2, SEV-1).",
    )
    description: str = Field(
        ...,
        description="The raw text, logs, or stack trace of the incident. Max 20,000 chars.",
    )

    @field_validator("title", mode="before")
    @classmethod
    def validate_title_type(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("title must be a string.")
        return value

    @field_validator("title", mode="after")
    @classmethod
    def validate_title_not_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("title must not be empty or whitespace-only.")
        return stripped

    @field_validator("severity", mode="before")
    @classmethod
    def validate_severity_type(cls, value: object) -> Optional[str]:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("severity must be a string if provided.")
        return value

    @field_validator("description", mode="before")
    @classmethod
    def validate_description_type(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("description must be a string.")
        return value

    @field_validator("description", mode="after")
    @classmethod
    def truncate_description_tail(cls, value: str) -> str:
        """Silently truncate from the beginning if description exceeds 20,000 chars.

        Preserves the most recent stack trace and log lines at the tail end.
        Empty strings ("") are explicitly allowed per Edge Case 1.
        """
        if len(value) > MAX_DESCRIPTION_LENGTH:
            return value[-MAX_DESCRIPTION_LENGTH:]
        return value


class IncidentDigestResponse(BaseModel):
    """Validated outgoing response payload for `POST /api/v1/digest`."""

    model_config = ConfigDict(extra="ignore")

    summary: str = Field(
        ...,
        description="A human-readable synthesis of the failing component and reason.",
    )
    recommended_action: str = Field(
        ...,
        description="The immediate next step the engineer should take.",
    )

    @field_validator("summary", "recommended_action", mode="before")
    @classmethod
    def validate_string_type(cls, value: object) -> str:
        if not isinstance(value, str):
            raise ValueError("Field must be a string.")
        return value

    @field_validator("summary", "recommended_action", mode="after")
    @classmethod
    def validate_non_empty(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("Field must not be empty or whitespace-only.")
        return stripped

    @classmethod
    def from_llm_raw(cls, raw_output: str) -> IncidentDigestResponse:
        """Sanitize and validate raw text output from the LLM into `IncidentDigestResponse`.

        Strips accidental Markdown code fences (e.g., ```json ... ```), parses JSON,
        and validates against the `IncidentDigestResponse` schema.

        Raises:
            LLMOutputValidationError: If the output is empty, not valid JSON, not a JSON
                object, or violates the `IncidentDigestResponse` schema.
        """
        if not isinstance(raw_output, str) or not raw_output.strip():
            raise LLMOutputValidationError("LLM returned an empty or non-string response.")

        cleaned = strip_markdown_code_fences(raw_output)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise LLMOutputValidationError(
                f"LLM output is not valid JSON: {exc.msg}"
            ) from exc

        if not isinstance(parsed, dict):
            raise LLMOutputValidationError(
                "LLM JSON output must be a top-level JSON object."
            )

        try:
            return cls.model_validate(parsed)
        except ValidationError as exc:
            raise LLMOutputValidationError(
                f"LLM JSON output failed schema validation: {exc}"
            ) from exc


def strip_markdown_code_fences(raw_text: str) -> str:
    """Remove surrounding Markdown code fences (```json ... ``` or ``` ... ```) if present."""
    stripped = raw_text.strip()
    match = _MARKDOWN_FENCE_PATTERN.match(stripped)
    if match:
        return match.group(1).strip()
    return stripped
