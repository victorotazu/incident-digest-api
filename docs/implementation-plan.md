# Implementation Plan: Incident Digest API (v0.1)

## Key Technical Decisions
* **LLM Provider:** Google Gemini (isolated within `llm_client.py` to preserve vendor agnosticism).
* **Authentication:** Static Bearer token validation via the `API_BEARER_TOKEN` environment variable.

## Target Project Structure

Following the 4-layer "Sandwich" architecture in `docs/architecture.md`:

```text
incident-digest-api/
├── prompts/
│   └── incident_v1.txt      # Layer 3: Decoupled system instructions
├── tests/
│   └── test_digest_api.py   # Unit & integration tests covering happy path + edge cases
├── llm_client.py            # Layer 4: Isolated async LLM adapter (Gemini)
├── schemas.py               # Layer 2: Pydantic input/output validation & truncation
├── main.py                  # Layer 1: FastAPI router, auth, & error handling
├── requirements.txt         # Python dependencies
└── .env.example             # Environment variable template
```

---

## Step 1: Project Setup & Configuration
**Goal:** Establish the runtime dependencies and environment configuration without coupling secrets or provider settings to code.

1. **Dependencies (`requirements.txt`):**
   - `fastapi` & `uvicorn` for the async HTTP server.
   - `pydantic` & `pydantic-settings` (or `python-dotenv`) for data validation and environment config.
   - `google-genai` (or `httpx`) for async Gemini API calls and `tenacity` for lightweight retry logic.
   - `pytest` & `pytest-asyncio` for automated testing.
2. **Environment Variables (`.env.example`):**
   - `API_BEARER_TOKEN`: Expected static bearer token for `bearerAuth` validation.
   - `GEMINI_API_KEY`: API key for Google Gemini.
   - `GEMINI_MODEL`: Default model (e.g., `gemini-2.5-flash`).
   - `LLM_TIMEOUT_SECONDS=60` and `LLM_MAX_TOKENS=500`.

---

## Step 2: Validation Layer (`schemas.py`)
**Goal:** Enforce the `docs/api-spec.yaml` contract and deterministic defensive boundaries around the LLM (`docs/architecture.md`).

1. **Input Model (`IncidentDigestRequest`):**
   - Fields:
     - `title` (`str`, required): Must not be blank/whitespace-only (`min_length=1` after stripping).
     - `severity` (`Optional[str] = None`): Optional priority level (e.g., `P1`, `P2`, `SEV-1`).
     - `description` (`str`, required): Allowed to be an empty string `""` per **Edge Case 1** in `docs/requirements.md` (when `description` is empty, the system still generates a recommendation from `title`).
   - **Silent Tail-Preserving Truncation:**
     - Add a Pydantic `@field_validator("description")` that checks if `len(description) > 20_000`.
     - If exceeded, silently truncate from the **beginning** (`description[-20000:]`) to preserve the most recent stack trace and log lines at the end (`docs/requirements.md`).
2. **Output Model (`IncidentDigestResponse`):**
   - Fields:
     - `summary` (`str`, required, non-empty)
     - `recommended_action` (`str`, required, non-empty)
   - **Format Guarantee & Sanitization:**
     - Provide a helper/validator to parse raw LLM string output: strip accidental Markdown code fences (e.g., ```` ```json ... ``` ````), parse JSON, and validate against `IncidentDigestResponse` (`docs/requirements.md`).
     - Raise a dedicated validation/parsing error if the LLM output violates the schema so the router can translate it into `HTTP 502`.

---

## Step 3: Prompt Asset (`prompts/incident_v1.txt`)
**Goal:** Decouple LLM system instructions from Python code (`docs/architecture.md`) while enforcing all behavioral requirements.

1. **Role & Objective:**
   - Instruct the model to act as an expert Site Reliability / On-Call Engineer assistant analyzing an incident alert.
2. **Output Contract Enforcement:**
   - Explicitly require strict raw JSON output with keys `"summary"` and `"recommended_action"` only—no markdown code blocks, no prose outside JSON, and concise responses within the 500-token budget.
3. **Edge Case Handling in Prompt:**
   - **Empty Description (Edge Case 1):** Instruct the model that if `description` is empty or minimal, infer the most sensible initial investigation step based solely on `title` (and `severity` if provided).
   - **Gibberish / Irrelevant Input (Edge Case 2):** Instruct the model that if the payload is gibberish or unrelated to a software/infrastructure incident, it must return the exact fallback:
     ```json
     {"summary": "Invalid incident data", "recommended_action": "None"}
     ```
   - **No Auto-Remediation (`docs/product-spec.md`):** Ensure `recommended_action` is framed as an actionable next step for the human engineer.

---

## Step 4: LLM Adapter (`llm_client.py`)
**Goal:** Isolate network communication, token limits, timeouts, and retries into a vendor-agnostic adapter (`docs/architecture.md`) backed by Gemini.

1. **Custom Exception Hierarchy:**
   - `LLMTimeoutError`: Raised when the upstream call exceeds **60 seconds** (maps to `HTTP 504`).
   - `LLMProviderError`: Raised on upstream 4xx/5xx errors, connection failures, or empty responses (maps to `HTTP 502`).
2. **Async Execution & Constraints:**
   - Implement `async def generate_digest_raw(system_prompt: str, incident_payload: dict) -> str`.
   - Configure Gemini generation with `max_output_tokens=500` (`docs/requirements.md`).
   - Enforce a strict overall timeout of **60.0 seconds** (`docs/requirements.md`, `docs/api-spec.yaml`).
   - Include lightweight retry logic (e.g., up to 2 attempts with short backoff for transient network/5xx errors) bounded within the 60s request budget.

---

## Step 5: API Router & Error Handling (`main.py`)
**Goal:** Wire the pipeline (`Client -> Router -> Input Validator -> [Loads Prompt] -> LLM Adapter -> Output Validator -> Router -> Client`) and map all outcomes to the exact status codes in `docs/api-spec.yaml`.

1. **Authentication (`HTTP 401`):**
   - Implement Bearer token verification matching `securitySchemes.bearerAuth` against `API_BEARER_TOKEN` (`docs/api-spec.yaml`).
   - Ensure both missing `Authorization` headers and invalid/malformed tokens return `HTTP 401 Unauthorized` (overriding FastAPI's default `403` behavior on missing `HTTPBearer` credentials).
2. **Schema Validation Override (`HTTP 400`):**
   - Register custom exception handlers for `RequestValidationError` and malformed JSON so invalid schemas or missing required fields (`title`, `description`) return `HTTP 400 Bad Request` instead of FastAPI's default `422 Unprocessable Entity` (`docs/api-spec.yaml`, `docs/requirements.md`).
3. **Endpoint (`POST /api/v1/digest`):**
   - Validate request body via `IncidentDigestRequest` (which also performs the 20,000-char tail truncation).
   - Load `prompts/incident_v1.txt`.
   - Call `llm_client.generate_digest_raw(...)`.
   - Parse and validate the raw string into `IncidentDigestResponse`.
   - Catch `LLMTimeoutError` -> return `HTTP 504 Gateway Timeout`.
   - Catch `LLMProviderError` or output validation failure -> return `HTTP 502 Bad Gateway` (ensuring no unhandled `500` stack traces leak to the client per **Edge Case 3**).

---

## Step 6: Verification & Test Suite (`tests/test_digest_api.py`)
**Goal:** Deterministically verify every requirement and edge case using mocked LLM responses before live testing.

1. **Contract & Auth Tests:**
   - `200 OK`: Valid payload returns `summary` and `recommended_action`.
   - `401 Unauthorized`: Missing or wrong bearer token.
   - `400 Bad Request`: Missing `title`, missing `description`, or invalid field types.
2. **Functional & Edge Case Tests:**
   - **20,000-Char Truncation:** Send a 25,000-char `description` and assert the LLM adapter receives only the **last** 20,000 characters.
   - **Markdown Stripping:** Mock LLM returning ```` ```json\n{"summary": "...", "recommended_action": "..."}\n``` ```` and verify `200 OK` clean JSON response.
   - **Empty Description (Edge Case 1):** Send `description=""` with a valid `title` and verify `200 OK`.
   - **Gibberish Input (Edge Case 2):** Verify fallback response structure `{"summary": "Invalid incident data", "recommended_action": "None"}`.
   - **Upstream Failure & Timeout (Edge Case 3):** Mock provider error -> assert `502`; mock timeout > 60s -> assert `504`.
