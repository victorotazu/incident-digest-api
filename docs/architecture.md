# Architecture: Incident Digest API (v0.1)

## 1. Core Principles
* **Defensive Boundaries:** The probabilistic LLM core must be tightly wrapped by deterministic validation layers.
* **Separation of Instructions:** Prompts are treated as configuration/assets, decoupled from execution logic.
* **Vendor Agnosticism:** The core business logic should not be tightly coupled to a specific LLM provider's SDK.

## 2. Component Layers

The application follows a 4-layer "Sandwich" architecture:

1. **API Router (`main.py`):**
   * Handles HTTP POST requests, authentication checks, and returning HTTP status codes.
2. **Validation Layer (`schemas.py`):**
   * Uses Pydantic models to enforce the `api-spec.yaml` contract.
   * Handles input truncation (max characters) before passing to the service.
   * Validates the final LLM output structure before returning it to the user.
3. **Prompt Asset (`prompts/incident_v1.txt`):**
   * A plain text or markdown file containing the system instructions for the LLM. Completely decoupled from Python logic.
4. **LLM Adapter (`llm_client.py`):**
   * The isolated module responsible for the fragile network call.
   * Handles asynchronous requests, timeouts, and basic retry logic.
   * Takes the prompt and user data, returns a raw string.

## 3. Data Flow
`Client -> Router -> Input Validator -> [Loads Prompt] -> LLM Adapter -> Output Validator -> Router -> Client`
