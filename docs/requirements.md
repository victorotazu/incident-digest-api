# Engineering Requirements: Incident Digest API (v0.1)

## 1. Functional Requirements
* **Input Validation:** The API must reject requests that do not match the expected JSON schema (HTTP 400).
* **Payload Management:** The system must enforce a strict input size limit to protect the LLM context window. If the `description` exceeds 20,000 characters, the API will silently truncate it from the beginning (keeping the most recent stack trace logs at the end) before sending it to the LLM.
* **Statelessness:** The API will not store incident data, logs, or LLM responses in any database.

## 2. Non-Functional Requirements
* **Latency:** The API must return a response within 60 seconds (HTTP 504 Gateway Timeout if exceeded).
* **Cost / Token Limits:** The system must strictly bound the LLM output to a maximum of 500 tokens to prevent runaway generation costs.
* **Format Guarantee:** The API must guarantee a valid JSON response to the client, handling any LLM markdown formatting errors internally before responding.

## 3. Edge Cases & Error Handling
* **Edge Case 1 - Empty/Insufficient Context:** If the `description` is empty but the `title` exists, the system must still attempt a recommendation based purely on the title, rather than failing.
* **Edge Case 2 - Gibberish/Irrelevant Data:** If the input data is unrelated to a software system incident (e.g., random text), the LLM must be instructed to return a safe fallback: `{"summary": "Invalid incident data", "recommended_action": "None"}` rather than hallucinating a response.
* **Edge Case 3 - LLM Provider Outage:** If the underlying LLM API is down or times out, our API must return a graceful HTTP 502/503 error, not a generic 500 stack trace.
