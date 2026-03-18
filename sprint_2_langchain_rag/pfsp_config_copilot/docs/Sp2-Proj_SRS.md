# Sp2-Proj_SRS: PFSP AI Configuration Copilot (Sprint 2 MVP)

## 1. Document Status

- Project: `PFSP AI Configuration Copilot`
- Location: `sprint_2_langchain_rag/pfsp_config_copilot`
- Document version: `0.4`
- Project status: `Working Sprint 2 MVP`
- Updated: `2026-03-18`
- Audience: Sprint 2 reviewer, project owner, solo implementer

This document is an SRS realignment of the currently implemented MVP to match the Sprint 2
"Building Applications with AI" assignment requirements.

This document is the **Sprint 2 MVP scope document**, and the source of truth for the full
product SRS remains the following file.

- `/workspace/capstone_design/PFSP_AI_Configuration_Copilot_SRS_ModuleOnly_v.2.2.json`

In other words, requirements that are intentionally reduced in scope or de-emphasized in this
document are not removed. They remain managed in the **final capstone SRS**.

Key assumptions:

- This project is a **PFSP-centered automotive software configuration AI web application**.
- This project centers on **structured configuration JSON draft generation**.
- This project uses **advanced RAG + query translation + tool calling** as its main implementation signals.
- This project does **not** aim to be a **production-grade compliance engine**.

### 1.1 Deferred Requirement Ownership Confirmation

Requirements that have been removed from the center of the Sprint 2 MVP document, or reduced in
scope, are still defined in the final SRS as shown below.

| Deferred / De-emphasized Area in Sprint 2 MVP | Final SRS Reference | Confirmation |
| --- | --- | --- |
| Detailed Error Handling field extraction (`ErrorTypes`, debouncing thresholds, `ReactionModes/States`) | `REQ-MOD-201` | Defined in the final SRS |
| AUTOSAR / ISO 26262 supporting-context usage | `REQ-RAG-403` | Defined in the final SRS |
| Smart Converter mapping generation | `REQ-MAP-201`, `REQ-MAP-202`, `REQ-MAP-203` | Defined in the final SRS |
| Field-Level Traceability (`source_ref`, `traceability_id`) | `REQ-TRC-401` | Defined in the final SRS |
| Strong Two-Phase Reasoning structure and Separation of Concerns | `REQ-PRO-501`, `REQ-PRO-502`, `REQ-PRO-503` | Defined in the final SRS |

In summary:

- `Sp2-Proj_SRS.md` describes the Sprint 2 assignment submission scope.
- `PFSP_AI_Configuration_Copilot_SRS_ModuleOnly_v.2.2.json` retains the full module requirements for the final capstone stage.
- Therefore, requirements omitted from the Sprint 2 document are **not deleted; they are deferred to the final SRS**.

---

## 2. Project Goal

The Sprint 2 MVP goal of PFSP AI Configuration Copilot is as follows.

When a user enters a natural-language configuration requirement, the application should:

1. retrieve relevant standards and reference documents,
2. generate a structured JSON draft for the core configuration parameters based on the retrieved context,
3. check draft completeness with Pydantic validation and basic rule checks, and
4. present source context, tool call results, and validation output in a Streamlit UI.

This MVP is a **domain-specific AI web app for automotive software configuration for a target platform**.

The core value of this MVP is:

- **structured output** aligned with the target configuration schema
- a Chroma-based **retrieval pipeline**
- **advanced RAG** including query translation
- at least 3 **explicit tool calls**
- an **explainable UI** where users can inspect intermediate artifacts

---

## 3. Sprint 2 MVP Scope

The following scope items must be included in the current Sprint 2 MVP.

### 3.1 Scope and Direct Assignment Mapping

| Scope Item | Description | Related Assignment No. | Current Module / Evidence |
| --- | --- | --- | --- |
| Domain-specific application | An automotive software configuration app centered on target-platform configuration workflows | Task `3` | [README.md](../README.md), [app.py](../app.py) |
| Structured output | Generate a JSON draft using `ServiceName`, `ID`, `Class`, `Frequency`, and `PlayType` | Task `3` | [services/schemas.py](../services/schemas.py), [services/extraction.py](../services/extraction.py) |
| Advanced RAG | Use embeddings, chunking, Chroma, similarity search, and structured retrieval | Task `1`, Optional `Easy 2` | [services/knowledge_base.py](../services/knowledge_base.py) |
| Query translation | Improve search quality with heuristic multi-query expansion | Task `1` | [services/knowledge_base.py](../services/knowledge_base.py) |
| Tool calling | Provide a flow with at least 3 reviewable tool calls | Task `2`, Optional `Medium 6` | [services/copilot.py](../services/copilot.py) |
| Streamlit UI | Show input, JSON, sources, tool trace, validation output, and progress state | Task `5`, Optional `Easy 3` | [components/app_runtime.py](../components/app_runtime.py) |
| Input validation | Empty input handling, missing fields, schema validation, and basic rule checks | Task `4` | [services/schemas.py](../services/schemas.py), [components/app_runtime.py](../components/app_runtime.py) |
| Error handling | Communicate missing API keys, retrieval failures, and validation warnings to the user | Task `4` | [components/app_runtime.py](../components/app_runtime.py), [services/config.py](../services/config.py) |
| Secure API key handling | Use `.env` or Streamlit secrets | Task `4` | [services/config.py](../services/config.py), [components/app_runtime.py](../components/app_runtime.py) |
| Guided demo prompts | Provide sidebar prompts for normal, ambiguous, and missing-field cases | Optional `Easy 4` | [components/app_runtime.py](../components/app_runtime.py) |
| Multi-model selection | Let the reviewer switch among supported OpenAI models | Optional `Medium 1` | [components/app_runtime.py](../components/app_runtime.py) |
| Knowledge base update path | Import standards PDF/Markdown files, convert PDF to Markdown when needed, list stored files, delete selected files, and reset/rebuild the vectorstore | Optional `Medium 2` | [services/knowledge_base.py](../services/knowledge_base.py), [components/app_runtime.py](../components/app_runtime.py), [scripts/import_pdf_to_knowledge_base.py](../scripts/import_pdf_to_knowledge_base.py), [scripts/manage_knowledge_base.py](../scripts/manage_knowledge_base.py) |
| Cached runtime resource | Cache copilot creation to avoid repeated initialization | Optional `Medium 3` | [components/app_runtime.py](../components/app_runtime.py) |
| Verification | Unit test and smoke test evidence | Internal MVP support | [tests/test_schemas.py](../tests/test_schemas.py), [tests/test_knowledge_base.py](../tests/test_knowledge_base.py) |

### 3.2 Assignment Signals Made Explicit

From a reviewer perspective, the table above already makes four signals explicit:
advanced RAG, 3+ tool calls, UI visibility of sources/tool results/progress, and basic reliability
through input validation, error handling, and logging.

---

## 4. Required Tool Calls

The Sprint 2 MVP must expose at least the following 3 tool calls in a way a reviewer can identify.

### 4.1 Required Tools

| Tool Name | Purpose | Input | Output | Pipeline Role | Current Implementation Status |
| --- | --- | --- | --- | --- | --- |
| `retrieve_reference_context(requirement_text, analysis_json)` | Retrieve standards / reference chunks | natural-language requirement, analysis summary JSON | translated queries + retrieved chunks + source metadata | RAG entry point | Currently implemented with a reviewer-friendly tool name in [services/copilot.py](../services/copilot.py) |
| `extract_configuration_parameters(requirement_text, retrieved_context_json, analysis_json)` | Extract the core configuration fields | natural-language requirement, retrieved context JSON, analysis summary JSON | configuration JSON draft | main product generation | Currently implemented with a reviewer-friendly tool name in [services/extraction.py](../services/extraction.py) |
| `validate_configuration_json(config_json, requirement_text, analysis_json, retrieved_context_json)` | Perform schema validation and basic rule checks | generated config JSON, requirement text, analysis summary JSON, retrieved context JSON | validation status, missing fields, warnings | basic reliability layer | Currently implemented as an explicit validation tool in [services/validation.py](../services/validation.py) |
| `analyze_requirement_text(requirement_text)` | Summarize ambiguity, contradiction, and missing information | natural-language requirement | safe analysis summary | optional supporting step | Currently implemented as a supporting step in [services/extraction.py](../services/extraction.py) |

### 4.2 Tool Framing and Runtime Order

- `analyze_requirement_text(...)` is a **supporting step**, not the core feature.
- The reviewer-visible core tools are `retrieve_reference_context(...)`, `extract_configuration_parameters(...)`, and `validate_configuration_json(...)`.
- The current runtime order is: optional analysis -> retrieval with query translation -> parameter extraction -> JSON validation.
- The architectural center of gravity is **retrieval-backed structured JSON generation with validation**, not raw reasoning architecture.

---

## 5. RAG / Knowledge Base Scope

### 5.1 MVP Knowledge Base Scope

The MVP knowledge base should be limited to the following scope.

- current shipped AUTOSAR reference excerpts
- current shipped ISO 26262 reference excerpts
- additional curated AUTOSAR, ASPICE, ISO 21434, or other automotive software standards excerpts when they are added under `data/standards/`
- optional project-specific reference notes when intentionally curated into the same corpus

In this MVP, AUTOSAR / ISO 26262 should be treated as **supporting context** only, and the
project should not be framed as a full standards engine.

### 5.2 Retrieval Expectations

The RAG pipeline should clearly include the following implementation elements.

- embeddings
- chunking
- vector indexing with Chroma
- similarity search
- query translation
- structured retrieval payload

Current implementation evidence:

- [services/knowledge_base.py](../services/knowledge_base.py)
- [data/standards/autosar_error_handling.md](../data/standards/autosar_error_handling.md)
- [data/standards/iso26262_error_handling.md](../data/standards/iso26262_error_handling.md)

### 5.3 Retrieval Framing

This MVP should be described using wording like the following.

- “reference guidance retrieval”
- “supporting context retrieval”
- “curated excerpts”
- “standards and supporting reference context”

The following phrasing should be avoided.

- “guaranteed compliance validation”
- “expert-level standards verification”
- “full safety compliance assessment”

---

## 6. UI Requirements

The current UI is Streamlit-based, and for the Sprint 2 assignment it should prioritize the following items.

### 6.1 Required UI Elements

| UI Element | Requirement | Current Evidence |
| --- | --- | --- |
| Requirement input | Let the user enter a natural-language configuration requirement | [components/app_runtime.py](../components/app_runtime.py) |
| Generated configuration JSON | Display the structured JSON draft | [components/app_runtime.py](../components/app_runtime.py) |
| Retrieved sources/context | Show source, excerpt, and query | [components/app_runtime.py](../components/app_runtime.py) |
| Tool call results / trace | Show which tools ran and what they returned | [components/app_runtime.py](../components/app_runtime.py) |
| Validation result | Show schema/basic rule check output | [components/app_runtime.py](../components/app_runtime.py) |
| Progress state | Show a loading spinner or progress indicator | [components/app_runtime.py](../components/app_runtime.py) |
| Safe summary | Show an analysis summary instead of internal reasoning if needed | [components/app_runtime.py](../components/app_runtime.py) |

### 6.2 UI Guidance

The UI should target the following.

- No chain-of-thought exposure
- Only a safe analysis summary is shown
- JSON, sources, validation output, and tool trace are visible on one screen for the reviewer
- Loading state is visible
- Missing API keys or runtime failures produce clear user-facing messages

---

## 7. Validation / Reliability Requirements

Validation in the Sprint 2 MVP is limited to a **basic reliability level**.

### 7.1 Required Validation

- Pydantic schema validation
- required field visibility
- basic rule checks
  - missing `ID`
  - missing `Frequency`
  - contradictory timing hints
  - unsupported or unresolved class/play type
- user-visible validation errors or warnings

### 7.2 Error Handling

The MVP must at minimum handle the following cases.

- empty input
- missing API key
- retrieval failure
- parsing / validation failure
- partial output case

### 7.3 Logging

The MVP does not need production observability, but it should at minimum log the following items.

- request started
- selected model
- tool execution order
- retrieval result count
- validation status
- recoverable error message

The current implementation includes MVP-level logging for request start/end, tool execution,
retrieval count, validation status, and recoverable errors.

### 7.4 Secure API Key Handling

The current runtime accepts the following API key sources.

- `.env`
- Streamlit secrets
- an existing `st.session_state["OPENAI_API_KEY"]` entry in the active Streamlit session

Current implementation evidence:

- [services/config.py](../services/config.py)
- [components/app_runtime.py](../components/app_runtime.py)

### 7.5 Quality Requirements

In addition to functional requirements, the Sprint 2 MVP explicitly manages the following quality requirements.

| Quality Requirement ID | Requirement | Rationale | Current Evidence / Expected Module Impact |
| --- | --- | --- | --- |
| `QR-CODE-001` | A single function should generally stay under 200 lines. If it exceeds that, split it into helper functions or service units. | Improves readability, testability, and change safety for reviewers | [services/copilot.py](../services/copilot.py), [components/app_runtime.py](../components/app_runtime.py), [services/extraction.py](../services/extraction.py) |
| `QR-UI-002` | If no API key is available, the app must block execution immediately and show a clear warning/error dialog. | Reduces invalid runtime states and prevents reviewer demo failures | [components/app_runtime.py](../components/app_runtime.py), [services/config.py](../services/config.py) |
| `QR-MNT-003` | Reduce duplicated logic and duplicated wording, and consolidate shared logic into shared helpers/services. | Improves maintainability and consistency | [services/copilot.py](../services/copilot.py), [services/validation.py](../services/validation.py), [components/app_runtime.py](../components/app_runtime.py) |
| `QR-DOC-004` | Keep the README concise and focused on quick start, structure, and the main demo flow. Move detailed design/SRS content into `docs/`. | Helps reviewers find high-signal information quickly | [README.md](../README.md), [docs/Sp2-Proj_SRS.md](./Sp2-Proj_SRS.md) |

Interpretation of the quality requirements:

- `QR-CODE-001` is based on **function length**, not file length.
- `QR-UI-002` requires a **user-visible warning/error state**, not just a log entry.
- `QR-MNT-003` means repeated validation wording, tool trace formatting, and error message generation should be consolidated when duplication appears.
- `QR-DOC-004` means the README should stay focused on product overview and run instructions, while longer design explanations belong under `docs/`.

---

## 8. Verification Status

### 8.1 Current Test Evidence

The current implementation includes the following verification.

- [tests/test_schemas.py](../tests/test_schemas.py)
  - schema normalization
  - unresolved field detection
- [tests/test_knowledge_base.py](../tests/test_knowledge_base.py)
  - seed corpus loading
  - translated query expansion
  - PDF / Markdown import path
  - KB reset / list / selective delete path
  - vectorstore rebuild behavior
- [tests/test_validation.py](../tests/test_validation.py)
  - incomplete / needs_review / ready validation states
  - basic rule checks for cyclic, conflict, request-response mismatch
- [tests/test_tools.py](../tests/test_tools.py)
  - explicit tool naming
  - retrieval / extraction / validation tool payload structure
- [tests/test_copilot.py](../tests/test_copilot.py)
  - empty input handling
  - success path orchestration
  - analysis / retrieval / extraction / validation failure handling
- [tests/test_app_runtime.py](../tests/test_app_runtime.py)
  - API key fallback logic
  - missing API key UI error
  - spinner usage
  - demo prompt coverage
  - knowledge base update panel
- [tests/test_config.py](../tests/test_config.py)
  - logging setup
  - secure API key lookup
- [tests/test_quality_guards.py](../tests/test_quality_guards.py)
  - function length guard
  - concise README guard
- [scripts/run_smoke_test.py](../scripts/run_smoke_test.py)
  - real OpenAI call + Chroma retrieval smoke test
  - writes the latest report to [docs/smoke_test_report.md](./smoke_test_report.md)
  - writes the raw result to [docs/smoke_test_report.json](./smoke_test_report.json)

Current result:

- `40 passed`

### 8.2 Smoke Test Evidence

The following items are verified by [scripts/run_smoke_test.py](../scripts/run_smoke_test.py), which uses a real OpenAI call and Chroma retrieval and writes the latest result to [docs/smoke_test_report.md](./smoke_test_report.md).

- app runtime dependency path works correctly
- copilot creation works correctly
- sample requirement input works correctly
- retrieval works correctly
- extraction works correctly
- validation response works correctly

### 8.3 Recommended Reviewer Demo

The final demo should clearly show the following flow.

1. Enter configuration requirement text
2. Run the retrieval tool
3. Run the extraction tool
4. Run the validation tool
5. Show generated JSON + sources + validation output

---

## 9. Out of Scope

The following items are out of scope for the Sprint 2 MVP.

- full AUTOSAR / ISO 26262 compliance engine
- full PFSP schema coverage
- multi-hop retrieval orchestration
- advanced LLM decomposition retrieval
- automated Ragas benchmark pipeline
- LangGraph state machine migration
- export / feedback / artifact download
- production deployment features
- multilingual optimization
- large-scale standards corpus lifecycle management
- guaranteed self-healing retry loops

These items are left as future work and should not appear as central MVP capabilities.

---

## 10. Minimum Success Criteria

The Sprint 2 MVP is considered successful if it satisfies the following checklist.

- The app runs.
- The user can enter configuration requirement text.
- The app retrieves relevant reference context.
- At least 3 tool calls are identifiable to the reviewer.
- The app generates a configuration JSON draft.
- The app validates the JSON with Pydantic and basic rule checks.
- The UI shows source snippets.
- The UI shows tool results / trace.
- The UI shows validation results.
- The UI shows a loading state.
- Empty input or API/retrieval failure is clearly communicated to the user.

This section should be usable directly as a reviewer checklist.

---

## 11. Current Implementation Notes

### Implemented and Stable

- The automotive software configuration use case, Pydantic-based structured JSON generation, Streamlit UI, Chroma retrieval,
  heuristic query translation, tool trace visibility, and automated tests are implemented and in use.
- Reviewer-friendly tool names are exposed in both the runtime flow and the UI:
  `retrieve_reference_context(...)`, `extract_configuration_parameters(...)`, and `validate_configuration_json(...)`.
- Error handling, logging, API key gating, demo prompts, and quality guard tests are part of the current MVP.

Relevant modules:

- [services/schemas.py](../services/schemas.py)
- [services/knowledge_base.py](../services/knowledge_base.py)
- [services/copilot.py](../services/copilot.py)
- [components/app_runtime.py](../components/app_runtime.py)

### Deliberately Simplified

- The project is framed as **reference-guided validation**, not standards-compliance validation.
- AUTOSAR / ISO 26262 are used as small supporting excerpts, not as a full standards engine.
- Validation remains at the MVP level: schema parsing, required fields, basic rule checks,
  and user-visible warnings.

Relevant modules:

- [services/copilot.py](../services/copilot.py)
- [services/validation.py](../services/validation.py)
- [README.md](../README.md)

### Current Gaps

- Explicit rate limiting is not implemented.
- Manual PDF / Markdown import, list/delete management, and vectorstore reset/rebuild are implemented, but automated or continuously synchronized corpus updates are not.
- In-app RAG evaluation, remote MCP integration, export flows, and advanced indexing are not part of the current app runtime.

Relevant modules:

- [services/config.py](../services/config.py)
- [services/knowledge_base.py](../services/knowledge_base.py)
- [components/app_runtime.py](../components/app_runtime.py)

### Out-of-scope / Future work

- full compliance engine
- full PFSP schema coverage
- advanced multi-hop retrieval
- Ragas automation inside the app
- LangGraph migration
- export / deployment / large corpus management

These items remain clearly outside the MVP core in both the implementation and the documentation.

---

## One-Line Summary

> The most appropriate framing for the Sprint 2 MVP of PFSP AI Configuration Copilot is an **automotive software configuration AI app that performs standards/reference retrieval + structured JSON generation + basic validation and shows the result together with source/context in a Streamlit UI**.
