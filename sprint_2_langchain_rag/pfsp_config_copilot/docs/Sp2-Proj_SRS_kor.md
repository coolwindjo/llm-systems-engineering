# Sp2-Proj_SRS: PFSP AI Configuration Copilot (Sprint 2 MVP)

## 1. Document Status

- Project: `PFSP AI Configuration Copilot`
- Location: `sprint_2_langchain_rag/pfsp_config_copilot`
- Document version: `0.4`
- Project status: `Working Sprint 2 MVP`
- Updated: `2026-03-18`
- Audience: Sprint 2 reviewer, project owner, solo implementer

이 문서는 현재 구현된 MVP를 기준으로, 프로젝트를 Sprint 2 “Building Applications with AI” 과제 요구사항에 맞게 다시 정렬한 SRS입니다.

이 문서는 **Sprint 2 MVP 범위 문서**이며, 최종 제품 전체 SRS의 source of truth는 다음 파일입니다.

- `/workspace/capstone_design/PFSP_AI_Configuration_Copilot_SRS_ModuleOnly_v.2.2.json`

즉, 이 문서에서 의도적으로 축소되거나 de-emphasize된 요구사항이 사라지는 것이 아니라,
**최종 캡스톤 SRS 문서에서 계속 관리되는 것**을 전제로 합니다.

중요한 전제:

- 이 프로젝트는 **PFSP 중심의 automotive software configuration AI 웹 애플리케이션**입니다.
- 이 프로젝트는 **구조화된 configuration JSON draft 생성**을 중심으로 합니다.
- 이 프로젝트는 **advanced RAG + query translation + tool calling**을 핵심 구현 신호로 삼습니다.
- 이 프로젝트는 **production-grade compliance engine**을 목표로 하지 않습니다.

### 1.1 Deferred Requirement Ownership Confirmation

Sprint 2 MVP 문서에서 빠졌거나 중심에서 내려온 요구사항은 아래처럼 최종 SRS에 계속 정의되어 있습니다.

| Deferred / De-emphasized Area in Sprint 2 MVP | Final SRS Reference | Confirmation |
| --- | --- | --- |
| 상세 Error Handling 로직 추출 (`ErrorTypes`, debouncing threshold, `ReactionModes/States`) | `REQ-MOD-201` | 최종 SRS에 정의되어 있음 |
| AUTOSAR / ISO 26262 supporting-context 사용 | `REQ-RAG-403` | 최종 SRS에 정의되어 있음 |
| Smart Converter mapping 생성 | `REQ-MAP-201`, `REQ-MAP-202`, `REQ-MAP-203` | 최종 SRS에 정의되어 있음 |
| Field-Level Traceability (`source_ref`, `traceability_id`) | `REQ-TRC-401` | 최종 SRS에 정의되어 있음 |
| 강한 Two-Phase Reasoning 구조 및 Separation of Concerns | `REQ-PRO-501`, `REQ-PRO-502`, `REQ-PRO-503` | 최종 SRS에 정의되어 있음 |

정리하면:

- `Sp2-Proj_SRS.md`는 Sprint 2 과제 제출용 MVP 범위를 설명합니다.
- `PFSP_AI_Configuration_Copilot_SRS_ModuleOnly_v.2.2.json`는 최종 캡스톤 단계에서 관리될 전체 모듈 요구사항을 유지합니다.
- 따라서 Sprint 2 문서에서 빠진 SRS는 **삭제된 것이 아니라 최종 SRS로 위임된 것**입니다.

---

## 2. Project Goal

PFSP AI Configuration Copilot의 Sprint 2 MVP 목표는 다음과 같습니다.

사용자가 자연어로 configuration requirement를 입력하면,

1. 관련 standards 및 reference 문서를 검색하고,
2. 검색된 문맥을 바탕으로 핵심 configuration 파라미터를 구조화된 JSON draft로 생성하고,
3. Pydantic 및 기본 규칙 검사로 초안의 완성도를 확인하며,
4. Streamlit UI에서 source context, tool call 결과, validation 결과를 함께 보여주는

**target platform을 대상으로 한 automotive software configuration AI 웹 앱**을 구현하는 것입니다.

이 MVP의 핵심 가치는 다음에 있습니다.

- target configuration schema에 맞는 **structured output**
- Chroma 기반 **retrieval pipeline**
- query translation을 포함한 **advanced RAG**
- 최소 3개의 **명시적 tool call**
- 사용자가 intermediate artifacts를 직접 볼 수 있는 **설명 가능한 UI**

---

## 3. Sprint 2 MVP Scope

현재 Sprint 2 MVP에서 반드시 포함해야 하는 범위는 아래와 같습니다.

### 3.1 Scope and Direct Assignment Mapping

| Scope Item | Description | Related Assignment No. | Current Module / Evidence |
| --- | --- | --- | --- |
| Domain-specific application | target-platform configuration workflow를 다루는 automotive software configuration 특화 앱 | Task `3` | [README.md](../README.md), [app.py](../app.py) |
| Structured output | `ServiceName`, `ID`, `Class`, `Frequency`, `PlayType` 기반 JSON draft 생성 | Task `3` | [services/schemas.py](../services/schemas.py), [services/extraction.py](../services/extraction.py) |
| Advanced RAG | embeddings, chunking, Chroma, similarity search, structured retrieval 사용 | Task `1`, Optional `Easy 2` | [services/knowledge_base.py](../services/knowledge_base.py) |
| Query translation | heuristic multi-query expansion으로 검색 개선 | Task `1` | [services/knowledge_base.py](../services/knowledge_base.py) |
| Tool calling | 최소 3개의 reviewable tool call 흐름 | Task `2`, Optional `Medium 6` | [services/copilot.py](../services/copilot.py) |
| Streamlit UI | 입력, JSON, sources, tool trace, validation 결과, progress state 표시 | Task `5`, Optional `Easy 3` | [components/app_runtime.py](../components/app_runtime.py) |
| Input validation | 빈 입력, 누락 필드, schema validation, basic rule checks | Task `4` | [services/schemas.py](../services/schemas.py), [components/app_runtime.py](../components/app_runtime.py) |
| Error handling | API key 미설정, retrieval 실패, validation 경고를 사용자에게 전달 | Task `4` | [components/app_runtime.py](../components/app_runtime.py), [services/config.py](../services/config.py) |
| Secure API key handling | `.env` 또는 Streamlit secrets 사용 | Task `4` | [services/config.py](../services/config.py), [components/app_runtime.py](../components/app_runtime.py) |
| Guided demo prompts | 정상/모호/누락 케이스용 sidebar prompt 제공 | Optional `Easy 4` | [components/app_runtime.py](../components/app_runtime.py) |
| Multi-model selection | reviewer가 지원 모델을 전환할 수 있도록 제공 | Optional `Medium 1` | [components/app_runtime.py](../components/app_runtime.py) |
| Knowledge base update path | standards PDF/Markdown 파일을 import 하고, 필요 시 Markdown 변환 후, 저장된 파일 목록 조회, 개별 삭제, vectorstore reset/rebuild 수행 | Optional `Medium 2` | [services/knowledge_base.py](../services/knowledge_base.py), [components/app_runtime.py](../components/app_runtime.py), [scripts/import_pdf_to_knowledge_base.py](../scripts/import_pdf_to_knowledge_base.py), [scripts/manage_knowledge_base.py](../scripts/manage_knowledge_base.py) |
| Cached runtime resource | copilot 생성 과정을 cache 하여 반복 초기화 축소 | Optional `Medium 3` | [components/app_runtime.py](../components/app_runtime.py) |
| Verification | unit test + smoke test evidence | Internal MVP support | [tests/test_schemas.py](../tests/test_schemas.py), [tests/test_knowledge_base.py](../tests/test_knowledge_base.py) |

### 3.2 Assignment Signals Made Explicit

Sprint 2 reviewer 관점에서는 위 표만으로도 다음 4가지가 이미 분명히 드러납니다.
advanced RAG, 3+ tool calls, sources/tool results/progress 가 보이는 UI, 그리고 input validation,
error handling, logging 으로 구성된 basic reliability 입니다.

---

## 4. Required Tool Calls

Sprint 2 MVP는 최소 아래 3개의 tool call을 reviewer가 식별할 수 있어야 합니다.

### 4.1 Required Tools

| Tool Name | Purpose | Input | Output | Pipeline Role | Current Implementation Status |
| --- | --- | --- | --- | --- | --- |
| `retrieve_reference_context(requirement_text, analysis_json)` | standards / reference chunk 검색 | natural-language requirement, analysis summary JSON | translated queries + retrieved chunks + source metadata | RAG entry point | 현재 [services/copilot.py](../services/copilot.py)에서 reviewer-friendly tool 명으로 구현됨 |
| `extract_configuration_parameters(requirement_text, retrieved_context_json, analysis_json)` | 핵심 configuration 필드 추출 | natural-language requirement, retrieved context JSON, analysis summary JSON | configuration JSON draft | main product generation | 현재 [services/extraction.py](../services/extraction.py)에서 reviewer-friendly tool 명으로 구현됨 |
| `validate_configuration_json(config_json, requirement_text, analysis_json, retrieved_context_json)` | schema validation + basic rule checks | generated config JSON, requirement text, analysis summary JSON, retrieved context JSON | validation status, missing fields, warnings | basic reliability layer | 현재 [services/validation.py](../services/validation.py)에서 명시적 validation tool로 구현됨 |
| `analyze_requirement_text(requirement_text)` | 모호성, 충돌, 누락 정보 요약 | natural-language requirement | safe analysis summary | optional supporting step | 현재 [services/extraction.py](../services/extraction.py)에서 보조 단계로 구현됨 |

### 4.2 Tool Framing and Runtime Order

- `analyze_requirement_text(...)`는 **보조 단계**이지 중심 기능이 아닙니다.
- reviewer가 먼저 인식해야 할 core tool은 `retrieve_reference_context(...)`, `extract_configuration_parameters(...)`, `validate_configuration_json(...)` 입니다.
- 현재 runtime order 는 optional analysis -> retrieval with query translation -> parameter extraction -> JSON validation 입니다.
- 문서의 중심축은 raw reasoning architecture 가 아니라 **retrieval-backed structured JSON generation with validation** 입니다.

---

## 5. RAG / Knowledge Base Scope

### 5.1 MVP Knowledge Base Scope

MVP의 지식 베이스는 아래 수준으로 제한합니다.

- 현재 저장소에 포함된 AUTOSAR reference excerpt
- 현재 저장소에 포함된 ISO 26262 reference excerpt
- `data/standards/` 아래에 추가되는 AUTOSAR, ASPICE, ISO 21434, 기타 automotive software standard 발췌
- 의도적으로 같은 corpus에 포함시키는 경우에 한해 project-specific reference note

MVP에서 AUTOSAR / ISO 26262는 **supporting context**일 뿐이며,
프로젝트를 full standards engine으로 설명하지 않습니다.

### 5.2 Retrieval Expectations

RAG 파이프라인은 아래 구현 요소를 분명히 포함해야 합니다.

- embeddings
- chunking
- vector indexing with Chroma
- similarity search
- query translation
- structured retrieval payload

현재 구현 근거:

- [services/knowledge_base.py](../services/knowledge_base.py)
- [data/standards/autosar_error_handling.md](../data/standards/autosar_error_handling.md)
- [data/standards/iso26262_error_handling.md](../data/standards/iso26262_error_handling.md)

### 5.3 Retrieval Framing

이 MVP는 다음처럼 설명하는 것이 적절합니다.

- “reference guidance retrieval”
- “supporting context retrieval”
- “curated excerpts”
- “standards and supporting reference context”

피해야 할 표현:

- “guaranteed compliance validation”
- “expert-level standards verification”
- “full safety compliance assessment”

---

## 6. UI Requirements

현재 UI는 Streamlit 기반이며, Sprint 2 과제 기준으로 아래 항목을 우선적으로 보여야 합니다.

### 6.1 Required UI Elements

| UI Element | Requirement | Current Evidence |
| --- | --- | --- |
| Requirement input | 사용자가 자연어 configuration requirement를 직접 입력 | [components/app_runtime.py](../components/app_runtime.py) |
| Generated configuration JSON | 구조화된 JSON draft 표시 | [components/app_runtime.py](../components/app_runtime.py) |
| Retrieved sources/context | 검색된 source, excerpt, query 표시 | [components/app_runtime.py](../components/app_runtime.py) |
| Tool call results / trace | 어떤 tool이 실행되었고 무엇을 반환했는지 표시 | [components/app_runtime.py](../components/app_runtime.py) |
| Validation result | schema/basic rule check 결과 표시 | [components/app_runtime.py](../components/app_runtime.py) |
| Progress state | loading spinner 또는 progress indicator 표시 | [components/app_runtime.py](../components/app_runtime.py) |
| Safe summary | 필요 시 internal reasoning 대신 analysis summary 표시 | [components/app_runtime.py](../components/app_runtime.py) |

### 6.2 UI Guidance

UI는 아래를 목표로 합니다.

- chain-of-thought 노출 없음
- safe analysis summary만 노출
- JSON, sources, validation, tool trace를 reviewer가 한 화면에서 확인 가능
- loading 상태가 보임
- API key 부재 또는 실행 실패 시 명확한 사용자 메시지 제공

---

## 7. Validation / Reliability Requirements

Sprint 2 MVP에서 validation은 **기본 신뢰성 확보 수준**으로 제한합니다.

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

MVP는 아래 상황을 최소한 처리해야 합니다.

- empty input
- missing API key
- retrieval failure
- parsing / validation failure
- partial output case

### 7.3 Logging

MVP는 production observability까지는 아니더라도, 최소한 아래 로그는 남길 수 있어야 합니다.

- request started
- selected model
- tool execution order
- retrieval result count
- validation status
- recoverable error message

현재 구현에는 request start/end, tool execution, retrieval count, validation status, recoverable error를 남기는 MVP 수준 logging이 포함되어 있습니다.

### 7.4 Secure API Key Handling

현재 runtime은 아래 API key source를 지원합니다.

- `.env`
- Streamlit secrets
- active Streamlit session 안의 `st.session_state["OPENAI_API_KEY"]`

현재 구현 근거:

- [services/config.py](../services/config.py)
- [components/app_runtime.py](../components/app_runtime.py)

### 7.5 Quality Requirements

Sprint 2 MVP는 기능 요구사항 외에도 아래 품질 요구사항을 명시적으로 관리합니다.

| Quality Requirement ID | Requirement | Rationale | Current Evidence / Expected Module Impact |
| --- | --- | --- | --- |
| `QR-CODE-001` | 단일 함수는 원칙적으로 200 lines 이하를 유지한다. 초과 시 helper function 또는 service 단위로 분리한다. | reviewer가 읽기 쉽고, 테스트 가능성과 변경 안전성을 높이기 위함 | [services/copilot.py](../services/copilot.py), [components/app_runtime.py](../components/app_runtime.py), [services/extraction.py](../services/extraction.py) |
| `QR-UI-002` | API key가 없을 때 앱은 즉시 진행을 막고, 명확한 warning/error dialog를 표시한다. | 잘못된 실행 상태를 줄이고 reviewer demo 실패를 방지하기 위함 | [components/app_runtime.py](../components/app_runtime.py), [services/config.py](../services/config.py) |
| `QR-MNT-003` | 중복 로직과 중복 문구를 줄이고, 공통 로직은 shared helper/service로 통합한다. | 유지보수성과 일관성을 높이기 위함 | [services/copilot.py](../services/copilot.py), [services/validation.py](../services/validation.py), [components/app_runtime.py](../components/app_runtime.py) |
| `QR-DOC-004` | README는 빠른 실행, 구조, 핵심 데모 흐름 중심의 concise 문서로 유지하고, 상세 설계/SRS는 `docs/`로 분리한다. | reviewer가 핵심 정보를 빠르게 찾을 수 있게 하기 위함 | [README.md](../README.md), [docs/Sp2-Proj_SRS.md](./Sp2-Proj_SRS.md) |

품질 요구사항 해석:

- `QR-CODE-001`은 파일 길이가 아니라 **함수 길이** 기준입니다.
- `QR-UI-002`는 단순 로그가 아니라 **사용자에게 보이는 경고/에러 표시**를 요구합니다.
- `QR-MNT-003`은 validation wording, tool trace formatting, error message 생성 로직 등에서 반복이 생기면 정리 대상임을 의미합니다.
- `QR-DOC-004`는 README를 제품 소개 및 실행 문서로 제한하고, 장문의 설계 설명은 `docs/` 아래 문서가 담당한다는 원칙입니다.

---

## 8. Verification Status

### 8.1 Current Test Evidence

현재 구현에는 아래 검증이 존재합니다.

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
  - 최신 결과를 [docs/smoke_test_report.md](./smoke_test_report.md)에 기록
  - raw result를 [docs/smoke_test_report.json](./smoke_test_report.json)에 기록

현재 결과:

- `40 passed`

### 8.2 Smoke Test Evidence

[scripts/run_smoke_test.py](../scripts/run_smoke_test.py)는 실제 OpenAI 호출 + Chroma retrieval 기반으로 아래 항목을 검증하며, 최신 결과는 [docs/smoke_test_report.md](./smoke_test_report.md)에 기록됩니다.

- app runtime dependency path 정상
- copilot 생성 정상
- sample requirement 입력 정상
- retrieval 정상
- extraction 정상
- validation response 정상

### 8.3 Recommended Reviewer Demo

최종 데모에서는 아래 흐름이 명확히 보여야 합니다.

1. configuration requirement text 입력
2. retrieval tool 호출
3. extraction tool 호출
4. validation tool 호출
5. generated JSON + sources + validation 표시

---

## 9. Out of Scope

아래 항목은 Sprint 2 MVP 범위 밖입니다.

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

이 항목들은 future work로 남겨두며, MVP 설명에서 중심 기능처럼 보이지 않도록 해야 합니다.

---

## 10. Minimum Success Criteria

Sprint 2 MVP는 아래 체크리스트를 만족하면 성공으로 간주합니다.

- 앱이 실행된다.
- 사용자가 configuration requirement text를 입력할 수 있다.
- 앱이 관련 reference context를 retrieval 한다.
- 최소 3개의 tool call이 reviewer에게 식별 가능하다.
- 앱이 configuration JSON draft를 생성한다.
- 앱이 Pydantic + basic rule checks로 JSON을 검증한다.
- UI가 source snippets를 보여준다.
- UI가 tool result / trace를 보여준다.
- UI가 validation 결과를 보여준다.
- UI가 loading state를 보여준다.
- 빈 입력 또는 API/retrieval failure를 사용자에게 명확히 알린다.

이 섹션은 reviewer checklist로 바로 사용될 수 있어야 합니다.

---

## 11. 현재 구현 상태 메모

### 구현 완료 및 안정화된 범위

- automotive software configuration use case, Pydantic 기반 structured JSON generation, Streamlit UI, Chroma retrieval,
  heuristic query translation, tool trace visibility, automated tests 가 현재 구현에 포함되어 있습니다.
- reviewer-friendly tool name 이 runtime flow 와 UI 에 직접 노출됩니다:
  `retrieve_reference_context(...)`, `extract_configuration_parameters(...)`, `validate_configuration_json(...)`.
- error handling, logging, API key gating, demo prompts, quality guard tests 가 현재 MVP 범위에 포함되어 있습니다.

관련 모듈:

- [services/schemas.py](../services/schemas.py)
- [services/knowledge_base.py](../services/knowledge_base.py)
- [services/copilot.py](../services/copilot.py)
- [components/app_runtime.py](../components/app_runtime.py)

### 의도적으로 단순화한 범위

- 프로젝트는 **reference-guided validation** 으로 설명되며, standards-compliance validation 으로 설명하지 않습니다.
- AUTOSAR / ISO 26262 는 작은 supporting excerpt 로만 사용되며, full standards engine 으로 다루지 않습니다.
- validation 은 MVP 수준인 schema parsing, required fields, basic rule checks, user-visible warnings 에 머뭅니다.

관련 모듈:

- [services/copilot.py](../services/copilot.py)
- [services/validation.py](../services/validation.py)
- [README.md](../README.md)

### 현재 남아 있는 갭

- explicit rate limiting 은 아직 구현되지 않았습니다.
- manual PDF / Markdown import, 목록 조회/개별 삭제, vectorstore reset/rebuild 는 구현되었지만, automated 또는 continuously synchronized corpus update 는 아직 아닙니다.
- in-app RAG evaluation, remote MCP integration, export flows, advanced indexing 은 현재 app runtime 에 포함되지 않습니다.

관련 모듈:

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

이 항목들은 현재 구현과 문서 모두에서 MVP 핵심과 분리된 범위로 유지됩니다.

---

## One-Line Summary

> PFSP AI Configuration Copilot의 Sprint 2 MVP는 **standards/reference retrieval + structured JSON generation + basic validation을 수행하고, 그 결과를 Streamlit UI에서 source/context와 함께 보여주는 automotive software configuration AI 앱**으로 정리하는 것이 가장 적절합니다.
