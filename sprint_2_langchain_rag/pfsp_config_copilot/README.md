# PFSP AI Configuration Copilot

Sprint 2 MVP for an automotive software configuration AI web app that generates reviewable configuration JSON drafts for a target platform. In this project, that target product is PFSP (Plugin Fractal Software Platform).
The app centers on advanced RAG, query translation, reviewer-visible tool calls, and basic validation in a Streamlit UI.

Primary references:
- `docs/Sp2-Proj_SRS.md`
- `/workspace/capstone_design/PFSP_AI_Configuration_Copilot_SRS_ModuleOnly_v.2.2.json`

## Quick Start

```bash
cd /workspace/sprint_2_langchain_rag/pfsp_config_copilot
pip install -r requirements.txt
```

Set an API key in `/workspace/.env` or Streamlit secrets.
The runtime also accepts an existing `st.session_state["OPENAI_API_KEY"]` during the current Streamlit session:

```bash
OPENAI_API_KEY=sk-...
```

Run the app:

```bash
streamlit run app.py
```

Run tests:

```bash
pytest tests -q
```
Run the live smoke test:

```bash
python scripts/run_smoke_test.py
```

## Project Scope

- Configuration requirement input, not a general chatbot.
- Advanced RAG with embeddings, chunking, Chroma indexing, similarity search, and query translation.
- Reviewer-visible tool flow:
  `analyze_requirement_text` -> `retrieve_reference_context` -> `extract_configuration_parameters` -> `validate_configuration_json`.
- Structured configuration JSON draft for `ServiceName`, `ID`, `Class`, `Frequency`, `PlayType`.
- Streamlit UI showing JSON, retrieved sources, tool trace, validation status, and progress/loading state.
- MVP-level reliability only: schema validation, basic rule checks, input validation, error handling, and logging.
- Curated automotive software standards corpus with PDF/Markdown import, document listing/deletion, and vectorstore reset/rebuild support. The current repo ships AUTOSAR and ISO 26262 excerpts, and the intended KB scope also includes ASPICE, ISO 21434, and future automotive software standards as curated documents are added under `data/standards/`.

## Verified Behavior

- `app.py` launches the Streamlit UI in `components/app_runtime.py`.
- `services/knowledge_base.py` provides chunk loading, translated queries, structured retrieval payloads, and knowledge-base import/reset/delete management.
- `services/extraction.py`, `services/copilot.py`, and `services/validation.py` implement analysis, orchestration, tool trace, JSON drafting, and validation.
- Missing API key, empty input, retrieval failure, extraction failure, and validation failure return user-visible errors.
- Logging is configured in `services/config.py` and written to `logs/config_copilot.log`.
- Current automated verification result: `40 passed` via `pytest tests -q`.
- A live smoke test script exists at `scripts/run_smoke_test.py`; the latest run report is written to `docs/smoke_test_report.md` and `docs/smoke_test_report.json`.

## Coverage Mapping to `Building Applications with AI`

### 1) Task Requirements

| PDF Requirement No. | Requirement | Coverage |
| --- | --- | --- |
| 1 | RAG Implementation | **Covered**: a small automotive standards/reference knowledge base, embeddings, chunking strategies, and similarity search are implemented in `services/knowledge_base.py`. |
| 2 | Tool Calling | **Covered**: at least 3 domain-relevant tools are implemented: `retrieve_reference_context`, `extract_configuration_parameters`, `validate_configuration_json`. |
| 3 | Domain Specialisation | **Covered**: an automotive software configuration use case for a target platform, a focused automotive standards corpus, a configuration schema, and automotive configuration prompts/responses are implemented. |
| 4 | Technical Implementation | **Covered with one gap**: LangChain, error handling, logging, input validation, and API key management exist; explicit rate limiting is not implemented yet. |
| 5 | User Interface | **Covered**: Streamlit UI shows context, sources, tool results, and progress/loading state. |

### 2) Optional Tasks

| PDF Optional No. | Optional task | Status |
| --- | --- | --- |
| Easy 1 | Add conversation history and export functionality | **Partially covered**: conversation history exists in Streamlit session state; export is not implemented. |
| Easy 2 | Add visualisation of RAG process | **Covered**: translated queries, retrieved chunks, and tool trace are shown in the UI. |
| Easy 3 | Include source citations in responses | **Covered**: source, title, query, and excerpt are displayed for retrieved chunks. |
| Easy 4 | Add an interactive help feature or chatbot guide | **Partially covered**: sidebar flow and demo prompts help the reviewer, but there is no dedicated help guide. |
| Medium 1 | Implement multi-model support | **Covered**: multiple OpenAI chat models are selectable in the Streamlit sidebar. |
| Medium 2 | Add real-time data updates to knowledge base | **Covered**: standards PDF/Markdown files can be imported from the app or CLI, converted when needed, listed, selectively deleted, and used to reset/rebuild the Chroma vectorstore. |
| Medium 3 | Implement advanced caching strategies | **Partially covered**: cached copilot creation exists, but no broader retrieval/result caching strategy. |
| Medium 4 | Add user authentication and personalisation | **Not implemented**. |
| Medium 5 | Calculate and display token usage and costs | **Not implemented**. |
| Medium 6 | Add visualisation of tool call results | **Covered**: tool outputs and trace are displayed in dedicated UI sections. |
| Medium 7 | Implement conversation export in various formats | **Not implemented**. |
| Medium 8 | Connect to tools from a publicly available remote MCP server | **Not implemented**. |
| Hard 1 | Deploy to cloud with proper scaling | **Not implemented**. |
| Hard 2 | Implement advanced indexing | **Not implemented**. |
| Hard 3 | Implement A/B testing for different RAG strategies | **Not implemented**. |
| Hard 4 | Add automated knowledge base updates | **Not implemented**. |
| Hard 5 | Fine-tune the model for your specific domain | **Not implemented**. |
| Hard 6 | Add multi-language support | **Not implemented**. |
| Hard 7 | Implement advanced analytics dashboard | **Not implemented**. |
| Hard 8 | Implement your tools as MCP servers | **Not implemented**. |
| Hard 9 | Implement an evaluation of your RAG system, using RAGAs or otherwise | **Partially covered outside the app**: Sprint 2 evaluation scripts exist in the repository, but are not yet integrated into this app. |

### 3) Evaluation Criteria Coverage

| Criterion | Coverage |
| --- | --- |
| Understanding Core Concepts: basic RAG principles | **Covered**: the app visibly uses embeddings, chunking, vector search, and retrieval-backed generation. |
| Understanding Core Concepts: explain tool calling clearly | **Covered**: reviewer-facing tool names and tool trace expose the orchestration path clearly. |
| Understanding Core Concepts: good code organisation | **Covered**: services, components, docs, and tests are split by responsibility, with quality guard tests. |
| Understanding Core Concepts: identify error scenarios and edge cases | **Covered**: empty input, API key absence, retrieval failure, extraction failure, and validation failure are handled and tested. |
| Technical Implementation: use a front-end library | **Covered**: the project uses Streamlit. |
| Technical Implementation: project works as intended | **Covered**: the app runs as a chat-style configuration copilot and the automated suite passes. |
| Technical Implementation: relevant knowledge base for domain | **Covered**: the app uses a curated automotive standards/reference corpus for the automotive software configuration use case. |
| Technical Implementation: appropriate security considerations | **Partially covered**: API key handling, input validation, and safe summary display exist; rate limiting and stronger project-specific security controls remain open. |
| Reflection and Improvement: understand application problems | **Covered**: README and SRS explicitly describe MVP limits and out-of-scope items. |
| Reflection and Improvement: suggest improvements | **Covered**: SRS includes recommended next steps and implementation change areas. |
| Bonus Points | **Not targeted for this MVP**: the project does not currently claim 2 medium and 1 hard optional task in full. |

## Notes

- The UI shows a safe analysis summary, not hidden chain-of-thought.
- The knowledge base is curated and currently ships AUTOSAR and ISO 26262 excerpts; its intended scope expands as AUTOSAR, ASPICE, ISO 21434, and other automotive software standards are added under `data/standards/`.
- Full AUTOSAR / ISO 26262 compliance behavior remains out of scope for this repository snapshot.
