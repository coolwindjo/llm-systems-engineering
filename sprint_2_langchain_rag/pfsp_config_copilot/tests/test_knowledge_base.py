from __future__ import annotations

from pathlib import Path

from langchain_core.documents import Document

from services.knowledge_base import (
    build_translated_queries,
    delete_reference_document,
    list_reference_documents,
    load_standard_documents,
    prepare_markdown_reference,
    store_reference_document,
    store_uploaded_document,
)
from services.schemas import ConfigurationDraft, RequirementAnalysis


def test_load_standard_documents_reads_seed_corpus() -> None:
    docs = load_standard_documents()
    sources = {doc.metadata["source"] for doc in docs}

    assert len(docs) >= 2
    assert "autosar_error_handling.md" in sources
    assert "iso26262_error_handling.md" in sources


def test_build_translated_queries_adds_context_for_missing_info_and_conflicts() -> None:
    analysis = RequirementAnalysis(
        summary="The requirement mixes cyclic timing with on-change behavior and omits a stable identifier.",
        contradictions=["cyclic timing conflicts with on-change triggering"],
        ambiguities=["service class is not fully explicit"],
        missing_information=["service ID", "diagnostic fallback"],
        follow_up_questions=["Should this be cyclic or on-change?"],
        risk_level="high",
    )
    extraction = ConfigurationDraft(
        ServiceName="TorqueCommandSync",
        ID=None,
        Class="Event",
        Frequency="10 ms",
        PlayType="Cyclic",
    )

    queries = build_translated_queries(
        "Configure TorqueCommandSync as cyclic every 10 ms, but also say it should be on-change.",
        analysis,
        extraction,
    )

    assert queries[0].startswith("Configure TorqueCommandSync")
    assert any("AUTOSAR" in query for query in queries)
    assert any("ISO 26262 requirement ambiguity" in query for query in queries)
    assert any("configuration missing fields" in query for query in queries)


def test_retrieve_deduplicates_and_sorts_structured_chunks(monkeypatch) -> None:
    from services.knowledge_base import StandardsKnowledgeBase

    class FakeVectorStore:
        def similarity_search_with_score(self, query: str, k: int = 2):
            docs = [
                (
                    Document(
                        page_content="First excerpt about cyclic timing.",
                        metadata={
                            "source": "standards_reference.md",
                            "title": "Standards Reference",
                        },
                    ),
                    0.35,
                ),
                (
                    Document(
                        page_content="Second excerpt about project naming.",
                        metadata={"source": "project_notes.md", "title": "Project Notes"},
                    ),
                    0.11,
                ),
                (
                    Document(
                        page_content="First excerpt about cyclic timing.",
                        metadata={
                            "source": "standards_reference.md",
                            "title": "Standards Reference",
                        },
                    ),
                    0.35,
                ),
            ]
            return docs[:k]

    analysis = RequirementAnalysis(
        summary="Safe summary",
        contradictions=[],
        ambiguities=[],
        missing_information=[],
        follow_up_questions=[],
        risk_level="low",
    )
    kb = StandardsKnowledgeBase.__new__(StandardsKnowledgeBase)
    kb.vectorstore = FakeVectorStore()
    monkeypatch.setattr(
        "services.knowledge_base.build_translated_queries",
        lambda user_requirement, analysis, extraction=None: ["timing query", "naming query"],
    )

    translated_queries, chunks = kb.retrieve("Create a cyclic service.", analysis, extraction=None)

    assert translated_queries == ["timing query", "naming query"]
    assert len(chunks) == 2
    assert chunks[0].title == "Project Notes"
    assert chunks[1].title == "Standards Reference"


def test_store_reference_document_copies_pdf_into_standards_dir(tmp_path) -> None:
    source_pdf = tmp_path / "source.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 sample")
    standards_dir = tmp_path / "standards"

    stored_path = store_reference_document(source_pdf, standards_dir=standards_dir)

    assert stored_path == standards_dir / "source.pdf"
    assert stored_path.read_bytes() == b"%PDF-1.4 sample"


def test_store_reference_document_copies_markdown_into_standards_dir(tmp_path) -> None:
    source_md = tmp_path / "source.md"
    source_md.write_text("# source\n", encoding="utf-8")
    standards_dir = tmp_path / "standards"

    stored_path = store_reference_document(source_md, standards_dir=standards_dir)

    assert stored_path == standards_dir / "source.md"
    assert stored_path.read_text(encoding="utf-8") == "# source\n"


def test_store_uploaded_document_writes_pdf_content(tmp_path) -> None:
    standards_dir = tmp_path / "standards"

    stored_path = store_uploaded_document(
        "uploaded.pdf",
        b"%PDF-1.4 uploaded",
        standards_dir=standards_dir,
    )

    assert stored_path == standards_dir / "uploaded.pdf"
    assert stored_path.read_bytes() == b"%PDF-1.4 uploaded"


def test_store_uploaded_document_writes_markdown_content(tmp_path) -> None:
    standards_dir = tmp_path / "standards"

    stored_path = store_uploaded_document(
        "uploaded.md",
        b"# uploaded\n",
        standards_dir=standards_dir,
    )

    assert stored_path == standards_dir / "uploaded.md"
    assert stored_path.read_text(encoding="utf-8") == "# uploaded\n"


def test_prepare_markdown_reference_returns_markdown_path_for_md(tmp_path) -> None:
    markdown_path = tmp_path / "reference.md"
    markdown_path.write_text("# reference\n", encoding="utf-8")

    assert prepare_markdown_reference(markdown_path) == markdown_path


def test_list_reference_documents_reports_pairing_and_indexing(tmp_path) -> None:
    standards_dir = tmp_path / "standards"
    standards_dir.mkdir()
    (standards_dir / "manual_note.md").write_text("# manual\n", encoding="utf-8")
    (standards_dir / "iso_excerpt.PDF").write_bytes(b"%PDF-1.4 sample")
    (standards_dir / "iso_excerpt.md").write_text("# converted\n", encoding="utf-8")

    documents = list_reference_documents(standards_dir)

    assert [doc["name"] for doc in documents] == [
        "iso_excerpt.PDF",
        "iso_excerpt.md",
        "manual_note.md",
    ]
    assert documents[0]["paired_markdown"] == "iso_excerpt.md"
    assert documents[0]["indexed_source"] is False
    assert documents[1]["paired_pdf"] == "iso_excerpt.PDF"
    assert documents[1]["indexed_source"] is True
    assert documents[2]["paired_pdf"] is None


def test_delete_reference_document_removes_pdf_and_paired_markdown(tmp_path) -> None:
    standards_dir = tmp_path / "standards"
    standards_dir.mkdir()
    pdf_path = standards_dir / "iso_excerpt.PDF"
    markdown_path = standards_dir / "iso_excerpt.md"
    pdf_path.write_bytes(b"%PDF-1.4 sample")
    markdown_path.write_text("# converted\n", encoding="utf-8")

    result = delete_reference_document("iso_excerpt.PDF", standards_dir=standards_dir)

    assert result["deleted_documents"] == ["iso_excerpt.PDF", "iso_excerpt.md"]
    assert not pdf_path.exists()
    assert not markdown_path.exists()


def test_delete_reference_document_removes_only_requested_markdown(tmp_path) -> None:
    standards_dir = tmp_path / "standards"
    standards_dir.mkdir()
    pdf_path = standards_dir / "iso_excerpt.PDF"
    markdown_path = standards_dir / "manual_note.md"
    pdf_path.write_bytes(b"%PDF-1.4 sample")
    markdown_path.write_text("# manual\n", encoding="utf-8")

    result = delete_reference_document("manual_note.md", standards_dir=standards_dir)

    assert result["deleted_documents"] == ["manual_note.md"]
    assert pdf_path.exists()
    assert not markdown_path.exists()


def test_rebuild_vectorstore_resets_collection_and_reindexes(monkeypatch) -> None:
    from services.knowledge_base import StandardsKnowledgeBase

    class FakeVectorStore:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.added_documents: list[Document] = []

        def reset_collection(self) -> None:
            self.reset_calls += 1

        def add_documents(self, documents) -> None:
            self.added_documents.extend(documents)

    kb = StandardsKnowledgeBase.__new__(StandardsKnowledgeBase)
    kb.vectorstore = FakeVectorStore()
    monkeypatch.setattr(
        "services.knowledge_base.load_standard_documents",
        lambda standards_dir=Path("."): [
            Document(page_content="A", metadata={"source": "a.md", "title": "A"})
        ],
    )
    monkeypatch.setattr(
        "services.knowledge_base.build_standard_chunks",
        lambda standards_dir=Path("."): [
            Document(page_content="Chunk A", metadata={"source": "a.md", "title": "A"})
        ],
    )

    result = kb.rebuild_vectorstore()

    assert kb.vectorstore.reset_calls == 1
    assert len(kb.vectorstore.added_documents) == 1
    assert result["document_count"] == 1
    assert result["chunk_count"] == 1
    assert result["sources"] == ["a.md"]


def test_import_pdf_and_rebuild_uses_converter_then_reindexes(monkeypatch, tmp_path) -> None:
    from services.knowledge_base import StandardsKnowledgeBase

    class FakeVectorStore:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.added_documents: list[Document] = []

        def reset_collection(self) -> None:
            self.reset_calls += 1

        def add_documents(self, documents) -> None:
            self.added_documents.extend(documents)

    source_pdf = tmp_path / "incoming.pdf"
    source_pdf.write_bytes(b"%PDF-1.4 sample")
    standards_dir = tmp_path / "standards"

    kb = StandardsKnowledgeBase.__new__(StandardsKnowledgeBase)
    kb.vectorstore = FakeVectorStore()
    def fake_convert(pdf_path):
        markdown_path = pdf_path.with_suffix(".md")
        markdown_path.write_text("# converted\n", encoding="utf-8")
        return markdown_path

    monkeypatch.setattr(
        "services.knowledge_base.convert_pdf_to_markdown",
        fake_convert,
    )
    monkeypatch.setattr(
        "services.knowledge_base.load_standard_documents",
        lambda standards_dir=Path("."): [
            Document(page_content="Converted", metadata={"source": "incoming.md", "title": "Incoming"})
        ],
    )
    monkeypatch.setattr(
        "services.knowledge_base.build_standard_chunks",
        lambda standards_dir=Path("."): [
            Document(page_content="Chunk", metadata={"source": "incoming.md", "title": "Incoming"})
        ],
    )

    result = kb.import_pdf_and_rebuild(source_pdf, standards_dir=standards_dir)

    assert result["stored_document_path"].endswith("incoming.pdf")
    assert result["markdown_path"].endswith("incoming.md")
    assert kb.vectorstore.reset_calls == 1
    assert result["document_count"] == 1


def test_import_markdown_and_rebuild_reindexes_without_conversion(monkeypatch, tmp_path) -> None:
    from services.knowledge_base import StandardsKnowledgeBase

    class FakeVectorStore:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.added_documents: list[Document] = []

        def reset_collection(self) -> None:
            self.reset_calls += 1

        def add_documents(self, documents) -> None:
            self.added_documents.extend(documents)

    source_md = tmp_path / "incoming.md"
    source_md.write_text("# incoming\n", encoding="utf-8")
    standards_dir = tmp_path / "standards"

    kb = StandardsKnowledgeBase.__new__(StandardsKnowledgeBase)
    kb.vectorstore = FakeVectorStore()
    monkeypatch.setattr(
        "services.knowledge_base.load_standard_documents",
        lambda standards_dir=Path("."): [
            Document(page_content="Converted", metadata={"source": "incoming.md", "title": "Incoming"})
        ],
    )
    monkeypatch.setattr(
        "services.knowledge_base.build_standard_chunks",
        lambda standards_dir=Path("."): [
            Document(page_content="Chunk", metadata={"source": "incoming.md", "title": "Incoming"})
        ],
    )

    result = kb.import_document_and_rebuild(source_md, standards_dir=standards_dir)

    assert result["stored_document_path"].endswith("incoming.md")
    assert result["markdown_path"].endswith("incoming.md")
    assert kb.vectorstore.reset_calls == 1
    assert result["document_count"] == 1


def test_delete_document_and_rebuild_removes_file_then_reindexes(monkeypatch, tmp_path) -> None:
    from services.knowledge_base import StandardsKnowledgeBase

    class FakeVectorStore:
        def __init__(self) -> None:
            self.reset_calls = 0
            self.added_documents: list[Document] = []

        def reset_collection(self) -> None:
            self.reset_calls += 1

        def add_documents(self, documents) -> None:
            self.added_documents.extend(documents)

    standards_dir = tmp_path / "standards"
    standards_dir.mkdir()
    (standards_dir / "manual_note.md").write_text("# manual\n", encoding="utf-8")

    kb = StandardsKnowledgeBase.__new__(StandardsKnowledgeBase)
    kb.vectorstore = FakeVectorStore()
    monkeypatch.setattr(
        "services.knowledge_base.load_standard_documents",
        lambda standards_dir=Path("."): [],
    )
    monkeypatch.setattr(
        "services.knowledge_base.build_standard_chunks",
        lambda standards_dir=Path("."): [],
    )

    result = kb.delete_document_and_rebuild("manual_note.md", standards_dir=standards_dir)

    assert result["deleted_documents"] == ["manual_note.md"]
    assert kb.vectorstore.reset_calls == 1
    assert result["document_count"] == 0
