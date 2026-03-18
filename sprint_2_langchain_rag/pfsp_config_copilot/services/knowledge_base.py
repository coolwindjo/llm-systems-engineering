from __future__ import annotations

import json
import shutil
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from scripts.convert_pdf_to_md import convert_pdf_to_markdown
from services.config import (
    DEFAULT_EMBEDDING_MODEL,
    STANDARDS_DIR,
    VECTORSTORE_DIR,
)
from services.schemas import ConfigurationDraft, RequirementAnalysis, RetrievedStandardChunk

COLLECTION_NAME = "automotive_config_standards"
SUPPORTED_REFERENCE_SUFFIXES = {".pdf", ".md"}


def load_standard_documents(standards_dir: Path = STANDARDS_DIR) -> list[Document]:
    documents: list[Document] = []
    for path in sorted(standards_dir.glob("*.md")):
        documents.append(
            Document(
                page_content=path.read_text(encoding="utf-8"),
                metadata={"source": path.name, "title": path.stem.replace("_", " ").title()},
            )
        )
    return documents


def _supported_reference_paths(standards_dir: Path = STANDARDS_DIR) -> list[Path]:
    standards_dir.mkdir(parents=True, exist_ok=True)
    return sorted(
        path
        for path in standards_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_REFERENCE_SUFFIXES
    )


def _find_sibling_with_suffix(path: Path, suffix: str) -> Path | None:
    for candidate in sorted(path.parent.glob(f"{path.stem}.*")):
        if candidate.suffix.lower() == suffix.lower():
            return candidate
    return None


def build_text_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=120,
        separators=["\n## ", "\n\n", "\n", ". "],
    )


def build_standard_chunks(standards_dir: Path = STANDARDS_DIR) -> list[Document]:
    splitter = build_text_splitter()
    documents = load_standard_documents(standards_dir)
    return splitter.split_documents(documents)


def _validate_reference_path(path: Path) -> None:
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_REFERENCE_SUFFIXES:
        raise ValueError(
            f"Expected one of {sorted(SUPPORTED_REFERENCE_SUFFIXES)}, got: {path.name}"
        )


def store_reference_document(
    document_path: Path,
    *,
    standards_dir: Path = STANDARDS_DIR,
) -> Path:
    standards_dir.mkdir(parents=True, exist_ok=True)
    if not document_path.exists():
        raise FileNotFoundError(f"Reference file not found: {document_path}")
    _validate_reference_path(document_path)

    target_path = standards_dir / document_path.name
    if document_path.resolve() != target_path.resolve():
        shutil.copyfile(document_path, target_path)
    return target_path


def store_uploaded_document(
    document_name: str,
    document_bytes: bytes,
    *,
    standards_dir: Path = STANDARDS_DIR,
) -> Path:
    standards_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(document_name).name
    _validate_reference_path(Path(safe_name))

    target_path = standards_dir / safe_name
    target_path.write_bytes(document_bytes)
    return target_path


def prepare_markdown_reference(document_path: Path) -> Path:
    if document_path.suffix.lower() == ".pdf":
        return convert_pdf_to_markdown(document_path)
    if document_path.suffix.lower() == ".md":
        return document_path
    raise ValueError(f"Unsupported reference file: {document_path.name}")


def _resolve_reference_document_path(
    document_name_or_path: str | Path,
    *,
    standards_dir: Path = STANDARDS_DIR,
) -> Path:
    standards_dir.mkdir(parents=True, exist_ok=True)
    candidate = Path(document_name_or_path)
    target_path = standards_dir / candidate.name
    if not target_path.exists():
        raise FileNotFoundError(f"Reference file not found: {target_path.name}")
    _validate_reference_path(target_path)
    return target_path


def list_reference_documents(standards_dir: Path = STANDARDS_DIR) -> list[dict]:
    records: list[dict] = []

    for path in _supported_reference_paths(standards_dir):
        derived_markdown = (
            _find_sibling_with_suffix(path, ".md") if path.suffix.lower() == ".pdf" else None
        )
        source_pdf = (
            _find_sibling_with_suffix(path, ".pdf") if path.suffix.lower() == ".md" else None
        )

        records.append(
            {
                "name": path.name,
                "path": str(path),
                "file_type": path.suffix.lower().lstrip("."),
                "indexed_source": path.suffix.lower() == ".md",
                "paired_pdf": source_pdf.name if source_pdf else None,
                "paired_markdown": derived_markdown.name if derived_markdown else None,
                "delete_behavior": (
                    "Deleting this PDF also removes the paired Markdown file."
                    if path.suffix.lower() == ".pdf" and derived_markdown
                    else "Deletes only this file."
                ),
            }
        )

    return records


def delete_reference_document(
    document_name_or_path: str | Path,
    *,
    standards_dir: Path = STANDARDS_DIR,
) -> dict:
    target_path = _resolve_reference_document_path(
        document_name_or_path,
        standards_dir=standards_dir,
    )
    deleted_paths = [target_path]

    # A converted Markdown file is coupled to its sibling PDF in this MVP.
    if target_path.suffix.lower() == ".pdf":
        paired_markdown = _find_sibling_with_suffix(target_path, ".md")
        if paired_markdown:
            deleted_paths.append(paired_markdown)

    deleted_names: list[str] = []
    for path in deleted_paths:
        path.unlink(missing_ok=True)
        deleted_names.append(path.name)

    return {
        "requested_document": target_path.name,
        "deleted_documents": deleted_names,
        "deleted_count": len(deleted_names),
    }


def build_translated_queries(
    user_requirement: str,
    analysis: RequirementAnalysis,
    extraction: ConfigurationDraft | None = None,
) -> list[str]:
    raw_queries = [
        user_requirement,
        "AUTOSAR service communication publication subscription error handling mode manager diagnostics safe state",
    ]

    if extraction and (extraction.ServiceName or extraction.Class or extraction.PlayType):
        raw_queries.append(
            " ".join(
                part
                for part in [
                    "AUTOSAR",
                    extraction.Class or "service",
                    extraction.PlayType or "triggering",
                    extraction.ServiceName or "",
                    "configuration timing diagnostics",
                ]
                if part
            )
        )

    if analysis.missing_information:
        raw_queries.append(
            "ISO 26262 requirement ambiguity "
            + " ".join(analysis.missing_information[:4])
        )

    if analysis.contradictions:
        raw_queries.append(
            "AUTOSAR communication contradiction resolution "
            + " ".join(analysis.contradictions[:4])
        )

    unresolved = extraction.unresolved_fields() if extraction else []
    if unresolved:
        raw_queries.append("configuration missing fields " + " ".join(unresolved))

    normalized_queries: list[str] = []
    seen: set[str] = set()
    for query in raw_queries:
        normalized = " ".join(query.split()).strip()
        if not normalized:
            continue
        if normalized not in seen:
            normalized_queries.append(normalized)
            seen.add(normalized)
    return normalized_queries


class StandardsKnowledgeBase:
    def __init__(
        self,
        api_key: str,
        persist_directory: Path = VECTORSTORE_DIR,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:
        self.api_key = api_key
        self.persist_directory = persist_directory
        self.embedding_model = embedding_model
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        self.embeddings = OpenAIEmbeddings(
            model=self.embedding_model,
            api_key=self.api_key,
        )
        self.vectorstore = self._build_vectorstore()

    def _create_vectorstore(self) -> Chroma:
        return Chroma(
            collection_name=COLLECTION_NAME,
            persist_directory=str(self.persist_directory),
            embedding_function=self.embeddings,
        )

    def _index_documents(self, standards_dir: Path = STANDARDS_DIR) -> dict:
        documents = load_standard_documents(standards_dir)
        chunks = build_standard_chunks(standards_dir)
        if chunks:
            self.vectorstore.add_documents(chunks)
        return {
            "document_count": len(documents),
            "chunk_count": len(chunks),
            "sources": [doc.metadata["source"] for doc in documents],
        }

    def _build_vectorstore(self) -> Chroma:
        vectorstore = self._create_vectorstore()
        current_count = vectorstore._collection.count()
        if current_count == 0:
            self.vectorstore = vectorstore
            self._index_documents()

        return vectorstore

    def retrieve(
        self,
        user_requirement: str,
        analysis: RequirementAnalysis,
        extraction: ConfigurationDraft | None = None,
        k_per_query: int = 2,
    ) -> tuple[list[str], list[RetrievedStandardChunk]]:
        translated_queries = build_translated_queries(user_requirement, analysis, extraction)
        collected: dict[str, RetrievedStandardChunk] = {}

        for query in translated_queries:
            for doc, score in self.vectorstore.similarity_search_with_score(query, k=k_per_query):
                key = json.dumps(
                    {
                        "source": doc.metadata.get("source", ""),
                        "excerpt": doc.page_content[:180],
                    },
                    sort_keys=True,
                )
                if key not in collected:
                    collected[key] = RetrievedStandardChunk(
                        source=str(doc.metadata.get("source", "unknown")),
                        title=str(doc.metadata.get("title", "Standard excerpt")),
                        query=query,
                        excerpt=doc.page_content[:320].strip(),
                        score=float(score),
                    )

        chunks = sorted(
            collected.values(),
            key=lambda item: item.score if item.score is not None else 9999.0,
        )
        return translated_queries, list(chunks[:6])

    def corpus_sources(self) -> list[str]:
        return [doc.metadata["source"] for doc in load_standard_documents()]

    def list_reference_documents(self, standards_dir: Path = STANDARDS_DIR) -> list[dict]:
        return list_reference_documents(standards_dir)

    def rebuild_vectorstore(self, standards_dir: Path = STANDARDS_DIR) -> dict:
        self.vectorstore.reset_collection()
        index_stats = self._index_documents(standards_dir)
        return {
            "collection_name": COLLECTION_NAME,
            **index_stats,
        }

    def reset_knowledge_base(self, standards_dir: Path = STANDARDS_DIR) -> dict:
        return self.rebuild_vectorstore(standards_dir)

    def import_pdf_and_rebuild(
        self,
        pdf_path: Path,
        *,
        standards_dir: Path = STANDARDS_DIR,
    ) -> dict:
        return self.import_document_and_rebuild(pdf_path, standards_dir=standards_dir)

    def import_document_and_rebuild(
        self,
        document_path: Path,
        *,
        standards_dir: Path = STANDARDS_DIR,
    ) -> dict:
        stored_document_path = store_reference_document(
            document_path,
            standards_dir=standards_dir,
        )
        markdown_path = prepare_markdown_reference(stored_document_path)
        rebuild_stats = self.rebuild_vectorstore(standards_dir)
        return {
            "stored_document_path": str(stored_document_path),
            "markdown_path": str(markdown_path),
            **rebuild_stats,
        }

    def import_pdf_bytes_and_rebuild(
        self,
        pdf_name: str,
        pdf_bytes: bytes,
        *,
        standards_dir: Path = STANDARDS_DIR,
    ) -> dict:
        return self.import_document_bytes_and_rebuild(
            pdf_name,
            pdf_bytes,
            standards_dir=standards_dir,
        )

    def import_document_bytes_and_rebuild(
        self,
        document_name: str,
        document_bytes: bytes,
        *,
        standards_dir: Path = STANDARDS_DIR,
    ) -> dict:
        stored_document_path = store_uploaded_document(
            document_name,
            document_bytes,
            standards_dir=standards_dir,
        )
        markdown_path = prepare_markdown_reference(stored_document_path)
        rebuild_stats = self.rebuild_vectorstore(standards_dir)
        return {
            "stored_document_path": str(stored_document_path),
            "markdown_path": str(markdown_path),
            **rebuild_stats,
        }

    def delete_document_and_rebuild(
        self,
        document_name_or_path: str | Path,
        *,
        standards_dir: Path = STANDARDS_DIR,
    ) -> dict:
        delete_stats = delete_reference_document(
            document_name_or_path,
            standards_dir=standards_dir,
        )
        rebuild_stats = self.rebuild_vectorstore(standards_dir)
        return {
            **delete_stats,
            **rebuild_stats,
        }
