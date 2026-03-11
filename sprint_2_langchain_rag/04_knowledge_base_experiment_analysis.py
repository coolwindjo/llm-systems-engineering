import argparse
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"
EMBEDDING_CACHE_PATH = CACHE_DIR / "04_embedding_cache.json"

DEFAULT_PDF = NOTEBOOKS_DIR / "attention_is_all_you_need.pdf"


@dataclass(frozen=True)
class QueryCase:
    query: str
    expected_keywords: tuple[str, ...]


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    pdf_path: Path
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    embedding_model: str


EVALUATION_SET = [
    QueryCase(
        query="What replaces recurrence and convolution in the Transformer?",
        expected_keywords=("self-attention", "attention mechanism", "recurrence"),
    ),
    QueryCase(
        query="Why is the Transformer good at modeling long-range dependencies?",
        expected_keywords=("long-range dependencies", "path length", "constant"),
    ),
    QueryCase(
        query="How does the model represent token order without recurrence?",
        expected_keywords=("positional encoding", "positional encodings", "sine", "cosine"),
    ),
    QueryCase(
        query="How is multi-head attention described in the paper?",
        expected_keywords=("multi-head attention", "heads", "different representation subspaces"),
    ),
    QueryCase(
        query="What does scaled dot-product attention compute?",
        expected_keywords=("scaled dot-product attention", "query", "key", "value"),
    ),
    QueryCase(
        query="What are the main encoder and decoder stack sizes in the base model?",
        expected_keywords=("encoder", "decoder", "layers", "N=6"),
    ),
]


class EmbeddingCache:
    def __init__(self, api_key: str, cache_path: Path) -> None:
        self.api_key = api_key
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            self.cache: dict[str, dict[str, list[float]]] = json.loads(cache_path.read_text())
        else:
            self.cache = {}

    def embed_texts(self, texts: list[str], model: str) -> np.ndarray:
        model_cache = self.cache.setdefault(model, {})
        missing = [text for text in texts if text not in model_cache]

        if missing:
            embedder = OpenAIEmbeddings(model=model, openai_api_key=self.api_key)
            vectors = embedder.embed_documents(missing)
            for text, vector in zip(missing, vectors, strict=True):
                model_cache[text] = vector
            self.cache_path.write_text(json.dumps(self.cache))

        return np.array([model_cache[text] for text in texts], dtype=np.float32)

    def embed_query(self, text: str, model: str) -> np.ndarray:
        return self.embed_texts([text], model=model)[0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Section 8 knowledge-base experiments and compare them with tables and visualizations."
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        default=DEFAULT_PDF,
        help="Primary PDF for the experiments.",
    )
    parser.add_argument(
        "--alt-pdf",
        type=Path,
        default=None,
        help="Optional second PDF to compare against the primary PDF.",
    )
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to /workspace/.env before running this script.")
    return api_key


def load_chunks(pdf_path: Path, chunk_size: int, chunk_overlap: int) -> list[dict]:
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()  # [Documents Load]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    chunks = splitter.split_documents(docs)  # [Split]

    records: list[dict] = []
    for index, chunk in enumerate(chunks):
        records.append(
            {
                "chunk_id": index,
                "text": chunk.page_content,
                "page": chunk.metadata.get("page"),
                "source": chunk.metadata.get("source"),
                "start_index": chunk.metadata.get("start_index"),
            }
        )
    return records


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return matrix / norms


def retrieve(
    chunks: list[dict],
    chunk_embeddings: np.ndarray,
    query: str,
    retrieval_k: int,
    cache: EmbeddingCache,
    model: str,
) -> list[dict]:
    query_embedding = cache.embed_query(query, model=model).astype(np.float32)  # [Embedding Query]
    query_embedding = query_embedding / max(np.linalg.norm(query_embedding), 1e-12)
    scores = chunk_embeddings @ query_embedding  # [Retrieval - Similarity Search]
    ranked_indices = np.argsort(scores)[::-1][:retrieval_k]

    results = []
    for rank, index in enumerate(ranked_indices, start=1):
        row = dict(chunks[int(index)])
        row["rank"] = rank
        row["score"] = float(scores[int(index)])
        results.append(row)
    return results


def contains_expected_keyword(text: str, keywords: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(keyword.lower() in lowered for keyword in keywords)


def evaluate_query(results: list[dict], case: QueryCase) -> dict:
    relevance_flags = [
        contains_expected_keyword(result["text"], case.expected_keywords) for result in results
    ]

    first_relevant_rank = None
    for result, is_relevant in zip(results, relevance_flags, strict=True):
        if is_relevant:
            first_relevant_rank = result["rank"]
            break

    return {
        "query": case.query,
        "expected_keywords": ", ".join(case.expected_keywords),
        "retrieved_ranks": [result["rank"] for result in results],
        "relevant_ranks": [result["rank"] for result, ok in zip(results, relevance_flags, strict=True) if ok],
        "top1_relevant": bool(relevance_flags and relevance_flags[0]),
        "hit_at_k": bool(any(relevance_flags)),
        "mrr": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "relevant_count": int(sum(relevance_flags)),
        "top_result_page": results[0]["page"] if results else None,
        "top_result_score": results[0]["score"] if results else None,
        "top_result_preview": (results[0]["text"][:240] if results else ""),
    }


def run_experiment(config: ExperimentConfig, cache: EmbeddingCache) -> tuple[pd.DataFrame, dict]:
    chunks = load_chunks(config.pdf_path, config.chunk_size, config.chunk_overlap)
    chunk_embeddings = cache.embed_texts([chunk["text"] for chunk in chunks], model=config.embedding_model)  # [Embedding Chunks]
    chunk_embeddings = normalize_rows(chunk_embeddings)

    query_rows = []
    for case in EVALUATION_SET:
        results = retrieve(
            chunks=chunks,
            chunk_embeddings=chunk_embeddings,
            query=case.query,
            retrieval_k=config.retrieval_k,
            cache=cache,
            model=config.embedding_model,
        )
        metrics = evaluate_query(results, case)
        metrics.update(
            {
                "config_name": config.name,
                "pdf_name": config.pdf_path.name,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "retrieval_k": config.retrieval_k,
                "embedding_model": config.embedding_model,
                "chunk_count": len(chunks),
                "avg_chunk_chars": float(np.mean([len(chunk["text"]) for chunk in chunks])),
            }
        )
        query_rows.append(metrics)

    query_df = pd.DataFrame(query_rows)
    summary = {
        "config_name": config.name,
        "pdf_name": config.pdf_path.name,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_k": config.retrieval_k,
        "embedding_model": config.embedding_model,
        "chunk_count": len(chunks),
        "avg_chunk_chars": float(np.mean([len(chunk["text"]) for chunk in chunks])),
        "top1_accuracy": float(query_df["top1_relevant"].mean()),
        "hit_rate_at_k": float(query_df["hit_at_k"].mean()),
        "mrr": float(query_df["mrr"].mean()),
        "avg_relevant_count": float(query_df["relevant_count"].mean()),
    }
    return query_df, summary


def build_experiment_configs(primary_pdf: Path, alt_pdf: Path | None) -> list[ExperimentConfig]:
    configs = [
        ExperimentConfig(
            name="baseline",
            pdf_path=primary_pdf,
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
        ),
        ExperimentConfig(
            name="chunk_size_500",
            pdf_path=primary_pdf,
            chunk_size=500,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
        ),
        ExperimentConfig(
            name="chunk_size_1500",
            pdf_path=primary_pdf,
            chunk_size=1500,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
        ),
        ExperimentConfig(
            name="chunk_overlap_50",
            pdf_path=primary_pdf,
            chunk_size=1000,
            chunk_overlap=50,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
        ),
        ExperimentConfig(
            name="retrieval_k_5",
            pdf_path=primary_pdf,
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=5,
            embedding_model="text-embedding-3-small",
        ),
        ExperimentConfig(
            name="embedding_large",
            pdf_path=primary_pdf,
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-large",
        ),
    ]

    if alt_pdf is not None:
        configs.append(
            ExperimentConfig(
                name="alternate_pdf",
                pdf_path=alt_pdf,
                chunk_size=1000,
                chunk_overlap=200,
                retrieval_k=3,
                embedding_model="text-embedding-3-small",
            )
        )

    return configs


def plot_overall_scores(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["top1_accuracy", "hit_rate_at_k", "mrr"]
    x = np.arange(len(summary_df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(14, 7))
    for offset, metric in zip([-width, 0, width], metrics, strict=True):
        ax.bar(x + offset, summary_df[metric], width=width, label=metric)

    ax.set_title("Knowledge Base Experiment Comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["config_name"], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_query_heatmap(query_df: pd.DataFrame, output_path: Path) -> None:
    pivot = query_df.pivot(index="query", columns="config_name", values="mrr")
    fig, ax = plt.subplots(figsize=(14, 6))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_title("MRR by Query and Experiment")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=25, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)

    for row in range(pivot.shape[0]):
        for col in range(pivot.shape[1]):
            value = pivot.values[row, col]
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=8)

    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.04, label="MRR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def plot_chunk_tradeoffs(summary_df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    scatter = ax.scatter(
        summary_df["chunk_count"],
        summary_df["hit_rate_at_k"],
        c=summary_df["mrr"],
        s=180,
        cmap="viridis",
        edgecolor="black",
    )

    for _, row in summary_df.iterrows():
        ax.annotate(
            row["config_name"],
            (row["chunk_count"], row["hit_rate_at_k"]),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_title("Chunk Count vs Retrieval Quality")
    ax.set_xlabel("Number of chunks")
    ax.set_ylabel("Hit Rate@k")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.colorbar(scatter, ax=ax, label="MRR")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    primary_pdf = args.pdf.resolve()
    alt_pdf = args.alt_pdf.resolve() if args.alt_pdf else None

    if not primary_pdf.exists():
        raise FileNotFoundError(f"Primary PDF not found: {primary_pdf}")
    if alt_pdf is not None and not alt_pdf.exists():
        raise FileNotFoundError(f"Alternate PDF not found: {alt_pdf}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(api_key=api_key, cache_path=EMBEDDING_CACHE_PATH)
    configs = build_experiment_configs(primary_pdf=primary_pdf, alt_pdf=alt_pdf)

    query_frames = []
    summary_rows = []
    for config in configs:
        query_df, summary = run_experiment(config, cache)
        query_frames.append(query_df)
        summary_rows.append(summary)

    full_query_df = pd.concat(query_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["hit_rate_at_k", "mrr", "top1_accuracy"],
        ascending=False,
    )

    summary_csv = OUTPUT_DIR / "04_knowledge_base_experiment_summary.csv"
    query_csv = OUTPUT_DIR / "04_knowledge_base_query_details.csv"
    overall_png = OUTPUT_DIR / "04_knowledge_base_overall_scores.png"
    heatmap_png = OUTPUT_DIR / "04_knowledge_base_query_heatmap.png"
    tradeoff_png = OUTPUT_DIR / "04_knowledge_base_chunk_tradeoffs.png"

    summary_df.to_csv(summary_csv, index=False)
    full_query_df.to_csv(query_csv, index=False)
    plot_overall_scores(summary_df, overall_png)
    plot_query_heatmap(full_query_df, heatmap_png)
    plot_chunk_tradeoffs(summary_df, tradeoff_png)

    display_cols = [
        "config_name",
        "pdf_name",
        "chunk_size",
        "chunk_overlap",
        "retrieval_k",
        "embedding_model",
        "chunk_count",
        "top1_accuracy",
        "hit_rate_at_k",
        "mrr",
    ]

    print("\nExperiment summary")
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")
    print(f"Saved per-query table to: {query_csv}")
    print(f"Saved overall score chart to: {overall_png}")
    print(f"Saved query heatmap to: {heatmap_png}")
    print(f"Saved chunk tradeoff chart to: {tradeoff_png}")


if __name__ == "__main__":
    main()
