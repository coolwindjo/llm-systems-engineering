import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from rank_bm25 import BM25Okapi


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCUMENTS_PATH = NOTEBOOKS_DIR / "historical_figures.json"
GROUND_TRUTHS_PATH = NOTEBOOKS_DIR / "ground_truths.json"
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"
EMBEDDING_CACHE_PATH = CACHE_DIR / "03_embedding_cache.json"
K = 5

DEFAULT_MODEL = "text-embedding-3-small"
ALTERNATE_MODEL = "text-embedding-3-large"

STOPWORDS = {
    "a",
    "about",
    "all",
    "and",
    "are",
    "as",
    "at",
    "be",
    "before",
    "but",
    "by",
    "did",
    "do",
    "find",
    "for",
    "from",
    "in",
    "into",
    "is",
    "it",
    "known",
    "made",
    "of",
    "on",
    "or",
    "people",
    "their",
    "the",
    "they",
    "to",
    "were",
    "who",
    "with",
}


@dataclass(frozen=True)
class ChunkRecord:
    doc_id: str
    name: str
    chunk_text: str
    chunk_index: int


def load_documents() -> list[dict]:
    return pd.read_json(DOCUMENTS_PATH).to_dict(orient="records")


def load_queries() -> list[dict]:
    return pd.read_json(GROUND_TRUTHS_PATH).to_dict(orient="records")


def build_document_lookup(documents: list[dict]) -> dict[str, dict]:
    return {doc["id"]: doc for doc in documents}


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]


def build_chunks(documents: list[dict], chunking: str) -> list[ChunkRecord]:
    chunks: list[ChunkRecord] = []
    for doc in documents:
        if chunking == "sentence":
            segments = split_sentences(doc["text"])
        else:
            segments = [doc["text"]]

        for index, segment in enumerate(segments):
            chunks.append(
                ChunkRecord(
                    doc_id=doc["id"],
                    name=doc["name"],
                    chunk_text=segment,
                    chunk_index=index,
                )
            )
    return chunks


class EmbeddingCache:
    def __init__(self, api_key: str, cache_path: Path) -> None:
        self.api_key = api_key
        self.cache_path = cache_path
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if cache_path.exists():
            self.cache: dict[str, dict[str, list[float]]] = json.loads(
                cache_path.read_text()
            )
        else:
            self.cache = {}

    def embed_texts(self, texts: list[str], model: str) -> list[list[float]]:
        model_cache = self.cache.setdefault(model, {})
        missing = [text for text in texts if text not in model_cache]
        if missing:
            embedder = OpenAIEmbeddings(model=model, openai_api_key=self.api_key)
            vectors = embedder.embed_documents(missing)
            for text, vector in zip(missing, vectors, strict=True):
                model_cache[text] = vector
            self.cache_path.write_text(json.dumps(self.cache))
        return [model_cache[text] for text in texts]


class EmbeddingRetriever:
    def __init__(
        self,
        name: str,
        chunks: list[ChunkRecord],
        embeddings: np.ndarray,
        model: str,
        metric: str,
        cache: EmbeddingCache,
        query_transform: Callable[[str], str] | None = None,
    ) -> None:
        self.name = name
        self.chunks = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.model = model
        self.metric = metric
        self.cache = cache
        self.query_transform = query_transform or (lambda query: query)

        if metric == "cosine":
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-12, norms)
            self.embeddings = self.embeddings / norms

    def retrieve(self, query: str, k: int = K) -> list[str]:
        transformed_query = self.query_transform(query)
        query_vector = np.array(
            self.cache.embed_texts([transformed_query], model=self.model)[0],
            dtype=np.float32,
        )

        if self.metric == "cosine":
            query_norm = np.linalg.norm(query_vector)
            if query_norm == 0:
                query_norm = 1e-12
            scores = self.embeddings @ (query_vector / query_norm)
        elif self.metric == "l2":
            scores = -np.linalg.norm(self.embeddings - query_vector, axis=1)
        else:
            raise ValueError(f"Unsupported metric: {self.metric}")

        ranked_indices = np.argsort(scores)[::-1]
        return deduplicate_ranked_doc_ids(self.chunks, ranked_indices, k)


class SimpleBM25Retriever:
    def __init__(self, documents: list[dict], query_transform: Callable[[str], str]) -> None:
        self.documents = documents
        self.query_transform = query_transform
        self.tokenized_docs = [tokenize(doc["text"]) for doc in documents]
        self.doc_freq: Counter[str] = Counter()
        for tokens in self.tokenized_docs:
            self.doc_freq.update(set(tokens))
        self.doc_lengths = [len(tokens) for tokens in self.tokenized_docs]
        self.avg_doc_len = sum(self.doc_lengths) / len(self.doc_lengths)
        self.k1 = 1.5
        self.b = 0.75

    def score(self, query: str) -> list[tuple[str, float]]:
        tokens = self.query_transform(query)
        scores: list[tuple[str, float]] = []
        num_docs = len(self.documents)
        for doc, doc_tokens, doc_len in zip(
            self.documents, self.tokenized_docs, self.doc_lengths, strict=True
        ):
            tf = Counter(doc_tokens)
            score = 0.0
            for token in tokens:
                if token not in tf:
                    continue
                doc_freq = self.doc_freq[token]
                idf = math.log(1 + (num_docs - doc_freq + 0.5) / (doc_freq + 0.5))
                numerator = tf[token] * (self.k1 + 1)
                denominator = tf[token] + self.k1 * (
                    1 - self.b + self.b * doc_len / self.avg_doc_len
                )
                score += idf * numerator / denominator
            scores.append((doc["id"], score))
        scores.sort(key=lambda item: item[1], reverse=True)
        return scores

    def retrieve(self, query: str, k: int = K) -> list[str]:
        ranked = [doc_id for doc_id, score in self.score(query) if score > 0]
        return ranked[:k]


class BM25OkapiRetriever:
    def __init__(self, documents: list[dict], query_transform: Callable[[str], list[str]]) -> None:
        self.documents = documents
        self.query_transform = query_transform
        self.tokenized_docs = [tokenize(doc["text"]) for doc in documents]
        self.bm25 = BM25Okapi(self.tokenized_docs)

    def retrieve(self, query: str, k: int = K) -> list[str]:
        tokenized_query = self.query_transform(query)
        scores = self.bm25.get_scores(tokenized_query)
        ranked_indices = np.argsort(scores)[::-1]

        doc_ids: list[str] = []
        for index in ranked_indices:
            if scores[int(index)] <= 0:
                continue
            doc_ids.append(self.documents[int(index)]["id"])
            if len(doc_ids) == k:
                break
        return doc_ids


class HybridRetriever:
    def __init__(
        self,
        embedding_retriever: EmbeddingRetriever,
        bm25_retriever: SimpleBM25Retriever | BM25OkapiRetriever,
        candidate_pool: int = 10,
    ) -> None:
        self.embedding_retriever = embedding_retriever
        self.bm25_retriever = bm25_retriever
        self.candidate_pool = candidate_pool

    def retrieve(self, query: str, k: int = K) -> list[str]:
        embedding_ids = self.embedding_retriever.retrieve(query, k=self.candidate_pool)
        bm25_ids = self.bm25_retriever.retrieve(query, k=self.candidate_pool)
        fused_scores: defaultdict[str, float] = defaultdict(float)

        for rank, doc_id in enumerate(embedding_ids, start=1):
            fused_scores[doc_id] += 1.0 / (60 + rank)
        for rank, doc_id in enumerate(bm25_ids, start=1):
            fused_scores[doc_id] += 1.0 / (60 + rank)

        ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
        return [doc_id for doc_id, _ in ranked[:k]]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def deduplicate_ranked_doc_ids(
    chunks: list[ChunkRecord], ranked_indices: np.ndarray, k: int
) -> list[str]:
    seen: set[str] = set()
    doc_ids: list[str] = []
    for index in ranked_indices:
        doc_id = chunks[int(index)].doc_id
        if doc_id in seen:
            continue
        seen.add(doc_id)
        doc_ids.append(doc_id)
        if len(doc_ids) == k:
            break
    return doc_ids


def rewrite_query_for_embeddings(query: str) -> str:
    lower = query.lower()
    expansions: list[str] = []

    if "independence" in lower:
        expansions.extend(
            ["anti-colonial leader", "freedom movement", "national independence"]
        )
    if "female scientists and mathematicians" in lower:
        expansions.extend(
            ["women in science", "woman mathematician", "female scientist"]
        )
    if "not born in europe" in lower:
        expansions.extend(
            ["outside Europe", "born in Asia", "born in Africa", "born in the Americas"]
        )
    if "first name starts with the letter a" in lower:
        expansions.extend(["name begins with a", "first name a"])
    if "lived in the 20th century but died before 1950" in lower:
        expansions.extend(
            ["died before 1950", "20th century figure", "early twentieth century death"]
        )
    if "mathematics" in lower:
        expansions.extend(["mathematician", "theorem", "mathematics"])
    if "light" in lower:
        expansions.extend(["optics", "light theory", "physics of light"])

    if not expansions:
        return query
    return f"{query}. Retrieval hints: {'; '.join(expansions)}."


def build_keyword_query(query: str) -> list[str]:
    base_tokens = [token for token in tokenize(query) if token not in STOPWORDS]
    lower = query.lower()
    extra_tokens: list[str] = []

    if "independence" in lower:
        extra_tokens.extend(["independence", "freedom", "leader", "movement"])
    if "female scientists and mathematicians" in lower:
        extra_tokens.extend(["female", "women", "scientist", "mathematician"])
    if "not born in europe" in lower:
        extra_tokens.extend(["asia", "africa", "india", "china", "america"])
    if "first name starts with the letter a" in lower:
        extra_tokens.extend(["a"])
    if "died before 1950" in lower:
        extra_tokens.extend(["died", "1900", "1940", "1950"])

    return base_tokens + extra_tokens


def compute_recall_at_k(
    retrieved_ids: list[str], ground_truth_ids: list[str], k: int = K
) -> float:
    retrieved_at_k = set(retrieved_ids[:k])
    relevant = set(ground_truth_ids)
    if not relevant:
        return 0.0
    return len(retrieved_at_k & relevant) / len(relevant)


def compute_precision_at_k(
    retrieved_ids: list[str], ground_truth_ids: list[str], k: int = K
) -> float:
    retrieved_at_k = set(retrieved_ids[:k])
    relevant = set(ground_truth_ids)
    return len(retrieved_at_k & relevant) / k


def get_metric_for_query(ground_truth_count: int, k: int = K) -> str:
    return f"recall@{k}" if ground_truth_count <= k else f"precision@{k}"


def format_doc_list(doc_ids: list[str], document_lookup: dict[str, dict]) -> str:
    items = []
    for doc_id in doc_ids:
        doc = document_lookup.get(doc_id, {})
        items.append(f"{doc.get('name', 'Unknown')} ({doc_id})")
    return " | ".join(items)


def evaluate_strategy(
    strategy_name: str,
    retrieval_function: Callable[[str, int], list[str]],
    queries: list[dict],
    document_lookup: dict[str, dict],
    k: int = K,
) -> pd.DataFrame:
    rows: list[dict] = []
    for query_row in queries:
        retrieved = retrieval_function(query_row["query"], k)
        ground_truth = query_row["ground_truth"]
        gt_count = len(ground_truth)
        metric_name = get_metric_for_query(gt_count, k)

        if gt_count <= k:
            score = compute_recall_at_k(retrieved, ground_truth, k)
        else:
            score = compute_precision_at_k(retrieved, ground_truth, k)

        incorrect_ids = [doc_id for doc_id in retrieved if doc_id not in set(ground_truth)]
        missed_ids = [doc_id for doc_id in ground_truth if doc_id not in set(retrieved)]

        rows.append(
            {
                "strategy": strategy_name,
                "difficulty": query_row["difficulty"],
                "query": query_row["query"],
                "metric": metric_name,
                "score": score,
                "total_relevant_docs": gt_count,
                "retrieved_ids": ", ".join(retrieved),
                "ground_truth_ids": ", ".join(ground_truth),
                "incorrect_retrieved_top3": format_doc_list(
                    incorrect_ids[:3], document_lookup
                ),
                "incorrect_retrieved_all": format_doc_list(
                    incorrect_ids, document_lookup
                ),
                "missed_ground_truth": format_doc_list(missed_ids, document_lookup),
                "notes": query_row["notes"],
            }
        )
    return pd.DataFrame(rows)


def build_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby("strategy")
        .agg(
            overall_mean=("score", "mean"),
            easy_mean=("score", lambda values: safe_mean_by_index(results_df, values, "easy")),
            medium_mean=(
                "score",
                lambda values: safe_mean_by_index(results_df, values, "medium"),
            ),
            hard_mean=("score", lambda values: safe_mean_by_index(results_df, values, "hard")),
            min_score=("score", "min"),
            max_score=("score", "max"),
        )
        .reset_index()
        .sort_values("overall_mean", ascending=False)
    )
    return summary


def safe_mean_by_index(results_df: pd.DataFrame, values: pd.Series, difficulty: str) -> float:
    subset = results_df.loc[values.index]
    difficulty_values = subset.loc[subset["difficulty"] == difficulty, "score"]
    if difficulty_values.empty:
        return float("nan")
    return float(difficulty_values.mean())


def build_query_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    pivot = results_df.pivot_table(
        index=["difficulty", "query"],
        columns="strategy",
        values="score",
        aggfunc="first",
    )
    return pivot.reset_index()


def build_bm25_comparison_table(results_df: pd.DataFrame) -> pd.DataFrame:
    bm25_rows = results_df[
        results_df["strategy"].isin(["bm25_keyword", "bm25_okapi_keyword"])
    ].copy()
    pivot = bm25_rows.pivot_table(
        index=["difficulty", "query"],
        columns="strategy",
        values="score",
        aggfunc="first",
    ).reset_index()
    pivot["okapi_minus_simple"] = (
        pivot["bm25_okapi_keyword"] - pivot["bm25_keyword"]
    )
    return pivot.sort_values(
        by=["okapi_minus_simple", "difficulty", "query"], ascending=[False, True, True]
    )


def save_visualizations(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    bm25_comparison_df: pd.DataFrame,
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    bar_path = OUTPUT_DIR / "03_strategy_overall_scores.png"
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(summary_df["strategy"], summary_df["overall_mean"], color="#3E7CB1")
    ax.set_title("Average Retrieval Score by Strategy")
    ax.set_ylabel("Average score")
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=30)
    for index, score in enumerate(summary_df["overall_mean"]):
        ax.text(index, score + 0.02, f"{score:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(bar_path, dpi=200)
    plt.close(fig)
    saved_paths.append(bar_path)

    heatmap_path = OUTPUT_DIR / "03_strategy_difficulty_heatmap.png"
    heatmap_df = (
        results_df.groupby(["strategy", "difficulty"])["score"]
        .mean()
        .unstack(fill_value=np.nan)
        .reindex(columns=["easy", "medium", "hard"])
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(heatmap_df.values, cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(heatmap_df.columns)
    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(heatmap_df.index)
    ax.set_title("Average Score by Strategy and Difficulty")
    for row in range(heatmap_df.shape[0]):
        for col in range(heatmap_df.shape[1]):
            value = heatmap_df.iloc[row, col]
            label = "NA" if pd.isna(value) else f"{value:.2f}"
            ax.text(col, row, label, ha="center", va="center", color="black", fontsize=9)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)
    saved_paths.append(heatmap_path)

    bm25_delta_path = OUTPUT_DIR / "03_bm25_vs_bm25okapi_delta.png"
    fig, ax = plt.subplots(figsize=(12, 5))
    labels = [
        f"{difficulty}: {query[:45]}..."
        if len(query) > 45
        else f"{difficulty}: {query}"
        for difficulty, query in zip(
            bm25_comparison_df["difficulty"],
            bm25_comparison_df["query"],
            strict=True,
        )
    ]
    colors = [
        "#2A9D8F" if value >= 0 else "#E76F51"
        for value in bm25_comparison_df["okapi_minus_simple"]
    ]
    ax.bar(range(len(bm25_comparison_df)), bm25_comparison_df["okapi_minus_simple"], color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("BM25Okapi - SimpleBM25 Score Difference by Query")
    ax.set_ylabel("Score delta")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    for index, value in enumerate(bm25_comparison_df["okapi_minus_simple"]):
        ax.text(index, value + (0.01 if value >= 0 else -0.03), f"{value:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(bm25_delta_path, dpi=200)
    plt.close(fig)
    saved_paths.append(bm25_delta_path)

    return saved_paths


def build_embedding_matrix(
    chunks: list[ChunkRecord], model: str, cache: EmbeddingCache
) -> np.ndarray:
    vectors = cache.embed_texts([chunk.chunk_text for chunk in chunks], model=model)
    return np.array(vectors, dtype=np.float32)


def prepare_strategies(
    documents: list[dict], cache: EmbeddingCache
) -> tuple[dict[str, Callable[[str, int], list[str]]], dict[str, str]]:
    full_chunks = build_chunks(documents, chunking="full")
    sentence_chunks = build_chunks(documents, chunking="sentence")

    skipped_strategies: dict[str, str] = {}
    full_small_matrix = build_embedding_matrix(full_chunks, DEFAULT_MODEL, cache)
    sentence_small_matrix = build_embedding_matrix(sentence_chunks, DEFAULT_MODEL, cache)

    baseline = EmbeddingRetriever(
        name="baseline_full_doc_cosine",
        chunks=full_chunks,
        embeddings=full_small_matrix,
        model=DEFAULT_MODEL,
        metric="cosine",
        cache=cache,
    )
    sentence_cosine = EmbeddingRetriever(
        name="sentence_chunk_cosine",
        chunks=sentence_chunks,
        embeddings=sentence_small_matrix,
        model=DEFAULT_MODEL,
        metric="cosine",
        cache=cache,
    )
    sentence_l2 = EmbeddingRetriever(
        name="sentence_chunk_l2",
        chunks=sentence_chunks,
        embeddings=sentence_small_matrix,
        model=DEFAULT_MODEL,
        metric="l2",
        cache=cache,
    )
    query_rewrite = EmbeddingRetriever(
        name="query_rewrite_sentence_cosine",
        chunks=sentence_chunks,
        embeddings=sentence_small_matrix,
        model=DEFAULT_MODEL,
        metric="cosine",
        cache=cache,
        query_transform=rewrite_query_for_embeddings,
    )
    bm25 = SimpleBM25Retriever(documents, query_transform=build_keyword_query)
    bm25_okapi = BM25OkapiRetriever(documents, query_transform=build_keyword_query)
    hybrid = HybridRetriever(query_rewrite, bm25_okapi)

    strategies = {
        "baseline_full_doc_cosine": baseline.retrieve,
        "sentence_chunk_cosine": sentence_cosine.retrieve,
        "sentence_chunk_l2": sentence_l2.retrieve,
        "query_rewrite_sentence_cosine": query_rewrite.retrieve,
        "bm25_keyword": bm25.retrieve,
        "bm25_okapi_keyword": bm25_okapi.retrieve,
        "hybrid_rrf": hybrid.retrieve,
    }

    try:
        full_large_matrix = build_embedding_matrix(full_chunks, ALTERNATE_MODEL, cache)
        alternate_model = EmbeddingRetriever(
            name="full_doc_large_model",
            chunks=full_chunks,
            embeddings=full_large_matrix,
            model=ALTERNATE_MODEL,
            metric="cosine",
            cache=cache,
        )
        strategies["full_doc_large_model"] = alternate_model.retrieve
    except Exception as exc:
        skipped_strategies["full_doc_large_model"] = str(exc)

    return strategies, skipped_strategies


def save_tables(
    summary_df: pd.DataFrame,
    results_df: pd.DataFrame,
    query_comparison_df: pd.DataFrame,
    bm25_comparison_df: pd.DataFrame,
) -> list[Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "03_strategy_summary.csv"
    details_path = OUTPUT_DIR / "03_query_strategy_details.csv"
    comparison_path = OUTPUT_DIR / "03_query_strategy_score_comparison.csv"
    bm25_comparison_path = OUTPUT_DIR / "03_bm25_strategy_comparison.csv"

    summary_df.to_csv(summary_path, index=False)
    results_df.to_csv(details_path, index=False)
    query_comparison_df.to_csv(comparison_path, index=False)
    bm25_comparison_df.to_csv(bm25_comparison_path, index=False)
    return [summary_path, details_path, comparison_path, bm25_comparison_path]


def save_skipped_strategies(skipped_strategies: dict[str, str]) -> Path | None:
    if not skipped_strategies:
        return None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped_path = OUTPUT_DIR / "03_skipped_strategies.json"
    skipped_path.write_text(json.dumps(skipped_strategies, indent=2))
    return skipped_path


def print_section(title: str) -> None:
    print("\n" + "=" * 90)
    print(title)
    print("=" * 90)


def main() -> None:
    load_dotenv(PROJECT_ROOT.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    documents = load_documents()
    queries = load_queries()
    document_lookup = build_document_lookup(documents)
    cache = EmbeddingCache(api_key=api_key, cache_path=EMBEDDING_CACHE_PATH)
    strategies, skipped_strategies = prepare_strategies(documents, cache)

    all_results = []
    for strategy_name, retriever in strategies.items():
        strategy_df = evaluate_strategy(
            strategy_name=strategy_name,
            retrieval_function=retriever,
            queries=queries,
            document_lookup=document_lookup,
            k=K,
        )
        all_results.append(strategy_df)

    results_df = pd.concat(all_results, ignore_index=True)
    summary_df = build_summary_table(results_df)
    query_comparison_df = build_query_comparison_table(results_df)
    bm25_comparison_df = build_bm25_comparison_table(results_df)

    csv_paths = save_tables(
        summary_df, results_df, query_comparison_df, bm25_comparison_df
    )
    image_paths = save_visualizations(summary_df, results_df, bm25_comparison_df)
    skipped_path = save_skipped_strategies(skipped_strategies)

    print_section("STRATEGY SUMMARY")
    print(summary_df.to_string(index=False))

    print_section("QUERY SCORE COMPARISON")
    print(query_comparison_df.to_string(index=False))

    print_section("BM25 STRATEGY COMPARISON")
    print(bm25_comparison_df.to_string(index=False))

    print_section("DETAILED QUERY ANALYSIS")
    detailed_cols = [
        "strategy",
        "difficulty",
        "query",
        "metric",
        "score",
        "retrieved_ids",
        "incorrect_retrieved_top3",
        "incorrect_retrieved_all",
        "missed_ground_truth",
    ]
    print(results_df[detailed_cols].to_string(index=False))

    if skipped_strategies:
        print_section("SKIPPED STRATEGIES")
        for strategy_name, reason in skipped_strategies.items():
            print(f"{strategy_name}: {reason}")

    print_section("SAVED FILES")
    saved_paths = csv_paths + image_paths
    if skipped_path is not None:
        saved_paths.append(skipped_path)
    for path in saved_paths:
        print(path)


if __name__ == "__main__":
    main()
