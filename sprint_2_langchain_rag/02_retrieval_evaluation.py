import os
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
DOCUMENTS_PATH = NOTEBOOKS_DIR / "historical_figures.json"
GROUND_TRUTHS_PATH = NOTEBOOKS_DIR / "ground_truths.json"


def load_documents() -> list[dict]:
    return pd.read_json(DOCUMENTS_PATH).to_dict(orient="records")


def load_queries() -> list[dict]:
    return pd.read_json(GROUND_TRUTHS_PATH).to_dict(orient="records")


def build_document_lookup(documents: list[dict]) -> dict[str, dict]:
    return {doc["id"]: doc for doc in documents}


def build_vector_store(documents: list[dict]) -> FAISS:
    docs = [
        Document(
            page_content=doc.get("text", ""),
            metadata={"id": doc.get("id"), "name": doc.get("name")},
        )
        for doc in documents
    ]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunked_docs = text_splitter.split_documents(docs)
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])
    return FAISS.from_documents(chunked_docs, embeddings)
    # return FAISS.from_documents(docs, embeddings)


def retrieve(query: str, vector_store: FAISS, k: int = 5) -> list[str]:
    retrieved_docs = vector_store.similarity_search(query, k=max(k * 3, k))

    seen_ids: set[str] = set()
    result_ids: list[str] = []
    for doc in retrieved_docs:
        doc_id = doc.metadata.get("id")
        if not doc_id or doc_id in seen_ids:
            continue
        seen_ids.add(doc_id)
        result_ids.append(doc_id)
        if len(result_ids) == k:
            break

    return result_ids


def compute_recall_at_k(
    retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5
) -> float:
    retrieved_at_k = set(retrieved_ids[:k])
    relevant = set(ground_truth_ids)
    if not relevant:
        return 0.0
    return len(retrieved_at_k & relevant) / len(relevant)


def compute_precision_at_k(
    retrieved_ids: list[str], ground_truth_ids: list[str], k: int = 5
) -> float:
    retrieved_at_k = set(retrieved_ids[:k])
    relevant = set(ground_truth_ids)
    return len(retrieved_at_k & relevant) / k


def get_metric_for_query(ground_truth_count: int, k: int = 5) -> str:
    return f"recall@{k}" if ground_truth_count <= k else f"precision@{k}"


def get_incorrect_examples(
    retrieved_ids: list[str],
    ground_truth_ids: list[str],
    document_lookup: dict[str, dict],
    max_examples: int = 3,
) -> list[str]:
    ground_truth_set = set(ground_truth_ids)
    incorrect = []
    for doc_id in retrieved_ids:
        if doc_id in ground_truth_set:
            continue
        doc = document_lookup.get(doc_id, {})
        name = doc.get("name", "Unknown")
        incorrect.append(f"{name} ({doc_id})")
    return incorrect[:max_examples]


def evaluate_retrieval(
    retrieval_function: Callable[[str], list[str]],
    queries: list[dict],
    document_lookup: dict[str, dict],
    k: int = 5,
) -> pd.DataFrame:
    results: list[dict] = []
    for query_row in queries:
        retrieved = retrieval_function(query_row["query"])
        ground_truth = query_row["ground_truth"]
        gt_count = len(ground_truth)
        metric_name = get_metric_for_query(gt_count, k=k)

        if gt_count <= k:
            score = compute_recall_at_k(retrieved, ground_truth, k=k)
        else:
            score = compute_precision_at_k(retrieved, ground_truth, k=k)

        results.append(
            {
                "difficulty": query_row["difficulty"],
                "query": query_row["query"],
                "metric": metric_name,
                "score": score,
                "total_relevant_docs": gt_count,
                "retrieved_ids": retrieved,
                "incorrect_retrieved_examples": ", ".join(
                    get_incorrect_examples(
                        retrieved,
                        ground_truth,
                        document_lookup=document_lookup,
                        max_examples=3,
                    )
                ),
                "notes": query_row["notes"],
            }
        )

    return pd.DataFrame(results)


def display_results(results_df: pd.DataFrame) -> pd.DataFrame:
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    for difficulty in ["easy", "medium", "hard"]:
        subset = results_df[results_df["difficulty"] == difficulty]
        if subset.empty:
            continue
        print(f"\n  {difficulty.upper()} QUERIES (n={len(subset)}):")
        print(f"    Mean Score: {subset['score'].mean():.2f}")
        print(f"    Min: {subset['score'].min():.2f}, Max: {subset['score'].max():.2f}")

    print("\n" + "=" * 70)
    print("DETAILED RESULTS")
    print("=" * 70)

    return results_df[
        [
            "difficulty",
            "query",
            "metric",
            "score",
            "total_relevant_docs",
            "incorrect_retrieved_examples",
        ]
    ].sort_values(by=["metric", "difficulty", "score"], ascending=[True, True, False])


def main() -> None:
    load_dotenv(PROJECT_ROOT.parent / ".env")

    if "OPENAI_API_KEY" not in os.environ:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    documents = load_documents()
    queries = load_queries()
    document_lookup = build_document_lookup(documents)
    vector_store = build_vector_store(documents)
    retrieval_fn = lambda query: retrieve(query, vector_store=vector_store, k=5)

    results_df = evaluate_retrieval(
        retrieval_fn, queries, document_lookup=document_lookup, k=5
    )
    detailed_results = display_results(results_df)
    print(detailed_results.to_string(index=False))


if __name__ == "__main__":
    main()
