import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path

import bs4
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import tool
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"
EMBEDDING_CACHE_PATH = CACHE_DIR / "05_embedding_cache.json"
DEFAULT_WEB_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


@dataclass(frozen=True)
class QueryCase:
    query: str
    retrieval_keywords: tuple[str, ...]
    answer_keywords: tuple[str, ...]
    retrieval_min_matches: int = 2
    answer_min_matches: int = 2


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    chunk_size: int
    chunk_overlap: int
    retrieval_k: int
    embedding_model: str
    pipeline: str


EVALUATION_SET = [
    QueryCase(
        query="Which task decomposition technique asks the model to think step by step before answering?",
        retrieval_keywords=("chain of thought", "step by step", "task decomposition"),
        answer_keywords=("chain of thought", "step by step", "task decomposition"),
    ),
    QueryCase(
        query="Which planning method explores multiple reasoning paths at each step instead of following a single chain?",
        retrieval_keywords=("tree of thoughts", "multiple reasoning possibilities", "search"),
        answer_keywords=("tree of thoughts", "multiple reasoning possibilities", "search"),
    ),
    QueryCase(
        query="What type of memory stores information in an external vector store for later retrieval?",
        retrieval_keywords=("long-term memory", "vector store", "retrieval"),
        answer_keywords=("long-term memory", "vector store", "retrieval"),
    ),
    QueryCase(
        query="Which self-improvement framework turns feedback into verbal reinforcement for the next attempt?",
        retrieval_keywords=("reflexion", "verbal reinforcement", "self-reflection"),
        answer_keywords=("reflexion", "verbal reinforcement", "self-reflection"),
    ),
    QueryCase(
        query="Which tool-use pattern interleaves reasoning traces with actions?",
        retrieval_keywords=("react", "reasoning traces", "actions"),
        answer_keywords=("react", "reasoning traces", "actions"),
    ),
    QueryCase(
        query="What benchmark is proposed to evaluate tool use with a large collection of APIs?",
        retrieval_keywords=("apibank", "benchmark", "api"),
        answer_keywords=("apibank", "benchmark", "api"),
    ),
    QueryCase(
        query="Which model is fine-tuned on API documentation to improve tool-use behavior?",
        retrieval_keywords=("gorilla", "api documentation", "tool use"),
        answer_keywords=("gorilla", "api documentation", "tool use"),
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

    def build_vector_store(self, chunks: list, model: str) -> InMemoryVectorStore:
        embeddings = OpenAIEmbeddings(model=model, openai_api_key=self.api_key)
        vector_store = InMemoryVectorStore(embeddings)
        vector_store.add_documents(chunks)
        return vector_store


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run experiments for the LangChain RAG tutorial and save tables and plots."
    )
    parser.add_argument(
        "--web-url",
        default=DEFAULT_WEB_URL,
        help="Source URL used in the LangChain RAG tutorial.",
    )
    return parser.parse_args()


def load_api_key() -> str:
    load_dotenv(PROJECT_ROOT.parent / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set. Add it to /workspace/.env before running this script.")
    return api_key


def load_documents(web_url: str):
    loader = WebBaseLoader(
        web_paths=(web_url,),
        bs_kwargs={
            "parse_only": bs4.SoupStrainer(
                class_=("post-content", "post-title", "post-header")
            )
        },
    )
    return loader.load()


def split_documents(docs, chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
    )
    return splitter.split_documents(docs)


def normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-12, norms)
    return matrix / norms


def count_keyword_matches(text: str, keywords: tuple[str, ...]) -> int:
    lowered = text.lower()
    return sum(keyword.lower() in lowered for keyword in keywords)


def matches_keyword_threshold(text: str, keywords: tuple[str, ...], min_matches: int) -> bool:
    threshold = min(min_matches, len(keywords))
    return count_keyword_matches(text, keywords) >= threshold


def keyword_coverage(text: str, keywords: tuple[str, ...]) -> float:
    matched = count_keyword_matches(text, keywords)
    return matched / max(len(keywords), 1)


def evaluate_retrieval(results: list, case: QueryCase) -> dict:
    relevance_flags = [
        matches_keyword_threshold(
            doc.page_content,
            case.retrieval_keywords,
            case.retrieval_min_matches,
        )
        for doc in results
    ]

    first_relevant_rank = None
    for rank, is_relevant in enumerate(relevance_flags, start=1):
        if is_relevant:
            first_relevant_rank = rank
            break

    return {
        "retrieved_ranks": list(range(1, len(results) + 1)),
        "relevant_ranks": [rank for rank, ok in enumerate(relevance_flags, start=1) if ok],
        "top1_relevant": bool(relevance_flags and relevance_flags[0]),
        "hit_at_k": bool(any(relevance_flags)),
        "mrr": 0.0 if first_relevant_rank is None else 1.0 / first_relevant_rank,
        "relevant_count": int(sum(relevance_flags)),
        "top_result_preview": results[0].page_content[:240] if results else "",
        "top_result_source": str(results[0].metadata) if results else "",
        "top_result_keyword_matches": (
            count_keyword_matches(results[0].page_content, case.retrieval_keywords)
            if results
            else 0
        ),
    }


def build_two_step_chain(vector_store: InMemoryVectorStore, retrieval_k: int):
    model = init_chat_model("gpt-4o-mini")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's question using only the retrieved context. "
                "If the answer is not supported by the context, say you do not know.",
            ),
            (
                "human",
                "Question: {question}\n\nRetrieved context:\n{context}",
            ),
        ]
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    retriever = vector_store.as_retriever(search_kwargs={"k": retrieval_k})
    return (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | prompt
        | model
    )


def build_agentic_rag_agent(vector_store: InMemoryVectorStore, retrieval_k: int):
    @tool
    def retrieve_context(query: str) -> str:
        """Retrieve information related to a query."""
        docs = vector_store.similarity_search(query, k=retrieval_k)
        return "\n\n".join(
            f"Source: {doc.metadata}\nContent: {doc.page_content}" for doc in docs
        )

    return create_agent(
        init_chat_model("gpt-4o-mini"),
        tools=[retrieve_context],
        system_prompt=(
            "You are a helpful assistant for question-answering tasks. "
            "Use the retrieval tool when the user asks about the indexed blog content. "
            "If the answer is not supported by the retrieved context, say you do not know."
        ),
    )


def generate_answer(
    vector_store: InMemoryVectorStore,
    config: ExperimentConfig,
    question: str,
) -> str:
    if config.pipeline == "two_step":
        chain = build_two_step_chain(vector_store, retrieval_k=config.retrieval_k)
        return chain.invoke(question).content

    if config.pipeline == "agentic":
        agent = build_agentic_rag_agent(vector_store, retrieval_k=config.retrieval_k)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        return result["messages"][-1].content

    raise ValueError(f"Unsupported pipeline: {config.pipeline}")


def build_configs() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(
            name="baseline_two_step",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
            pipeline="two_step",
        ),
        ExperimentConfig(
            name="baseline_agentic",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
            pipeline="agentic",
        ),
        ExperimentConfig(
            name="chunk_size_500",
            chunk_size=500,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
            pipeline="two_step",
        ),
        ExperimentConfig(
            name="chunk_size_1500",
            chunk_size=1500,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
            pipeline="two_step",
        ),
        ExperimentConfig(
            name="chunk_overlap_50",
            chunk_size=1000,
            chunk_overlap=50,
            retrieval_k=3,
            embedding_model="text-embedding-3-small",
            pipeline="two_step",
        ),
        ExperimentConfig(
            name="retrieval_k_5",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=5,
            embedding_model="text-embedding-3-small",
            pipeline="two_step",
        ),
        ExperimentConfig(
            name="embedding_large",
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=3,
            embedding_model="text-embedding-3-large",
            pipeline="two_step",
        ),
    ]


def run_experiment(
    config: ExperimentConfig,
    docs,
    cache: EmbeddingCache,
) -> tuple[pd.DataFrame, dict]:
    chunks = split_documents(docs, config.chunk_size, config.chunk_overlap)
    vector_store = cache.build_vector_store(chunks, model=config.embedding_model)

    query_rows = []
    answer_hits = []
    answer_coverages = []

    for case in EVALUATION_SET:
        results = vector_store.similarity_search(case.query, k=config.retrieval_k)
        row = evaluate_retrieval(results, case)

        answer = generate_answer(vector_store, config, case.query)
        answer_hit = matches_keyword_threshold(
            answer,
            case.answer_keywords,
            case.answer_min_matches,
        )
        answer_coverage = keyword_coverage(answer, case.answer_keywords)
        answer_hits.append(float(answer_hit))
        answer_coverages.append(float(answer_coverage))

        row.update(
            {
                "query": case.query,
                "retrieval_keywords": ", ".join(case.retrieval_keywords),
                "answer_keywords": ", ".join(case.answer_keywords),
                "config_name": config.name,
                "pipeline": config.pipeline,
                "chunk_size": config.chunk_size,
                "chunk_overlap": config.chunk_overlap,
                "retrieval_k": config.retrieval_k,
                "embedding_model": config.embedding_model,
                "chunk_count": len(chunks),
                "avg_chunk_chars": float(np.mean([len(chunk.page_content) for chunk in chunks])),
                "answer_hit": answer_hit,
                "answer_keyword_coverage": answer_coverage,
                "answer_preview": answer[:240] if answer else "",
            }
        )
        query_rows.append(row)

    query_df = pd.DataFrame(query_rows)
    summary = {
        "config_name": config.name,
        "pipeline": config.pipeline,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_k": config.retrieval_k,
        "embedding_model": config.embedding_model,
        "chunk_count": len(chunks),
        "avg_chunk_chars": float(np.mean([len(chunk.page_content) for chunk in chunks])),
        "top1_accuracy": float(query_df["top1_relevant"].mean()),
        "hit_rate_at_k": float(query_df["hit_at_k"].mean()),
        "mrr": float(query_df["mrr"].mean()),
        "avg_relevant_count": float(query_df["relevant_count"].mean()),
        "answer_hit_rate": float(np.mean(answer_hits)),
        "avg_answer_keyword_coverage": float(np.mean(answer_coverages)),
    }
    return query_df, summary


def plot_overall_scores(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["top1_accuracy", "hit_rate_at_k", "mrr"]
    x = np.arange(len(summary_df))
    width = 0.22

    fig, ax = plt.subplots(figsize=(15, 7))
    for offset, metric in zip([-width, 0, width], metrics, strict=True):
        ax.bar(x + offset, summary_df[metric], width=width, label=metric)

    ax.set_title("RAG Tutorial Experiment Comparison")
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
    fig, ax = plt.subplots(figsize=(15, 6))
    image = ax.imshow(pivot.values, aspect="auto", cmap="YlGn", vmin=0, vmax=1)
    ax.set_title("MRR by Query and RAG Experiment")
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


def plot_answer_scores(summary_df: pd.DataFrame, output_path: Path) -> None:
    filtered = summary_df.dropna(subset=["answer_hit_rate", "avg_answer_keyword_coverage"])
    fig, ax = plt.subplots(figsize=(14, 7))
    x = np.arange(len(filtered))
    width = 0.35

    ax.bar(x - width / 2, filtered["answer_hit_rate"], width=width, label="answer_hit_rate")
    ax.bar(
        x + width / 2,
        filtered["avg_answer_keyword_coverage"],
        width=width,
        label="avg_answer_keyword_coverage",
    )

    ax.set_title("Answer-Level Scores for RAG Experiments")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(filtered["config_name"], rotation=25, ha="right")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    api_key = load_api_key()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = EmbeddingCache(api_key=api_key, cache_path=EMBEDDING_CACHE_PATH)
    docs = load_documents(args.web_url)
    configs = build_configs()

    query_frames = []
    summary_rows = []
    for config in configs:
        query_df, summary = run_experiment(
            config=config,
            docs=docs,
            cache=cache,
        )
        query_frames.append(query_df)
        summary_rows.append(summary)

    full_query_df = pd.concat(query_frames, ignore_index=True)
    summary_df = pd.DataFrame(summary_rows).sort_values(
        by=["hit_rate_at_k", "mrr", "top1_accuracy"],
        ascending=False,
    )

    summary_csv = OUTPUT_DIR / "05_rag_tutorial_experiment_summary.csv"
    query_csv = OUTPUT_DIR / "05_rag_tutorial_query_details.csv"
    overall_png = OUTPUT_DIR / "05_rag_tutorial_overall_scores.png"
    heatmap_png = OUTPUT_DIR / "05_rag_tutorial_query_heatmap.png"
    tradeoff_png = OUTPUT_DIR / "05_rag_tutorial_chunk_tradeoffs.png"
    answer_png = OUTPUT_DIR / "05_rag_tutorial_answer_scores.png"

    summary_df.to_csv(summary_csv, index=False)
    full_query_df.to_csv(query_csv, index=False)
    plot_overall_scores(summary_df, overall_png)
    plot_query_heatmap(full_query_df, heatmap_png)
    plot_chunk_tradeoffs(summary_df, tradeoff_png)
    plot_answer_scores(summary_df, answer_png)

    display_cols = [
        "config_name",
        "pipeline",
        "chunk_size",
        "chunk_overlap",
        "retrieval_k",
        "embedding_model",
        "chunk_count",
        "top1_accuracy",
        "hit_rate_at_k",
        "mrr",
        "answer_hit_rate",
        "avg_answer_keyword_coverage",
    ]

    print("\nExperiment summary")
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")
    print(f"Saved per-query table to: {query_csv}")
    print(f"Saved overall score chart to: {overall_png}")
    print(f"Saved query heatmap to: {heatmap_png}")
    print(f"Saved chunk tradeoff chart to: {tradeoff_png}")
    print(f"Saved answer score chart to: {answer_png}")


if __name__ == "__main__":
    main()
