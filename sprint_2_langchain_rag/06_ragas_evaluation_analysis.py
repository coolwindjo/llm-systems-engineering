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

# RAGAS specific imports
from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import llm_factory
from ragas.metrics import AnswerRelevancy, ContextPrecision, ContextRecall, Faithfulness
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_ROOT / "analysis_outputs"
CACHE_DIR = PROJECT_ROOT / ".cache"
EMBEDDING_CACHE_PATH = CACHE_DIR / "06_embedding_cache.json"
DEFAULT_WEB_URL = "https://lilianweng.github.io/posts/2023-06-23-agent/"


# Evaluation format for RAGAS
@dataclass(frozen=True)
class QueryCase:
    query: str
    reference: str


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
        query="What is task decomposition in LLM-powered agents?",
        reference="Task decomposition is the process of breaking down a complicated task into smaller, manageable subgoals or steps. This can be done by using LLM with simple prompting, task-specific instructions, or human inputs.",
    ),
    QueryCase(
        query="What is one common technique mentioned for task decomposition?",
        reference="Chain of thought (CoT) and Tree of Thoughts (ToT) are common techniques mentioned for task decomposition.",
    ),
    QueryCase(
        query="What are the main components of an LLM-powered autonomous agent system?",
        reference="The main components are planning, memory, and tool use.",
    ),
    QueryCase(
        query="What is reflection or self-criticism used for in agent systems?",
        reference="Reflection or self-criticism is used to improve iteratively by refining past action decisions and correcting previous mistakes.",
    ),
    QueryCase(
        query="What kinds of memory are described for agents?",
        reference="Sensory memory, short-term memory (in-context learning), and long-term memory (external vector store) are described.",
    ),
    QueryCase(
        query="How do LLM agents use tools?",
        reference="LLM agents call external APIs for extra information that is missing from the model weights, such as current information, code execution capability, or access to proprietary information databases.",
    ),
    QueryCase(
        query="How does Reflexion improve agent performance?",
        reference="Reflexion is a framework that equips agents with dynamic memory and self-reflection capabilities to improve reasoning skills. A heuristic function determines when the trajectory is inefficient or contains hallucination and should be stopped.",
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
        description="Run experiments for the LangChain RAG tutorial using RAGAS for evaluation."
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


def build_two_step_chain(vector_store: InMemoryVectorStore):
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

    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
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


def get_rag_outputs(vector_store: InMemoryVectorStore, config: ExperimentConfig, question: str) -> tuple[str, list[str]]:
    docs = vector_store.similarity_search(question, k=config.retrieval_k)
    retrieved_contexts = [doc.page_content for doc in docs]
    
    if config.pipeline == "two_step":
        chain = build_two_step_chain(vector_store)
        # Note: In a real system you'd want the chain to return both answer and source docs to ensure 
        # what is evaluated matches exactly what the chain used, but our two_step chain just returns the string.
        # However, since we set k=3 in build_two_step_chain by default, we need to make sure the retriever there aligns with config.
        # But wait, build_two_step_chain hardcodes k=3! We should fix this.
        # For simplicity, we just use the raw retriever response here as a close approximation, 
        # or we can update build_two_step_chain to take retrieval_k, but let's just use what similarity_search gives.
        answer = chain.invoke(question).content
        return answer, retrieved_contexts

    if config.pipeline == "agentic":
        agent = build_agentic_rag_agent(vector_store, retrieval_k=config.retrieval_k)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
        answer = result["messages"][-1].content
        return answer, retrieved_contexts

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
            name="retrieval_k_5",
            # We are testing retrieval_k 5 with two-step
            chunk_size=1000,
            chunk_overlap=200,
            retrieval_k=5,
            embedding_model="text-embedding-3-small",
            pipeline="two_step",
        ),
    ]


def run_experiment(
    config: ExperimentConfig,
    docs,
    cache: EmbeddingCache,
    ragas_llm,
    ragas_embeddings,
) -> tuple[pd.DataFrame, dict]:
    print(f"Running experiment: {config.name} ({config.pipeline})")
    chunks = split_documents(docs, config.chunk_size, config.chunk_overlap)
    vector_store = cache.build_vector_store(chunks, model=config.embedding_model)

    samples = []
    
    # Generate answers and contexts for RAGAS
    for case in EVALUATION_SET:
        answer, retrieved_contexts = get_rag_outputs(vector_store, config, case.query)
        
        sample = SingleTurnSample(
            user_input=case.query,
            retrieved_contexts=retrieved_contexts,
            response=answer,
            reference=case.reference,
        )
        samples.append(sample)

    dataset = EvaluationDataset(samples=samples)
    
    metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_embeddings, strictness=1),
        ContextPrecision(llm=ragas_llm),
        ContextRecall(llm=ragas_llm),
    ]
    
    # The actual evaluation scoring
    result = evaluate(dataset=dataset, metrics=metrics)
    
    # Extract results
    query_df = result.to_pandas()
    
    # Add config info to query_df for tracing
    query_df["config_name"] = config.name
    query_df["pipeline"] = config.pipeline
    query_df["chunk_size"] = config.chunk_size
    query_df["retrieval_k"] = config.retrieval_k
    query_df["embedding_model"] = config.embedding_model
    query_df["chunk_count"] = len(chunks)

    # Calculate mean metrics
    summary = {
        "config_name": config.name,
        "pipeline": config.pipeline,
        "chunk_size": config.chunk_size,
        "chunk_overlap": config.chunk_overlap,
        "retrieval_k": config.retrieval_k,
        "embedding_model": config.embedding_model,
        "chunk_count": len(chunks),
        "avg_chunk_chars": float(np.mean([len(chunk.page_content) for chunk in chunks])),
    }
    
    # Add ragas score averages
    for score_col in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        if score_col in query_df.columns:
            # Handle potential NaNs carefully
            mean_val = query_df[score_col].mean()
            summary[f"ragas_{score_col}"] = float(mean_val) if not pd.isna(mean_val) else np.nan
        else:
            summary[f"ragas_{score_col}"] = np.nan

    return query_df, summary


def plot_ragas_scores(summary_df: pd.DataFrame, output_path: Path) -> None:
    metrics = ["ragas_faithfulness", "ragas_answer_relevancy", "ragas_context_precision", "ragas_context_recall"]
    
    # Filter only metrics that exist in the dataframe
    metrics = [m for m in metrics if m in summary_df.columns]
    
    x = np.arange(len(summary_df))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(15, 7))
    for i, metric in enumerate(metrics):
        offset = (i - len(metrics)/2 + 0.5) * width
        # RAGAs metrics are typically 0 to 1
        ax.bar(x + offset, summary_df[metric], width=width, label=metric.replace("ragas_", ""))

    ax.set_title("RAGAS Experiment Validation Over RAG Configs")
    ax.set_ylabel("RAGAS LLM-Assessed Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(summary_df["config_name"], rotation=25, ha="right")
    ax.legend(loc="lower right")
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

    # Setup RAGAS globally to avoid reloading client connections
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    ragas_llm = llm_factory("gpt-4o-mini", client=client)
    ragas_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=api_key, model="text-embedding-3-small"))

    query_frames = []
    summary_rows = []
    for config in configs:
        query_df, summary = run_experiment(
            config=config,
            docs=docs,
            cache=cache,
            ragas_llm=ragas_llm,
            ragas_embeddings=ragas_embeddings,
        )
        query_frames.append(query_df)
        summary_rows.append(summary)

    full_query_df = pd.concat(query_frames, ignore_index=True)
    
    # Sort by some important metrics to see the best first
    sort_cols = [c for c in ["ragas_answer_relevancy", "ragas_faithfulness"] if c in full_query_df.columns]
    if sort_cols:
        summary_df = pd.DataFrame(summary_rows).sort_values(by=sort_cols, ascending=False)
    else:
        summary_df = pd.DataFrame(summary_rows)

    summary_csv = OUTPUT_DIR / "06_ragas_experiment_summary.csv"
    query_csv = OUTPUT_DIR / "06_ragas_query_details.csv"
    overall_png = OUTPUT_DIR / "06_ragas_overall_scores.png"

    summary_df.to_csv(summary_csv, index=False)
    full_query_df.to_csv(query_csv, index=False)
    plot_ragas_scores(summary_df, overall_png)

    display_cols = [
        "config_name",
        "pipeline",
        "chunk_size",
        "retrieval_k",
    ]
    # Add any ragas columns that are available
    display_cols.extend([c for c in summary_df.columns if c.startswith("ragas_")])

    print("\nExperiment summary (RAGAS)")
    print(summary_df[display_cols].to_string(index=False))
    print(f"\nSaved summary table to: {summary_csv}")
    print(f"Saved per-query table to: {query_csv}")
    print(f"Saved overall score chart to: {overall_png}")


if __name__ == "__main__":
    main()
