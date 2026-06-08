"""RAGAS evaluation for the group RAG pipeline."""

import json
import os
import sys
import types
from pathlib import Path

from dotenv import load_dotenv

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

load_dotenv(PROJECT_ROOT / ".env")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def patch_ragas_vertexai_import():
    """Work around an optional VertexAI import issue in some RAGAS installs."""
    module_name = "langchain_community.chat_models.vertexai"
    if module_name in sys.modules:
        return

    module = types.ModuleType(module_name)
    module.ChatVertexAI = object
    sys.modules[module_name] = module


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(rag_pipeline, question: str) -> dict:
    if hasattr(rag_pipeline, "generate_with_citation"):
        return rag_pipeline.generate_with_citation(question)
    return rag_pipeline(question)


def build_embeddings():
    """RAGAS 0.4.x metrics still expect embed_query/embed_documents."""
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    return LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=model))


def evaluate_with_ragas(rag_pipeline, golden_dataset: list[dict], limit: int | None = None):
    patch_ragas_vertexai_import()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
    )

    eval_data = {
        "user_input": [],
        "response": [],
        "retrieved_contexts": [],
        "reference": [],
    }

    items = golden_dataset[:limit] if limit else golden_dataset
    for index, item in enumerate(items, 1):
        print(f"[{index}/{len(items)}] {item['question']}")
        result = run_pipeline(rag_pipeline, item["question"])

        eval_data["user_input"].append(item["question"])
        eval_data["response"].append(result["answer"])
        eval_data["retrieved_contexts"].append(
            [source.get("content", "") for source in result.get("sources", [])]
        )
        eval_data["reference"].append(item["expected_answer"])

    dataset = Dataset.from_dict(eval_data)
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

    result = evaluate(
        dataset,
        metrics=metrics,
        embeddings=build_embeddings(),
        raise_exceptions=False,
        show_progress=True,
    )
    dataframe = result.to_pandas()

    return {
        "result": result,
        "dataframe": dataframe,
        "dataset": dataset,
        "metrics": [metric.name for metric in metrics],
    }


def export_results(results: dict):
    dataframe = results["dataframe"]
    metrics = results["metrics"]

    lines = [
        "# RAG Evaluation Results",
        "",
        "## Overall Scores",
        "",
        "| Metric | Average Score |",
        "|--------|---------------|",
    ]

    for metric in metrics:
        if metric in dataframe.columns:
            lines.append(f"| {metric} | {dataframe[metric].mean():.4f} |")

    lines.extend(["", "## Per-Question Scores", ""])

    columns = [column for column in ["user_input", *metrics] if column in dataframe.columns]
    lines.append(dataframe[columns].to_markdown(index=False))

    RESULTS_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(dataframe)
    print(f"Saved results to {RESULTS_PATH}")


if __name__ == "__main__":
    from task10_generation import generate_with_citation

    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    results = evaluate_with_ragas(generate_with_citation, golden_dataset)
    export_results(results)
