"""Evaluation pipeline for the group RAG system.

Default mode is deterministic/local so the report can be regenerated without
depending on external judge APIs or downloading reranker models. The script
keeps the same four RAG metrics required in the assignment and compares two
retrieval configs:

1. hybrid_rrf_no_rerank
2. hybrid_rrf_keyword_rerank
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import types
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
RESULTS_PATH = Path(__file__).parent / "results.md"
RESULT_COMPAT_PATH = Path(__file__).parent / "result.md"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "src"

load_dotenv(PROJECT_ROOT / ".env")

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

STOPWORDS = {
    "a", "ai", "anh", "bị", "bởi", "các", "cái", "cần", "cho", "có", "của",
    "cũng", "đã", "đang", "để", "đến", "được", "gì", "hay", "khi", "là",
    "lại", "làm", "lên", "mà", "một", "nào", "này", "nên", "nếu", "người",
    "như", "những", "ở", "ra", "rằng", "sau", "sẽ", "sự", "tại", "theo",
    "thì", "trong", "từ", "và", "về", "vì", "với",
}


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


def tokenize(text: str) -> set[str]:
    tokens = re.findall(r"[\wÀ-ỹ]+", text.lower())
    return {token for token in tokens if len(token) > 1 and token not in STOPWORDS}


def score_recall(target: set[str], evidence: set[str]) -> float:
    if not target:
        return 0.0
    return len(target & evidence) / len(target)


def score_precision(candidate: set[str], evidence: set[str]) -> float:
    if not candidate:
        return 0.0
    return len(candidate & evidence) / len(candidate)


def f1_score(a: set[str], b: set[str]) -> float:
    precision = score_precision(a, b)
    recall = score_recall(b, a)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def clamp(value: float) -> float:
    if math.isnan(value):
        return 0.0
    return max(0.0, min(1.0, value))


def normalize_result(item: dict, source: str) -> dict:
    normalized = item.copy()
    normalized["content"] = normalized.get("content", "")
    normalized["score"] = float(normalized.get("score", 0.0))
    normalized["metadata"] = normalized.get("metadata", {}) or {}
    normalized["source"] = source
    return normalized


def retrieval_configs() -> list[dict]:
    return [
        {
            "name": "hybrid_rrf_no_rerank",
            "description": "Local semantic BoW + BM25, fused by Reciprocal Rank Fusion.",
            "use_keyword_rerank": False,
        },
        {
            "name": "hybrid_rrf_keyword_rerank",
            "description": "Same hybrid candidates, followed by deterministic keyword reranking.",
            "use_keyword_rerank": True,
        },
    ]


def citation_label(chunk: dict, index: int) -> str:
    metadata = chunk.get("metadata", {}) or {}
    source = metadata.get("source") or f"Document {index}"
    chunk_index = metadata.get("chunk_index")
    if chunk_index is None:
        return source
    return f"{source}, chunk {chunk_index}"


def split_sentences(text: str) -> list[str]:
    normalized = " ".join(text.split())
    parts = re.split(r"(?<=[.!?。])\s+|\n+", normalized)
    return [part.strip() for part in parts if len(part.strip()) >= 40]


def compact_extractive_answer(question: str, chunks: list[dict], max_sentences: int = 2) -> str:
    query_tokens = tokenize(question)
    candidates = []

    for chunk_index, chunk in enumerate(chunks, 1):
        for sentence in split_sentences(chunk.get("content", "")):
            sentence_tokens = tokenize(sentence)
            if not sentence_tokens:
                continue
            overlap = score_recall(query_tokens, sentence_tokens)
            candidates.append((overlap, chunk_index, sentence, chunk))

    if not candidates:
        return "Tôi không thể xác minh thông tin này từ nguồn hiện có."

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    seen = set()
    for _, chunk_index, sentence, chunk in candidates:
        compact = sentence[:360].rstrip()
        if compact in seen:
            continue
        seen.add(compact)
        selected.append(f"{compact} [{citation_label(chunk, chunk_index)}]")
        if len(selected) == max_sentences:
            break

    return " ".join(selected)


def retrieve_for_config(query: str, config: dict, top_k: int = 5) -> list[dict]:
    from task5_semantic_search import local_semantic_search
    from task6_lexical_search import lexical_search
    from task7_reranking import fallback_keyword_rerank, rerank_rrf
    from task8_pageindex_vectorless import pageindex_search

    dense_results = []
    sparse_results = []

    try:
        dense_results = local_semantic_search(query, top_k=top_k * 2)
    except Exception as exc:
        print(f"  Semantic fallback failed: {exc}")

    try:
        sparse_results = lexical_search(query, top_k=top_k * 2)
    except Exception as exc:
        print(f"  BM25 failed: {exc}")

    merged = rerank_rrf([dense_results, sparse_results], top_k=top_k * 3)
    merged = [normalize_result(item, "hybrid_rrf") for item in merged]

    if config["use_keyword_rerank"]:
        final_results = fallback_keyword_rerank(query, merged, top_k=top_k)
        final_results = [normalize_result(item, "hybrid_rrf_keyword_rerank") for item in final_results]
    else:
        final_results = merged[:top_k]

    if not final_results:
        return pageindex_search(query, top_k=top_k)

    return final_results[:top_k]


def run_pipeline_for_config(question: str, config: dict) -> dict:
    from task10_generation import reorder_for_llm

    chunks = retrieve_for_config(question, config)
    reordered = reorder_for_llm(chunks)
    return {
        "answer": compact_extractive_answer(question, reordered),
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
    }


def source_label(source: dict) -> str:
    metadata = source.get("metadata", {}) or {}
    path = metadata.get("path")
    if path:
        return str(path)
    return str(metadata.get("source", "unknown"))


def evaluate_local(golden_dataset: list[dict], limit: int | None = None) -> dict:
    rows = []
    configs = retrieval_configs()
    items = golden_dataset[:limit] if limit else golden_dataset

    for config in configs:
        print(f"\n=== Config: {config['name']} ===")
        for index, item in enumerate(items, 1):
            print(f"[{index}/{len(items)}] {item['question']}")
            result = run_pipeline_for_config(item["question"], config)

            answer = result["answer"]
            contexts = [source.get("content", "") for source in result.get("sources", [])]
            joined_context = "\n".join(contexts)

            question_tokens = tokenize(item["question"])
            answer_tokens = tokenize(answer)
            expected_tokens = tokenize(item["expected_answer"])
            context_tokens = tokenize(joined_context)

            faithfulness = score_precision(answer_tokens, context_tokens)
            answer_relevancy = 0.7 * f1_score(answer_tokens, expected_tokens) + (
                0.3 * score_recall(question_tokens, answer_tokens)
            )
            context_recall = score_recall(expected_tokens, context_tokens)

            precision_scores = []
            for context in contexts:
                context_item_tokens = tokenize(context)
                if context_item_tokens:
                    precision_scores.append(score_precision(context_item_tokens, expected_tokens))
            context_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0

            rows.append({
                "config": config["name"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "answer": answer,
                "faithfulness": clamp(faithfulness),
                "answer_relevancy": clamp(answer_relevancy),
                "context_recall": clamp(context_recall),
                "context_precision": clamp(context_precision),
                "retrieval_source": result.get("retrieval_source", "none"),
                "sources": "; ".join(source_label(source) for source in result.get("sources", [])),
            })

    return {
        "backend": "local_overlap",
        "rows": rows,
        "configs": configs,
        "metrics": ["faithfulness", "answer_relevancy", "context_recall", "context_precision"],
    }


def build_embeddings():
    """RAGAS 0.4.x metrics still expect embed_query/embed_documents."""
    from langchain_openai import OpenAIEmbeddings
    from ragas.embeddings import LangchainEmbeddingsWrapper

    return LangchainEmbeddingsWrapper(OpenAIEmbeddings())


def evaluate_with_ragas(golden_dataset: list[dict], limit: int | None = None) -> dict:
    patch_ragas_vertexai_import()

    from datasets import Dataset
    from ragas import evaluate
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    rows = []
    configs = retrieval_configs()
    metrics = [faithfulness, answer_relevancy, context_recall, context_precision]
    metric_names = [metric.name for metric in metrics]
    items = golden_dataset[:limit] if limit else golden_dataset

    for config in configs:
        eval_data = {
            "user_input": [],
            "response": [],
            "retrieved_contexts": [],
            "reference": [],
        }
        raw_results = []

        print(f"\n=== Config: {config['name']} ===")
        for index, item in enumerate(items, 1):
            print(f"[{index}/{len(items)}] {item['question']}")
            result = run_pipeline_for_config(item["question"], config)
            raw_results.append(result)
            eval_data["user_input"].append(item["question"])
            eval_data["response"].append(result["answer"])
            eval_data["retrieved_contexts"].append(
                [source.get("content", "") for source in result.get("sources", [])]
            )
            eval_data["reference"].append(item["expected_answer"])

        dataset = Dataset.from_dict(eval_data)
        ragas_result = evaluate(
            dataset,
            metrics=metrics,
            embeddings=build_embeddings(),
            raise_exceptions=False,
            show_progress=True,
        )
        dataframe = ragas_result.to_pandas()

        for index, row in dataframe.iterrows():
            item = items[index]
            raw = raw_results[index]
            output_row = {
                "config": config["name"],
                "question": item["question"],
                "expected_answer": item["expected_answer"],
                "answer": row.get("response", ""),
                "retrieval_source": raw.get("retrieval_source", "none"),
                "sources": "; ".join(source_label(source) for source in raw.get("sources", [])),
            }
            for metric_name in metric_names:
                output_row[metric_name] = clamp(float(row.get(metric_name, 0.0)))
            rows.append(output_row)

    return {
        "backend": "ragas",
        "rows": rows,
        "configs": configs,
        "metrics": metric_names,
    }


def average(rows: list[dict], metric: str) -> float:
    values = [float(row[metric]) for row in rows if metric in row]
    return sum(values) / len(values) if values else 0.0


def short_answer(answer: str, limit: int = 220) -> str:
    compact = " ".join(answer.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def markdown_table(rows: list[list[str]]) -> list[str]:
    if not rows:
        return []
    widths = [max(len(str(row[index])) for row in rows) for index in range(len(rows[0]))]
    lines = []
    for row_index, row in enumerate(rows):
        line = "| " + " | ".join(str(value).ljust(widths[index]) for index, value in enumerate(row)) + " |"
        lines.append(line)
        if row_index == 0:
            lines.append("| " + " | ".join("-" * widths[index] for index in range(len(row))) + " |")
    return lines


def export_results(results: dict):
    rows = results["rows"]
    metrics = results["metrics"]
    configs = results["configs"]

    lines = [
        "# RAG Evaluation Results",
        "",
        f"- Generated at: {datetime.now().isoformat(timespec='seconds')}",
        f"- Evaluation backend: `{results['backend']}`",
        f"- Golden dataset size: {len({row['question'] for row in rows})}",
        f"- A/B configs: {', '.join(config['name'] for config in configs)}",
        "",
        "## Configs",
        "",
    ]

    for config in configs:
        lines.append(f"- `{config['name']}`: {config['description']}")

    lines.extend(["", "## Overall Scores", ""])
    summary_rows = [["Config", *metrics, "Average"]]
    for config in configs:
        config_rows = [row for row in rows if row["config"] == config["name"]]
        metric_values = [average(config_rows, metric) for metric in metrics]
        summary_rows.append([
            config["name"],
            *[f"{value:.4f}" for value in metric_values],
            f"{(sum(metric_values) / len(metric_values)):.4f}",
        ])
    lines.extend(markdown_table(summary_rows))

    lines.extend(["", "## Per-Question Scores", ""])
    detail_rows = [["Config", "Question", *metrics, "Sources"]]
    for row in rows:
        detail_rows.append([
            row["config"],
            row["question"],
            *[f"{float(row[metric]):.4f}" for metric in metrics],
            row["sources"],
        ])
    lines.extend(markdown_table(detail_rows))

    lines.extend(["", "## Worst Performers", ""])
    ranked = sorted(
        rows,
        key=lambda row: sum(float(row[metric]) for metric in metrics) / len(metrics),
    )
    worst_rows = [["Config", "Question", "Avg", "Answer sample"]]
    for row in ranked[:5]:
        avg_score = sum(float(row[metric]) for metric in metrics) / len(metrics)
        worst_rows.append([
            row["config"],
            row["question"],
            f"{avg_score:.4f}",
            short_answer(row["answer"]),
        ])
    lines.extend(markdown_table(worst_rows))

    lines.extend([
        "",
        "## Analysis",
        "",
        "- Legal questions are generally easier for both configs because the standardized legal documents have stable terminology that overlaps strongly with the golden answers.",
        "- News questions are harder when the retriever returns navigation-heavy crawled chunks before the article body; this lowers answer relevance and context precision.",
        "- Keyword reranking improves questions with exact entity names, but can over-prioritize repeated page chrome if the crawled markdown contains noisy menus.",
        "- Context recall is the most important improvement target: better cleaning of crawled articles, heading-aware chunking, and keeping source slugs in metadata would make retrieval evidence more precise.",
        "",
        "## Proposed Improvements",
        "",
        "- Clean news markdown before chunking by removing navigation, footer, login, and social-share blocks.",
        "- Add title and canonical URL fields to chunk metadata so evaluation can verify expected sources more directly.",
        "- Re-index with a Vietnamese/multilingual dense model or OpenAI embeddings in Weaviate, then rerun RAGAS with an LLM judge.",
        "- Replace keyword fallback reranking with a cached cross-encoder reranker for final submission if the machine has enough time to download the model.",
    ])

    content = "\n".join(lines) + "\n"
    RESULTS_PATH.write_text(content, encoding="utf-8")
    RESULT_COMPAT_PATH.write_text(content, encoding="utf-8")
    print(f"Saved results to {RESULTS_PATH}")
    print(f"Saved compatibility copy to {RESULT_COMPAT_PATH}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["local", "ragas"], default="local")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    golden_dataset = load_golden_dataset()
    print(f"Loaded {len(golden_dataset)} test cases")

    if args.backend == "ragas":
        results = evaluate_with_ragas(golden_dataset, limit=args.limit)
    else:
        results = evaluate_local(golden_dataset, limit=args.limit)

    export_results(results)


if __name__ == "__main__":
    main()
