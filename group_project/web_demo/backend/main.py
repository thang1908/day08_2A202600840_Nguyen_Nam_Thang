from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
EVALUATION_DIR = PROJECT_ROOT / "group_project" / "evaluation"
FRONTEND_DIST = PROJECT_ROOT / "group_project" / "web_demo" / "frontend" / "dist"
STANDARDIZED_DIR = PROJECT_ROOT / "data" / "standardized"

for path in (SRC_DIR, EVALUATION_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from task10_generation import (  # noqa: E402
    SYSTEM_PROMPT,
    extractive_answer,
    format_context,
    reorder_for_llm,
)
from task9_retrieval_pipeline import retrieve  # noqa: E402
from eval_pipeline import (  # noqa: E402
    evaluate_with_ragas,
    export_results,
    load_golden_dataset,
)


@dataclass
class DemoConfig:
    generation_model: str = os.getenv("OPENAI_GENERATION_MODEL", "gpt-4o-mini")
    top_k: int = 5
    score_threshold: float = 0.3
    use_reranking: bool = True
    temperature: float = 0.3
    top_p: float = 0.9


class RagRequest(BaseModel):
    question: str = Field(..., min_length=1)
    generation_model: str = "gpt-4o-mini"
    top_k: int = Field(5, ge=1, le=12)
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)
    use_reranking: bool = True
    temperature: float = Field(0.3, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.1, le=1.0)


class EvalRequest(BaseModel):
    generation_model: str = "gpt-4o-mini"
    top_k: int = Field(5, ge=1, le=12)
    score_threshold: float = Field(0.3, ge=0.0, le=1.0)
    use_reranking: bool = True
    temperature: float = Field(0.3, ge=0.0, le=1.5)
    top_p: float = Field(0.9, ge=0.1, le=1.0)
    limit: int = Field(5, ge=1, le=20)


app = FastAPI(title="DrugLaw RAG Demo", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _demo_config_from_request(request: RagRequest | EvalRequest) -> DemoConfig:
    return DemoConfig(
        generation_model=request.generation_model,
        top_k=request.top_k,
        score_threshold=request.score_threshold,
        use_reranking=request.use_reranking,
        temperature=request.temperature,
        top_p=request.top_p,
    )


def _serialize_source(source: dict[str, Any], index: int) -> dict[str, Any]:
    metadata = source.get("metadata", {}) or {}
    content = " ".join(str(source.get("content", "")).split())
    return {
        "rank": index,
        "content": content,
        "preview": content[:700],
        "score": _json_number(source.get("score")),
        "original_score": _json_number(source.get("original_score")),
        "source": source.get("source", "unknown"),
        "reranker": source.get("reranker", ""),
        "metadata": metadata,
    }


def _json_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _safe_record(record: dict[str, Any]) -> dict[str, Any]:
    safe: dict[str, Any] = {}
    for key, value in record.items():
        if isinstance(value, float):
            safe[key] = None if math.isnan(value) or math.isinf(value) else value
        else:
            safe[key] = value
    return safe


def generate_configured_answer(question: str, config: DemoConfig) -> dict[str, Any]:
    chunks = retrieve(
        question,
        top_k=config.top_k,
        score_threshold=config.score_threshold,
        use_reranking=config.use_reranking,
    )
    reordered = reorder_for_llm(chunks)
    context = format_context(reordered)
    user_message = f"Context:\n{context}\n\n---\n\nQuestion: {question}"

    try:
        if not os.getenv("OPENAI_API_KEY"):
            raise ValueError("Thiếu OPENAI_API_KEY trong .env")

        from openai import OpenAI

        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        response = client.chat.completions.create(
            model=config.generation_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=config.temperature,
            top_p=config.top_p,
        )
        answer = response.choices[0].message.content or ""
        generation_mode = "llm"
        error = None
    except Exception as exc:
        answer = extractive_answer(question, reordered)
        generation_mode = "extractive_fallback"
        error = str(exc)

    return {
        "answer": answer,
        "sources": chunks,
        "retrieval_source": chunks[0].get("source", "none") if chunks else "none",
        "generation_mode": generation_mode,
        "error": error,
    }


def _run_ragas_eval(config: DemoConfig, limit: int) -> dict[str, Any]:
    golden_dataset = load_golden_dataset()[:limit]

    def pipeline(question: str) -> dict[str, Any]:
        return generate_configured_answer(question, config)

    results = evaluate_with_ragas(pipeline, golden_dataset)
    export_results(results)

    dataframe = results["dataframe"]
    metrics = results["metrics"]
    averages = {
        metric: _json_number(dataframe[metric].mean())
        for metric in metrics
        if metric in dataframe.columns
    }
    records = [_safe_record(record) for record in dataframe.to_dict(orient="records")]

    return {
        "metrics": metrics,
        "averages": averages,
        "records": records,
        "results_path": str(EVALUATION_DIR / "results.md"),
        "config": asdict(config),
        "total_cases": len(records),
    }


@app.get("/api/config")
def get_config():
    default_config = DemoConfig()
    models = [
        default_config.generation_model,
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-4.1",
    ]
    deduped_models = list(dict.fromkeys(models))
    return {
        "defaults": asdict(default_config),
        "models": deduped_models,
    }


@app.get("/api/health")
def health():
    markdown_files = list(STANDARDIZED_DIR.rglob("*.md")) if STANDARDIZED_DIR.exists() else []
    nonempty_markdown_files = [
        path for path in markdown_files if path.is_file() and path.stat().st_size > 0
    ]
    return {
        "ok": True,
        "project_root": str(PROJECT_ROOT),
        "openai_api_key": bool(os.getenv("OPENAI_API_KEY")),
        "standardized_docs": len(nonempty_markdown_files),
        "golden_cases": len(load_golden_dataset()),
        "frontend_dist": FRONTEND_DIST.exists(),
    }


@app.get("/api/golden")
def golden_cases():
    return {"items": load_golden_dataset()}


@app.post("/api/chat")
async def chat(request: RagRequest):
    config = _demo_config_from_request(request)
    try:
        result = await run_in_threadpool(generate_configured_answer, request.question, config)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": [
            _serialize_source(source, index)
            for index, source in enumerate(result["sources"], 1)
        ],
        "retrieval_source": result["retrieval_source"],
        "generation_mode": result["generation_mode"],
        "error": result["error"],
        "config": asdict(config),
    }


@app.post("/api/evaluate")
async def evaluate(request: EvalRequest):
    config = _demo_config_from_request(request)
    try:
        return await run_in_threadpool(_run_ragas_eval, config, request.limit)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets")


@app.get("/{full_path:path}")
def serve_frontend(full_path: str):
    index_path = FRONTEND_DIST / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "message": "Frontend chưa build. Chạy `npm install && npm run dev` trong frontend hoặc build dist."
    }
