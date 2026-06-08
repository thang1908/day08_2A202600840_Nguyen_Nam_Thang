"""
Task 7 — Reranking Module.

Chọn 1 trong các phương pháp:
    - Cross-encoder reranker: Jina Reranker v2 (multilingual) hoặc Qwen3-Reranker
    - MMR (Maximal Marginal Relevance): tự implement
    - RRF (Reciprocal Rank Fusion): tự implement

Nếu dùng MMR hoặc RRF, đảm bảo hiểu và giải thích được cơ chế.
"""

import math
import os
import re

from dotenv import load_dotenv

load_dotenv()

QWEN_RERANKER_MODEL = os.getenv("QWEN_RERANKER_MODEL", "Qwen/Qwen3-Reranker-0.6B")
QWEN_RERANKER_BATCH_SIZE = int(os.getenv("QWEN_RERANKER_BATCH_SIZE", "8"))
QWEN_RERANKER_MAX_LENGTH = int(os.getenv("QWEN_RERANKER_MAX_LENGTH", "2048"))

_QWEN_RERANKER = None


def tokenize(text: str) -> set[str]:
    """Tokenize đơn giản để fallback khi local model chưa sẵn sàng."""
    return set(re.findall(r"[\wÀ-ỹ]+", text.lower()))


def get_qwen_reranker():
    """Load Qwen3 reranker một lần và cache lại."""
    global _QWEN_RERANKER

    if _QWEN_RERANKER is None:
        from sentence_transformers import CrossEncoder

        _QWEN_RERANKER = CrossEncoder(
            QWEN_RERANKER_MODEL,
            max_length=QWEN_RERANKER_MAX_LENGTH,
        )

    return _QWEN_RERANKER


def fallback_keyword_rerank(query: str, candidates: list[dict], top_k: int) -> list[dict]:
    """Fallback nhỏ gọn để test/dev chạy được khi chưa cài/tải Qwen."""
    query_tokens = tokenize(query)
    reranked = []

    for candidate in candidates:
        candidate_tokens = tokenize(candidate.get("content", ""))
        overlap = len(query_tokens & candidate_tokens)
        base_score = float(candidate.get("score", 0.0))
        score = overlap + 0.01 * base_score

        item = candidate.copy()
        item["original_score"] = base_score
        item["score"] = float(score)
        item["reranker"] = "keyword_fallback"
        reranked.append(item)

    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def rerank_cross_encoder(
    query: str, candidates: list[dict], top_k: int = 5
) -> list[dict]:
    """
    Rerank candidates sử dụng cross-encoder model.

    Args:
        query: Câu truy vấn
        candidates: List of {'content': str, 'score': float, 'metadata': dict}
        top_k: Số lượng kết quả sau rerank

    Returns:
        List of top_k candidates, re-scored và sorted by rerank_score descending.
    """
    if not candidates:
        return []

    try:
        model = get_qwen_reranker()
        pairs = [(query, candidate.get("content", "")) for candidate in candidates]
        scores = model.predict(
            pairs,
            batch_size=QWEN_RERANKER_BATCH_SIZE,
            show_progress_bar=False,
        )
    except Exception as exc:
        print(f"⚠ Không load/chạy được {QWEN_RERANKER_MODEL}, dùng fallback: {exc}")
        return fallback_keyword_rerank(query, candidates, top_k)

    reranked = []
    for candidate, score in zip(candidates, scores):
        item = candidate.copy()
        item["original_score"] = float(candidate.get("score", 0.0))
        item["score"] = float(score)
        item["reranker"] = QWEN_RERANKER_MODEL
        reranked.append(item)

    return sorted(reranked, key=lambda item: item["score"], reverse=True)[:top_k]


def cosine_sim(a: list[float], b: list[float]) -> float:
    """Cosine similarity giữa hai vector."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rerank_mmr(
    query_embedding: list[float],
    candidates: list[dict],
    top_k: int = 5,
    lambda_param: float = 0.7,
) -> list[dict]:
    """
    Maximal Marginal Relevance — chọn candidates vừa relevant vừa diverse.

    MMR = λ * sim(query, doc) - (1-λ) * max(sim(doc, selected_docs))

    Args:
        query_embedding: Vector embedding của query
        candidates: List of {'content': str, 'score': float, 'embedding': list, 'metadata': dict}
        top_k: Số lượng kết quả
        lambda_param: Trade-off giữa relevance (1.0) và diversity (0.0)

    Returns:
        List of top_k candidates selected by MMR.
    """
    selected = []
    selected_scores = {}
    remaining = list(range(len(candidates)))

    for _ in range(min(top_k, len(candidates))):
        best_idx = None
        best_score = float("-inf")

        for idx in remaining:
            embedding = candidates[idx].get("embedding", [])
            relevance = cosine_sim(query_embedding, embedding)

            max_sim_to_selected = 0.0
            for selected_idx in selected:
                selected_embedding = candidates[selected_idx].get("embedding", [])
                max_sim_to_selected = max(
                    max_sim_to_selected,
                    cosine_sim(embedding, selected_embedding),
                )

            mmr_score = lambda_param * relevance - (1 - lambda_param) * max_sim_to_selected
            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        if best_idx is None:
            break

        selected.append(best_idx)
        selected_scores[best_idx] = best_score
        remaining.remove(best_idx)

    results = []
    for idx in selected:
        item = candidates[idx].copy()
        item["original_score"] = float(item.get("score", 0.0))
        item["score"] = float(selected_scores[idx])
        item["reranker"] = "mmr"
        results.append(item)

    return results


def rerank_rrf(
    ranked_lists: list[list[dict]], top_k: int = 5, k: int = 60
) -> list[dict]:
    """
    Reciprocal Rank Fusion — gộp kết quả từ nhiều ranker.

    RRF(d) = Σ 1 / (k + rank_r(d))

    Args:
        ranked_lists: List of ranked result lists (mỗi list từ 1 ranker)
        top_k: Số lượng kết quả cuối cùng
        k: Smoothing constant (default=60, từ paper Cormack et al. 2009)

    Returns:
        List of top_k candidates sorted by RRF score descending.
    """
    rrf_scores = {}
    content_map = {}

    for ranked_list in ranked_lists:
        for rank, item in enumerate(ranked_list, 1):
            key = item.get("content", "")
            if not key:
                continue
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1 / (k + rank)
            content_map[key] = item

    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)

    results = []
    for content, score in sorted_items[:top_k]:
        item = content_map[content].copy()
        item["score"] = float(score)
        item["reranker"] = "rrf"
        results.append(item)

    return results


# =============================================================================
# Main rerank interface
# =============================================================================

def rerank(
    query: str,
    candidates: list[dict],
    top_k: int = 5,
    method: str = "cross_encoder",  # "cross_encoder" | "mmr" | "rrf"
) -> list[dict]:
    """
    Unified reranking interface.

    Args:
        query: Câu truy vấn
        candidates: Danh sách candidates từ retrieval
        top_k: Số lượng kết quả sau rerank
        method: Phương pháp reranking

    Returns:
        List of top_k reranked candidates.
    """
    if method == "cross_encoder":
        return rerank_cross_encoder(query, candidates, top_k)
    elif method == "mmr":
        # Cần query_embedding - embed query trước
        raise NotImplementedError("Call rerank_mmr with query_embedding")
    elif method == "rrf":
        # RRF cần nhiều ranked lists - gọi riêng
        raise NotImplementedError("Call rerank_rrf with ranked_lists")
    else:
        raise ValueError(f"Unknown rerank method: {method}")


if __name__ == "__main__":
    # Test with dummy data
    dummy_candidates = [
        {"content": "Điều 248: Tội tàng trữ trái phép chất ma tuý", "score": 0.8, "metadata": {}},
        {"content": "Nghệ sĩ X bị bắt vì sử dụng ma tuý", "score": 0.7, "metadata": {}},
        {"content": "Hình phạt tù từ 2-7 năm cho tội tàng trữ", "score": 0.6, "metadata": {}},
    ]
    results = rerank("hình phạt tàng trữ ma tuý", dummy_candidates, top_k=2)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content']}")
