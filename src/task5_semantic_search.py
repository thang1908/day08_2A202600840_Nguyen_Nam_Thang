"""
Task 5 — Semantic Search Module.

Viết module tìm kiếm ngữ nghĩa (dense retrieval) trên vector store.

Yêu cầu:
    - Input: query string + top_k
    - Output: danh sách chunks có score, sorted descending
    - Phải tương thích với embedding model và vector store ở Task 4
"""

import math
import os
import re

from dotenv import load_dotenv

try:
    from src.task4_chunking_indexing import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        OPENAI_EMBEDDING_DIM,
        WEAVIATE_COLLECTION,
        chunk_documents,
        connect_weaviate,
        load_documents,
    )
except ModuleNotFoundError:
    from task4_chunking_indexing import (
        EMBEDDING_DIM,
        EMBEDDING_MODEL,
        OPENAI_EMBEDDING_DIM,
        WEAVIATE_COLLECTION,
        chunk_documents,
        connect_weaviate,
        load_documents,
    )

load_dotenv()

LOCAL_CORPUS: list[dict] = []


def embed_query(query: str) -> list[float]:
    """Embed query bằng cùng OpenAI embedding model đã dùng ở Task 4."""
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Thiếu OPENAI_API_KEY trong .env")

    request = {
        "model": EMBEDDING_MODEL,
        "input": query,
    }
    if OPENAI_EMBEDDING_DIM:
        request["dimensions"] = EMBEDDING_DIM

    response = OpenAI().embeddings.create(**request)
    return response.data[0].embedding


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho fallback local semantic search."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


def cosine_from_counts(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Cosine similarity trên bag-of-words để fallback không cần API/index."""
    if not query_tokens or not doc_tokens:
        return 0.0

    query_counts = {}
    doc_counts = {}
    for token in query_tokens:
        query_counts[token] = query_counts.get(token, 0) + 1
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1

    common_tokens = set(query_counts) & set(doc_counts)
    dot = sum(query_counts[token] * doc_counts[token] for token in common_tokens)
    query_norm = math.sqrt(sum(count * count for count in query_counts.values()))
    doc_norm = math.sqrt(sum(count * count for count in doc_counts.values()))
    if query_norm == 0 or doc_norm == 0:
        return 0.0
    return dot / (query_norm * doc_norm)


def local_semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Fallback tìm kiếm local khi OpenAI/Weaviate chưa sẵn sàng.

    Vẫn giữ cùng schema với semantic_search để test và pipeline demo chạy được.
    """
    global LOCAL_CORPUS

    if not LOCAL_CORPUS:
        LOCAL_CORPUS = chunk_documents(load_documents())

    query_tokens = tokenize(query)
    results = []

    for chunk in LOCAL_CORPUS:
        content = chunk.get("content", "")
        score = cosine_from_counts(query_tokens, tokenize(content))
        if score <= 0:
            continue

        results.append({
            "content": content,
            "score": float(score),
            "metadata": chunk.get("metadata", {}) or {},
        })

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def semantic_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm ngữ nghĩa sử dụng vector similarity.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,      # Nội dung chunk
            'score': float,      # Cosine similarity score
            'metadata': dict     # source, doc_type, chunk_index
        }
        Sorted by score descending.
    """
    try:
        from weaviate.classes.query import MetadataQuery

        query_embedding = embed_query(query)
        client = connect_weaviate()

        try:
            if not client.collections.exists(WEAVIATE_COLLECTION):
                return local_semantic_search(query, top_k=top_k)

            collection = client.collections.get(WEAVIATE_COLLECTION)
            response = collection.query.near_vector(
                near_vector=query_embedding,
                limit=top_k,
                return_metadata=MetadataQuery(distance=True),
            )

            results = []
            for obj in response.objects:
                props = obj.properties
                distance = obj.metadata.distance
                score = 1 - distance if distance is not None else 0.0
                results.append({
                    "content": props.get("content", ""),
                    "score": float(score),
                    "metadata": {
                        "source": props.get("source"),
                        "path": props.get("path"),
                        "type": props.get("doc_type"),
                        "chunk_index": props.get("chunk_index"),
                    },
                })

            if results:
                return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
            return local_semantic_search(query, top_k=top_k)
        finally:
            client.close()
    except Exception as exc:
        print(f"⚠ Semantic search fallback local: {exc}")
        return local_semantic_search(query, top_k=top_k)


if __name__ == "__main__":
    # Test
    results = semantic_search("hình phạt cho tội tàng trữ ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
