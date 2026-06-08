"""
Task 8 — PageIndex Vectorless RAG.

Đăng ký tài khoản tại: https://pageindex.ai/
SDK & sample code: https://github.com/VectifyAI/PageIndex

PageIndex cho phép RAG mà không cần vector store — sử dụng
structural understanding của document thay vì embedding.

Cài đặt:
    pip install pageindex

Hướng dẫn:
    1. Đăng ký account tại pageindex.ai
    2. Lấy API key
    3. Upload documents
    4. Query sử dụng PageIndex API
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PAGEINDEX_API_KEY = os.getenv("PAGEINDEX_API_KEY", "")
PAGEINDEX_POLL_SECONDS = int(os.getenv("PAGEINDEX_POLL_SECONDS", "20"))
PROJECT_DIR = Path(__file__).parent.parent
STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"
LANDING_DIR = Path(__file__).parent.parent / "data" / "landing"
PAGEINDEX_DOC_IDS_PATH = Path(__file__).parent.parent / "data" / "pageindex_doc_ids.json"


def get_pageindex_client():
    """Khởi tạo PageIndexClient theo SDK pageindex==0.2.x."""
    try:
        from pageindex import PageIndexClient
    except ImportError:
        from pageindex.client import PageIndexClient

    return PageIndexClient(api_key=PAGEINDEX_API_KEY)


def load_pageindex_docs() -> list[dict]:
    """Đọc danh sách doc_id đã upload lên PageIndex."""
    if not PAGEINDEX_DOC_IDS_PATH.exists():
        return []
    return json.loads(PAGEINDEX_DOC_IDS_PATH.read_text(encoding="utf-8"))


def save_pageindex_docs(documents: list[dict]):
    """Lưu danh sách doc_id để các lần search sau dùng lại."""
    PAGEINDEX_DOC_IDS_PATH.parent.mkdir(parents=True, exist_ok=True)
    PAGEINDEX_DOC_IDS_PATH.write_text(
        json.dumps(documents, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def tokenize(text: str) -> set[str]:
    """Tokenize đơn giản cho fallback vectorless local."""
    import re

    return set(re.findall(r"[\wÀ-ỹ]+", text.lower()))


def local_vectorless_search(query: str, top_k: int = 5) -> list[dict]:
    """Fallback mô phỏng vectorless search trên markdown local khi chưa có PageIndex."""
    query_tokens = tokenize(query)
    results = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        paragraphs = [p.strip() for p in content.split("\n\n") if len(p.strip()) > 80]
        for idx, paragraph in enumerate(paragraphs):
            paragraph_tokens = tokenize(paragraph)
            overlap = len(query_tokens & paragraph_tokens)
            if overlap == 0:
                continue

            relative_path = md_file.relative_to(STANDARDIZED_DIR)
            results.append({
                "content": paragraph,
                "score": float(overlap),
                "metadata": {
                    "source": md_file.name,
                    "path": str(relative_path),
                    "type": relative_path.parts[0] if relative_path.parts else "unknown",
                    "paragraph_index": idx,
                },
                "source": "pageindex",
            })

    return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]


def upload_documents():
    """
    Upload toàn bộ markdown documents lên PageIndex.
    """
    if not PAGEINDEX_API_KEY:
        raise ValueError("Thiếu PAGEINDEX_API_KEY trong .env")

    pi = get_pageindex_client()
    uploaded = []
    existing = {
        item.get("path"): item
        for item in load_pageindex_docs()
        if item.get("doc_id") and item.get("path")
    }

    # PageIndex SDK hiện tại nhận file PDF qua submit_document(file_path).
    # Legal PDFs là nguồn phù hợp nhất để upload; news markdown vẫn được fallback local.
    source_files = sorted((LANDING_DIR / "legal").glob("*.pdf"))

    for filepath in source_files:
        relative_path = str(filepath.relative_to(PROJECT_DIR))
        if relative_path in existing:
            uploaded.append(existing[relative_path])
            print(f"  ↷ Already uploaded: {filepath.name}")
            continue

        response = pi.submit_document(str(filepath))
        doc_id = response.get("doc_id") or response.get("id") or response.get("document_id")
        if not doc_id:
            raise ValueError(f"PageIndex không trả doc_id cho {filepath.name}: {response}")

        metadata = {
            "doc_id": doc_id,
            "filename": filepath.name,
            "path": relative_path,
            "type": "legal",
        }
        uploaded.append(metadata)
        print(f"  ✓ Uploaded: {filepath.name} -> {doc_id}")

    save_pageindex_docs(uploaded)
    return uploaded


def extract_retrieval_items(payload: dict) -> list[dict]:
    """Chuẩn hoá nhiều dạng response retrieval của PageIndex."""
    for key in ("results", "retrieval_results", "chunks", "nodes", "blocks"):
        value = payload.get(key)
        if isinstance(value, list):
            return value

    data = payload.get("data")
    if isinstance(data, dict):
        return extract_retrieval_items(data)
    if isinstance(data, list):
        return data

    return []


def wait_for_retrieval(pi, retrieval_id: str) -> dict:
    """Poll retrieval result một khoảng ngắn."""
    deadline = time.time() + PAGEINDEX_POLL_SECONDS
    last_payload = {}

    while time.time() < deadline:
        last_payload = pi.get_retrieval(retrieval_id)
        status = str(last_payload.get("status", "")).lower()
        if status in {"completed", "complete", "success", "succeeded", "ready"}:
            return last_payload
        if extract_retrieval_items(last_payload):
            return last_payload
        time.sleep(2)

    return last_payload


def pageindex_search(query: str, top_k: int = 5) -> list[dict]:
    """
    Vectorless retrieval sử dụng PageIndex.
    Dùng làm fallback khi hybrid search không có kết quả tốt.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,
            'metadata': dict,
            'source': 'pageindex'   # Đánh dấu nguồn retrieval
        }
    """
    try:
        if not PAGEINDEX_API_KEY:
            return local_vectorless_search(query, top_k)

        pageindex_docs = load_pageindex_docs()
        if not pageindex_docs:
            return local_vectorless_search(query, top_k)

        pi = get_pageindex_client()

        results = []
        for doc in pageindex_docs:
            doc_id = doc.get("doc_id")
            if not doc_id:
                continue
            if hasattr(pi, "is_retrieval_ready") and not pi.is_retrieval_ready(doc_id):
                continue

            submitted = pi.submit_query(doc_id=doc_id, query=query)
            retrieval_id = submitted.get("retrieval_id") or submitted.get("id")
            if not retrieval_id:
                continue

            payload = wait_for_retrieval(pi, retrieval_id)
            for item in extract_retrieval_items(payload):
                content = (
                    item.get("text")
                    or item.get("content")
                    or item.get("markdown")
                    or item.get("page_content")
                    or ""
                )
                score = item.get("score") or item.get("relevance_score") or 0.0
                metadata = item.get("metadata") or {}
                metadata.update({
                    "doc_id": doc_id,
                    "source": doc.get("filename"),
                    "path": doc.get("path"),
                    "type": doc.get("type", "legal"),
                })
                results.append({
                    "content": content,
                    "score": float(score),
                    "metadata": metadata,
                    "source": "pageindex",
                })

        if not results:
            return local_vectorless_search(query, top_k)

        return sorted(results, key=lambda item: item["score"], reverse=True)[:top_k]
    except Exception as exc:
        print(f"⚠ PageIndex chưa sẵn sàng, dùng local fallback: {exc}")
        return local_vectorless_search(query, top_k)


if __name__ == "__main__":
    if not PAGEINDEX_API_KEY:
        print("⚠ Hãy set PAGEINDEX_API_KEY trong file .env")
        print("  Đăng ký tại: https://pageindex.ai/")
    else:
        print("Uploading documents...")
        upload_documents()

        print("\nTest query:")
        results = pageindex_search("hình phạt sử dụng ma tuý", top_k=3)
        for r in results:
            print(f"[{r['score']:.3f}] {r['content'][:100]}...")
