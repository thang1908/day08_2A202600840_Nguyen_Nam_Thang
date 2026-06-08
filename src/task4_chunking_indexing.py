"""
Task 4 — Chunking & Indexing vào Vector Store.

Hướng dẫn:
    1. Đọc toàn bộ markdown files từ data/standardized/
    2. Chọn 1 chunking strategy (giải thích lý do)
    3. Chọn 1 embedding model (giải thích lý do)
    4. Index vào vector store (Weaviate khuyến cáo)

Chunking options (langchain-text-splitters):
    - RecursiveCharacterTextSplitter: an toàn, phổ biến
    - MarkdownHeaderTextSplitter: tốt cho file có heading
    - SemanticChunker: dùng embedding để tách (nâng cao)

Embedding model options:
    - sentence-transformers/all-MiniLM-L6-v2 (384 dim, nhẹ)
    - BAAI/bge-m3 (1024 dim, multilingual, tốt cho tiếng Việt)
    - OpenAI text-embedding-3-small (1536 dim, API)

Vector store options:
    - Weaviate (khuyến cáo: hỗ trợ hybrid search built-in)
    - ChromaDB (đơn giản, local)
    - FAISS (chỉ dense search)

Cài đặt:
    pip install langchain-text-splitters openai python-dotenv weaviate-client
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

STANDARDIZED_DIR = Path(__file__).parent.parent / "data" / "standardized"


# =============================================================================
# CONFIGURATION — Giải thích lựa chọn của bạn trong comment
# =============================================================================

# RecursiveCharacterTextSplitter an toàn cho cả luật và báo vì ưu tiên tách theo đoạn,
# câu rồi mới tới từ/ký tự. 500 ký tự đủ ngắn cho retrieval, overlap 50 giữ ngữ cảnh.
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
CHUNKING_METHOD = "recursive"  # "recursive" | "markdown_header" | "semantic"

# Dùng OpenAI Embeddings để dễ đổi model qua .env mà không phải sửa code.
# Mặc định text-embedding-3-small nhẹ, rẻ, đủ tốt cho demo RAG.
EMBEDDING_PROVIDER = "openai"
EMBEDDING_MODEL = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
OPENAI_EMBEDDING_DIM = os.getenv("OPENAI_EMBEDDING_DIM", "")
EMBEDDING_DIM = int(OPENAI_EMBEDDING_DIM or 1536)
EMBEDDING_BATCH_SIZE = int(os.getenv("OPENAI_EMBEDDING_BATCH_SIZE", "64"))

# Weaviate được chọn vì hỗ trợ hybrid search dense + BM25 cho các task sau.
VECTOR_STORE = "weaviate"  # "weaviate" | "chromadb" | "faiss"
WEAVIATE_COLLECTION = os.getenv("WEAVIATE_COLLECTION", "DrugLawDocs")
WEAVIATE_HOST = os.getenv("WEAVIATE_HOST", "localhost")
WEAVIATE_PORT = int(os.getenv("WEAVIATE_PORT", "8080"))
WEAVIATE_GRPC_PORT = int(os.getenv("WEAVIATE_GRPC_PORT", "50051"))


# =============================================================================
# IMPLEMENTATION
# =============================================================================

def load_documents() -> list[dict]:
    """
    Đọc toàn bộ markdown files từ data/standardized/.

    Returns:
        List of {'content': str, 'metadata': {'source': str, 'type': str}}
    """
    documents = []

    for md_file in sorted(STANDARDIZED_DIR.rglob("*.md")):
        content = md_file.read_text(encoding="utf-8").strip()
        if not content:
            continue

        relative_path = md_file.relative_to(STANDARDIZED_DIR)
        doc_type = relative_path.parts[0] if relative_path.parts else "unknown"

        documents.append({
            "content": content,
            "metadata": {
                "source": md_file.name,
                "path": str(relative_path),
                "type": doc_type,
            },
        })

    return documents


def simple_split_text(text: str) -> list[str]:
    """Fallback chunker nếu chưa cài langchain-text-splitters."""
    chunks = []
    start = 0

    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP

    return [chunk for chunk in chunks if chunk]


def chunk_documents(documents: list[dict]) -> list[dict]:
    """
    Chunk documents theo strategy đã chọn.

    Returns:
        List of {'content': str, 'metadata': dict} — mỗi item là 1 chunk
    """
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        split_text = splitter.split_text
    except ImportError:
        split_text = simple_split_text

    chunks = []
    for doc in documents:
        splits = split_text(doc["content"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunking_method": CHUNKING_METHOD,
                },
            })

    return chunks


def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Embed toàn bộ chunks bằng model đã chọn.

    Returns:
        Mỗi chunk dict được thêm key 'embedding': list[float]
    """
    from openai import OpenAI

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("Thiếu OPENAI_API_KEY trong .env")

    client = OpenAI()
    texts = [chunk["content"] for chunk in chunks]
    embeddings = []

    for start in range(0, len(texts), EMBEDDING_BATCH_SIZE):
        batch = texts[start:start + EMBEDDING_BATCH_SIZE]
        request = {
            "model": EMBEDDING_MODEL,
            "input": batch,
        }
        if OPENAI_EMBEDDING_DIM:
            request["dimensions"] = EMBEDDING_DIM

        response = client.embeddings.create(**request)
        embeddings.extend(item.embedding for item in response.data)
        print(f"  Embedded {min(start + EMBEDDING_BATCH_SIZE, len(texts))}/{len(texts)} chunks")

    for chunk, embedding in zip(chunks, embeddings):
        chunk["embedding"] = embedding

    return chunks


def connect_weaviate():
    """Kết nối Weaviate local chạy bằng Docker."""
    import weaviate

    try:
        return weaviate.connect_to_local(
            host=WEAVIATE_HOST,
            port=WEAVIATE_PORT,
            grpc_port=WEAVIATE_GRPC_PORT,
        )
    except Exception as exc:
        raise RuntimeError(
            "Không kết nối được Weaviate local. "
            "Hãy chạy từ thư mục project:\n"
            "  docker compose up -d weaviate\n\n"
            f"Sau đó kiểm tra: http://{WEAVIATE_HOST}:{WEAVIATE_PORT}/v1/.well-known/ready"
        ) from exc


def index_to_vectorstore(chunks: list[dict]):
    """
    Lưu chunks vào vector store đã chọn.
    """
    if VECTOR_STORE != "weaviate":
        raise ValueError(f"Vector store chưa được hỗ trợ: {VECTOR_STORE}")

    from weaviate.classes.config import Configure, DataType, Property

    client = connect_weaviate()
    collection_name = WEAVIATE_COLLECTION

    if client.collections.exists(collection_name):
        client.collections.delete(collection_name)

    collection = client.collections.create(
        name=collection_name,
        vector_config=Configure.Vectors.self_provided(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="path", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
            Property(name="chunk_index", data_type=DataType.INT),
        ],
    )

    with collection.batch.dynamic() as batch:
        for chunk in chunks:
            metadata = chunk["metadata"]
            batch.add_object(
                properties={
                    "content": chunk["content"],
                    "source": metadata["source"],
                    "path": metadata["path"],
                    "doc_type": metadata["type"],
                    "chunk_index": metadata["chunk_index"],
                },
                vector=chunk["embedding"],
            )

    client.close()


def run_pipeline():
    """Chạy toàn bộ pipeline: load → chunk → embed → index."""
    print("=" * 50)
    print("Task 4: Chunking & Indexing")
    print(f"  Chunking: {CHUNKING_METHOD} (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})")
    print(f"  Embedding: {EMBEDDING_MODEL} (dim={EMBEDDING_DIM})")
    print(f"  Vector Store: {VECTOR_STORE}")
    print("=" * 50)

    docs = load_documents()
    print(f"\n✓ Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"✓ Created {len(chunks)} chunks")

    chunks = embed_chunks(chunks)
    print(f"✓ Embedded {len(chunks)} chunks")

    index_to_vectorstore(chunks)
    print("✓ Indexed to vector store")


if __name__ == "__main__":
    run_pipeline()
