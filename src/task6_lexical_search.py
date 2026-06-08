"""
Task 6 — Lexical Search Module (BM25).

Mặc định sử dụng BM25. Nếu dùng phương pháp khác (TF-IDF, Elasticsearch,
Weaviate BM25 built-in), hãy giải thích cơ chế trong buổi demo → +5 bonus.

Cài đặt:
    pip install rank-bm25

BM25 hoạt động thế nào:
    - Term Frequency (TF): từ xuất hiện nhiều trong document → điểm cao
    - Inverse Document Frequency (IDF): từ hiếm → quan trọng hơn
    - Document length normalization: document dài không bị ưu tiên quá mức
    - Formula: score(q,d) = Σ IDF(qi) * (tf(qi,d) * (k1+1)) / (tf(qi,d) + k1*(1-b+b*|d|/avgdl))
    - k1=1.5 (term saturation), b=0.75 (length normalization)
"""

import math
import re

try:
    from src.task4_chunking_indexing import chunk_documents, load_documents
except ModuleNotFoundError:
    from task4_chunking_indexing import chunk_documents, load_documents

CORPUS: list[dict] = []
BM25_INDEX = None


def tokenize(text: str) -> list[str]:
    """Tokenize đơn giản cho tiếng Việt: lowercase và giữ chữ/số."""
    return re.findall(r"[\wÀ-ỹ]+", text.lower())


class SimpleBM25:
    """Fallback BM25 nhỏ gọn nếu chưa cài rank-bm25."""

    def __init__(self, tokenized_corpus: list[list[str]], k1: float = 1.5, b: float = 0.75):
        self.tokenized_corpus = tokenized_corpus
        self.k1 = k1
        self.b = b
        self.doc_count = len(tokenized_corpus)
        self.doc_lengths = [len(doc) for doc in tokenized_corpus]
        self.avg_doc_length = sum(self.doc_lengths) / max(self.doc_count, 1)
        self.term_freqs = []
        self.doc_freqs = {}

        for doc in tokenized_corpus:
            frequencies = {}
            for token in doc:
                frequencies[token] = frequencies.get(token, 0) + 1
            self.term_freqs.append(frequencies)
            for token in frequencies:
                self.doc_freqs[token] = self.doc_freqs.get(token, 0) + 1

    def get_scores(self, query_tokens: list[str]) -> list[float]:
        scores = []

        for doc_index, frequencies in enumerate(self.term_freqs):
            doc_length = self.doc_lengths[doc_index]
            score = 0.0

            for token in query_tokens:
                term_frequency = frequencies.get(token, 0)
                if term_frequency == 0:
                    continue

                doc_frequency = self.doc_freqs.get(token, 0)
                idf = math.log(1 + (self.doc_count - doc_frequency + 0.5) / (doc_frequency + 0.5))
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * doc_length / max(self.avg_doc_length, 1)
                )
                score += idf * (term_frequency * (self.k1 + 1)) / denominator

            scores.append(score)

        return scores


def load_corpus() -> list[dict]:
    """Load corpus từ markdown đã chuẩn hoá và chunk giống Task 4."""
    documents = load_documents()
    return chunk_documents(documents)


def build_bm25_index(corpus: list[dict]):
    """
    Xây dựng BM25 index từ corpus.

    Args:
        corpus: List of {'content': str, 'metadata': dict}
    """
    tokenized_corpus = [tokenize(doc["content"]) for doc in corpus]

    try:
        from rank_bm25 import BM25Okapi

        return BM25Okapi(tokenized_corpus)
    except ImportError:
        return SimpleBM25(tokenized_corpus)


def lexical_search(query: str, top_k: int = 10) -> list[dict]:
    """
    Tìm kiếm từ khóa sử dụng BM25.

    Args:
        query: Câu truy vấn
        top_k: Số lượng kết quả tối đa

    Returns:
        List of {
            'content': str,
            'score': float,      # BM25 score
            'metadata': dict
        }
        Sorted by score descending.
    """
    global CORPUS, BM25_INDEX

    if not CORPUS:
        CORPUS = load_corpus()
    if BM25_INDEX is None:
        BM25_INDEX = build_bm25_index(CORPUS)

    tokenized_query = tokenize(query)
    scores = BM25_INDEX.get_scores(tokenized_query)
    ranked_indices = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)

    results = []
    for index in ranked_indices[:top_k]:
        score = float(scores[index])
        if score <= 0:
            continue

        results.append({
            "content": CORPUS[index]["content"],
            "score": score,
            "metadata": CORPUS[index]["metadata"],
        })

    return results


if __name__ == "__main__":
    # Test
    results = lexical_search("Điều 248 tàng trữ trái phép chất ma tuý", top_k=5)
    for r in results:
        print(f"[{r['score']:.3f}] {r['content'][:100]}...")
