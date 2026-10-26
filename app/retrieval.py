import time
from collections import OrderedDict
from pathlib import Path
from threading import Lock
from typing import List, Optional, Tuple

from app.config import settings
from app.models import SourceChunk
from app.vectorstore import get_embeddings, get_vectorstore


class QueryCache:
    """Thread-safe LRU Cache for query embeddings."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._store: OrderedDict[str, List[float]] = OrderedDict()
        self._lock = Lock()

    def get(self, key: str) -> Optional[List[float]]:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)
            return self._store[key]

    def put(self, key: str, value: List[float]) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self.capacity:
                self._store.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)


# Global query embedding cache instance
_query_cache = QueryCache(capacity=settings.CACHE_SIZE)


def retrieve(
    question: str, top_k: Optional[int] = None
) -> Tuple[List[SourceChunk], float, bool]:
    """
    Retrieve relevant chunks for a question.
    Returns: (chunks, retrieval_latency_ms, cache_hit)
    """
    k = top_k or settings.TOP_K
    cache_key = question.strip().lower()

    t0 = time.perf_counter()

    cached_embedding = _query_cache.get(cache_key)
    if cached_embedding is not None:
        cache_hit = True
        query_vector = cached_embedding
    else:
        cache_hit = False
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(question)
        _query_cache.put(cache_key, query_vector)

    vectorstore = get_vectorstore()

    try:
        results = vectorstore.similarity_search_by_vector_with_relevance_scores(
            query_vector, k=k
        )
    except Exception:
        # Fallback if relevance score calculation is unsupported
        docs = vectorstore.similarity_search_by_vector(query_vector, k=k)
        results = [(d, 1.0) for d in docs]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    chunks: List[SourceChunk] = []
    for doc, score in results:
        meta = doc.metadata or {}
        raw_source = meta.get("source", "unknown")
        filename = Path(raw_source).name
        raw_page = meta.get("page", 0)

        # Ensure 1-based page numbering for display
        page_num = raw_page if meta.get("page_1_indexed") else int(raw_page) + 1

        chunks.append(
            SourceChunk(
                file_name=filename,
                page=page_num,
                content=doc.page_content,
                score=round(float(score), 4) if score is not None else None,
            )
        )

    return chunks, elapsed_ms, cache_hit
