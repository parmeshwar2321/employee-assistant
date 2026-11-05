"""Hybrid retrieval: BGE dense search (Qdrant) + BM25 lexical search, fused with Reciprocal Rank Fusion."""
import re
import threading
from functools import lru_cache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, FieldCondition, Filter, FilterSelector, MatchValue, VectorParams
from rank_bm25 import BM25Okapi

from ..config import settings
from ..db import conn

BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(settings.embed_model)
        self.dim = self.model.get_sentence_embedding_dimension()
        # Embedded/local Qdrant: no server needed. Swap for QdrantClient(url=...) in production.
        self.qdrant = QdrantClient(path=str(settings.data_path / "qdrant"))
        if not self.qdrant.collection_exists(settings.collection):
            self.qdrant.create_collection(
                settings.collection, vectors_config=VectorParams(size=self.dim, distance=Distance.COSINE)
            )
        self._lock = threading.Lock()
        self._bm25: BM25Okapi | None = None
        self._chunks: list[dict] = []
        self.refresh_bm25()

    # ---- embeddings ----
    def embed_docs(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True, batch_size=32).tolist()

    def embed_query(self, q: str) -> list[float]:
        return self.model.encode(BGE_QUERY_PREFIX + q, normalize_embeddings=True).tolist()

    # ---- index maintenance ----
    def refresh_bm25(self) -> None:
        with conn() as c:
            rows = [dict(r) for r in c.execute("SELECT id, source, chunk_index, text FROM chunks")]
        with self._lock:
            self._chunks = rows
            self._bm25 = BM25Okapi([tokenize(r["text"]) for r in rows]) if rows else None

    def delete_source(self, source: str) -> None:
        self.qdrant.delete(
            settings.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="source", match=MatchValue(value=source))])
            ),
        )
        with conn() as c:
            c.execute("DELETE FROM chunks WHERE source=?", (source,))

    # ---- search ----
    def _dense(self, query: str, k: int) -> list[str]:
        res = self.qdrant.query_points(settings.collection, query=self.embed_query(query), limit=k)
        return [str(p.id) for p in res.points]

    def _sparse(self, query: str, k: int) -> list[str]:
        with self._lock:
            if not self._bm25:
                return []
            scores = self._bm25.get_scores(tokenize(query))
            top = sorted(range(len(scores)), key=scores.__getitem__, reverse=True)[:k]
            return [self._chunks[i]["id"] for i in top if scores[i] > 0]

    def search(self, query: str, k: int | None = None) -> list[dict]:
        k = k or settings.final_k
        dense = self._dense(query, settings.dense_k)
        sparse = self._sparse(query, settings.bm25_k)
        fused: dict[str, float] = {}
        for ranking in (dense, sparse):
            for rank, cid in enumerate(ranking):
                fused[cid] = fused.get(cid, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
        by_id = {c["id"]: c for c in self._chunks}
        out = []
        for cid, score in sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]:
            if cid in by_id:
                ch = by_id[cid]
                out.append({"id": cid, "source": ch["source"], "chunk_index": ch["chunk_index"],
                            "text": ch["text"], "score": round(score, 5)})
        return out


@lru_cache(maxsize=1)
def get_retriever() -> HybridRetriever:
    return HybridRetriever()
