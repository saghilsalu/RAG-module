"""Models + storage + hybrid retrieval.

Three SEPARATE retrieval paths, fused with Reciprocal Rank Fusion:
  dense   : BGE-M3 dense vectors in ChromaDB (cosine)
  sparse  : BGE-M3 learned lexical weights, kept in our own inverted index
            (Chroma does not hold BGE-M3 sparse vectors for you)
  bm25    : classic BM25 with a tokenizer that keeps 6737P, c7i.16xlarge, p95, 91.5 intact
Then a cross-encoder reranker (bge-reranker-v2-m3).
"""
from __future__ import annotations
import json
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import Settings, SETTINGS

# ---------------------------------------------------------------- models

class Embedder:
    def __init__(self, cfg: Settings):
        from FlagEmbedding import BGEM3FlagModel
        self.model = BGEM3FlagModel(cfg.embed_model, use_fp16=cfg.use_fp16)

    def encode(self, texts: list[str], batch_size: int = 16):
        out = self.model.encode(texts, batch_size=batch_size, max_length=1024,
                                return_dense=True, return_sparse=True, return_colbert_vecs=False)
        dense = np.asarray(out["dense_vecs"], dtype=np.float32)
        dense /= np.linalg.norm(dense, axis=1, keepdims=True) + 1e-12
        sparse = [{str(k): float(v) for k, v in lw.items()} for lw in out["lexical_weights"]]
        return dense, sparse


class Reranker:
    def __init__(self, cfg: Settings):
        from FlagEmbedding import FlagReranker
        self.model = FlagReranker(cfg.rerank_model, use_fp16=cfg.use_fp16)

    def rerank(self, query: str, chunks: list[dict], top_k: int) -> list[tuple[dict, float]]:
        if not chunks:
            return []
        scores = self.model.compute_score([[query, c["embed_text"]] for c in chunks], normalize=True)
        if not isinstance(scores, list):
            scores = [scores]
        return sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)[:top_k]


@lru_cache(maxsize=1)
def get_embedder(cfg: Settings = SETTINGS) -> Embedder:
    return Embedder(cfg)


@lru_cache(maxsize=1)
def get_reranker(cfg: Settings = SETTINGS) -> Reranker:
    return Reranker(cfg)

# ---------------------------------------------------------------- corpus

_TOK = re.compile(r"[a-z0-9]+(?:[.\-_][a-z0-9]+)*")


def lex_tokens(text: str) -> list[str]:
    toks = _TOK.findall(text.lower())
    extra = [p for t in toks for p in re.split(r"[.\-_]", t) if p and p != t]
    return toks + extra


def rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for r, cid in enumerate(ranking):
            fused[cid] += 1.0 / (k + r + 1)
    return fused


class Corpus:
    def __init__(self, cfg: Settings = SETTINGS):
        import chromadb
        self.cfg = cfg
        (cfg.data_dir / "meetings").mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(cfg.data_dir / "chroma"))
        self.col = self.client.get_or_create_collection(
            cfg.collection, metadata={"hnsw:space": "cosine"})
        self.reload()

    # ---- persistence
    def _mdir(self, mid: str) -> Path:
        return self.cfg.data_dir / "meetings" / mid

    def add_meeting(self, mid: str, turns: list[dict], chunks: list[dict],
                    dense: np.ndarray, sparse: list[dict]) -> None:
        d = self._mdir(mid)
        d.mkdir(parents=True, exist_ok=True)
        (d / "turns.json").write_text(json.dumps(turns))
        (d / "chunks.json").write_text(json.dumps(chunks))
        (d / "sparse.json").write_text(json.dumps(dict(zip([c["chunk_id"] for c in chunks], sparse))))
        try:
            self.col.delete(where={"meeting_id": mid})       # idempotent re-ingest
        except Exception:
            pass
        self.col.upsert(
            ids=[c["chunk_id"] for c in chunks],
            embeddings=dense.tolist(),
            documents=[c["embed_text"] for c in chunks],
            metadatas=[{"meeting_id": mid, "turn_start": c["turn_start"], "turn_end": c["turn_end"],
                        "start_ts": c["start_ts"], "topic_id": c["topic_id"],
                        "speakers": "|".join(c["speakers"]), "orgs": "|".join(c["orgs"])}
                       for c in chunks],
        )
        self.reload()

    def reload(self) -> None:
        self.turns: dict[str, list[dict]] = {}
        self.chunks: dict[str, dict] = {}
        sparse: dict[str, dict] = {}
        for d in sorted((self.cfg.data_dir / "meetings").glob("*")):
            if not (d / "chunks.json").exists():
                continue
            self.turns[d.name] = json.loads((d / "turns.json").read_text())
            for c in json.loads((d / "chunks.json").read_text()):
                self.chunks[c["chunk_id"]] = c
            sparse.update(json.loads((d / "sparse.json").read_text()))
        self.ids = list(self.chunks)
        self.inv: dict[str, list[tuple[str, float]]] = defaultdict(list)   # sparse inverted index
        for cid, w in sparse.items():
            for tok, val in w.items():
                self.inv[tok].append((cid, val))
        from rank_bm25 import BM25Okapi
        self.bm25 = BM25Okapi([lex_tokens(self.chunks[c]["text"] + " " + " ".join(self.chunks[c]["speakers"]))
                               for c in self.ids]) if self.ids else None

    # ---- three retrieval paths
    def _ok(self, cid: str, meeting_id: str | None) -> bool:
        return meeting_id is None or self.chunks[cid]["meeting_id"] == meeting_id

    def dense_search(self, qvec: np.ndarray, k: int, meeting_id: str | None) -> list[str]:
        n = self.col.count()
        if n == 0:
            return []
        kw = {"where": {"meeting_id": meeting_id}} if meeting_id else {}
        res = self.col.query(query_embeddings=[qvec.tolist()], n_results=min(k, n), **kw)
        return res["ids"][0]

    def sparse_search(self, qw: dict, k: int, meeting_id: str | None) -> list[str]:
        scores: dict[str, float] = defaultdict(float)
        for tok, qv in qw.items():
            for cid, dv in self.inv.get(tok, ()):
                if self._ok(cid, meeting_id):
                    scores[cid] += qv * dv
        return [c for c, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]

    def bm25_search(self, query: str, k: int, meeting_id: str | None) -> list[str]:
        if not self.bm25:
            return []
        sc = self.bm25.get_scores(lex_tokens(query))
        order = np.argsort(-sc)
        return [self.ids[i] for i in order if sc[i] > 0 and self._ok(self.ids[i], meeting_id)][:k]

    def hybrid_search(self, query: str, meeting_id: str | None = None) -> list[tuple[str, float]]:
        cfg = self.cfg
        dense, sparse = get_embedder(cfg).encode([query])
        fused = rrf([self.dense_search(dense[0], cfg.dense_k, meeting_id),
                     self.sparse_search(sparse[0], cfg.sparse_k, meeting_id),
                     self.bm25_search(query, cfg.bm25_k, meeting_id)], cfg.rrf_k)
        return sorted(fused.items(), key=lambda x: x[1], reverse=True)[:cfg.fused_k]

    # ---- context expansion (turn level)
    def expand(self, chunk: dict, window: int) -> list[tuple[int, bool]]:
        n = len(self.turns[chunk["meeting_id"]])
        lo, hi = max(0, chunk["turn_start"] - window), min(n - 1, chunk["turn_end"] + window)
        core = set(chunk["turn_ids"])
        return [(i, i in core) for i in range(lo, hi + 1)]
