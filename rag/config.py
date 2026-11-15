from __future__ import annotations
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    # storage
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "./data"))
    collection: str = "transcript_chunks"

    # models
    embed_model: str = "BAAI/bge-m3"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    use_fp16: bool = os.getenv("RAG_FP16", "0") == "1"

    # local LLM via Ollama (used by the LangGraph agent in graph.py)
    llm_model: str = os.getenv("RAG_LLM", "qwen3.5:4b-q4_K_M")
    ollama_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    llm_num_ctx: int = int(os.getenv("RAG_NUM_CTX", "16384"))      # Ollama's small default would silently truncate evidence
    llm_num_predict: int = int(os.getenv("RAG_NUM_PREDICT", "1024"))  # caps runaway generations
    llm_thinking: bool = os.getenv("RAG_THINKING", "0") == "1"     # qwen3.5 thinks by default; off = faster, cleaner JSON
    llm_faithfulness_check: bool = os.getenv("RAG_FAITH_CHECK", "1") == "1"

    # chunking (tokens are estimated as words * 1.35)
    short_turn_words: int = 8        # turns shorter than this are attached to their context
    min_chunk_tokens: int = 80
    target_chunk_tokens: int = 220
    max_chunk_tokens: int = 320
    gap_seconds: int = 90            # timestamp discontinuity signal

    # retrieval
    dense_k: int = 30
    sparse_k: int = 30               # BGE-M3 learned sparse
    bm25_k: int = 30                 # exact-token lexical (6737P, c7i.16xlarge, p95 ...)
    rrf_k: int = 60
    fused_k: int = 40
    rerank_k: int = 6
    neighbor_window: int = 1         # turns added before/after each hit

    # agent loop bounds (sized for a 4B model)
    max_iterations: int = 3
    max_retrieval_calls: int = 8
    max_subqueries: int = 4
    max_evidence_turns: int = 24
    max_revisions: int = 1


SETTINGS = Settings()
