# Transcript retrieval (Docling + BGE-M3 + Chroma + hybrid + reranker) - no LLM

    pip install -r requirements.txt
    python -m rag.cli ingest Transcript.pdf
    python -m rag.cli search "how will Intel fix the inference latency?" -k 8 --hops 1
    uvicorn rag.api:app --reload      # POST /ingest, POST /search

search = BGE-M3 dense (Chroma) + BGE-M3 sparse + BM25 -> RRF -> optional feedback hop
-> bge-reranker-v2-m3 -> neighbour-turn expansion. Settings in rag/config.py.

## LLM agent (local, Ollama + qwen3.5:4b-q4_K_M)

    ollama pull qwen3.5:4b-q4_K_M
    python -m rag.cli ask "How does Intel's proposal address Assessli's inference bottleneck?" --trace
    # POST /ask {"question": "..."}

Env: RAG_LLM, OLLAMA_BASE_URL, RAG_NUM_CTX (16384), RAG_NUM_PREDICT (1024),
RAG_THINKING=1 (enable thinking), RAG_FAITH_CHECK=0 (skip the LLM faithfulness judge).
graph.py = LangGraph loop (understand -> decompose -> retrieve -> evaluate -> generate -> validate).
`search` stays LLM-free.
