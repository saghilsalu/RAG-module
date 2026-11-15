"""Ollama chat model + robust helpers for small local models."""
from __future__ import annotations
import re
from functools import lru_cache

from .config import Settings, SETTINGS

_THINK = re.compile(r"<think>.*?</think>", re.S)


@lru_cache(maxsize=1)
def get_llm(cfg: Settings = SETTINGS):
    from langchain_ollama import ChatOllama
    return ChatOllama(
        model=cfg.llm_model,
        base_url=cfg.ollama_url,
        temperature=0,
        seed=42,
        num_ctx=cfg.llm_num_ctx,
        num_predict=cfg.llm_num_predict,
        reasoning=cfg.llm_thinking,      # False -> no thinking block
        keep_alive="30m",
    )


def text_of(msg) -> str:
    """Plain text of a model reply, with any leaked <think> block removed."""
    content = getattr(msg, "content", msg)
    return _THINK.sub("", content if isinstance(content, str) else str(content)).strip()


def structured(llm, schema, prompt: str, default, retries: int = 1):
    """Schema-constrained output through Ollama's `format` (JSON-schema) option - this avoids
    Ollama's tool-calling path, which has had Qwen3.5 parser bugs. Falls back to `default`
    so one malformed reply from a 4B model cannot crash the graph."""
    runner = llm.with_structured_output(schema, method="json_schema")
    for _ in range(retries + 1):
        try:
            out = runner.invoke(prompt)
            if out is not None:
                return out
        except Exception:
            continue
    return default
