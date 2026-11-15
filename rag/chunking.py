"""Multi-signal chunking for speaker-turn transcripts.

Signals combined (embedding similarity is only ONE of them):
  1. speaker-turn boundaries  - chunks are made of whole turns (long turns split on sentences)
  2. question->answer links   - a question turn and its answer are an atomic "exchange unit"
  3. short-turn attachment    - short replies are attached to context but keep their own
                                timestamp/speaker line inside the chunk
  4. token limits             - min / target / max
  5. timestamp continuity     - large gaps suggest a new segment
  6. lexical topic cues       - "let's look at...", "before we wrap up", ...
  7. semantic similarity drop - soft signal between consecutive exchange units
"""
from __future__ import annotations
import re
import numpy as np

from .config import Settings
from .parsing import Turn

CUE_RE = re.compile(
    r"let'?s (?:dig|look|talk|start|move|get)|before we wrap|walk me through|tell me about|"
    r"looking ahead|here is what we can|two main dependencies|on the hardware side|"
    r"on the software side|let me (?:double|check)",
    re.I,
)


def est_tokens(text: str) -> int:
    return int(len(text.split()) * 1.35) + 1


def build_units(turns: list[Turn], cfg: Settings) -> list[list[int]]:
    """Group turn indices into exchange units (Q+A, plus trailing short acknowledgements)."""
    units, i, n = [], 0, len(turns)
    while i < n:
        unit = [i]
        if turns[i].is_question and i + 1 < n and turns[i + 1].speaker != turns[i].speaker:
            i += 1
            unit.append(i)
        while (i + 1 < n and len(turns[i + 1].text.split()) < cfg.short_turn_words
               and not turns[i + 1].is_question):
            i += 1
            unit.append(i)
        units.append(unit)
        i += 1
    return units


def _split_turn(turn: Turn, max_tokens: int) -> list[tuple[int, str]]:
    if est_tokens(turn.text) <= max_tokens:
        return [(turn.idx, turn.text)]
    pieces, cur, tok = [], [], 0
    for s in re.split(r"(?<=[.!?])\s+", turn.text):
        t = est_tokens(s)
        if cur and tok + t > max_tokens:
            pieces.append((turn.idx, " ".join(cur)))
            cur, tok = [], 0
        cur.append(s)
        tok += t
    if cur:
        pieces.append((turn.idx, " ".join(cur)))
    return pieces


def render(turns: list[Turn], pieces: list[tuple[int, str]]) -> str:
    order, buf = [], {}
    for idx, txt in pieces:
        if idx not in buf:
            buf[idx] = []
            order.append(idx)
        buf[idx].append(txt)
    return "\n".join(
        f"[{turns[i].ts}] {turns[i].speaker} ({turns[i].org}): {' '.join(buf[i])}" for i in order
    )


def make_embed_text(chunk: dict, title: str) -> str:
    header = (f"Meeting: {title} | Segment {chunk['topic_id']} | "
              f"{chunk['start_ts']}-{chunk['end_ts']} | Speakers: {', '.join(chunk['speakers'])}")
    ctx = f"\nContext: {chunk['context']}" if chunk.get("context") else ""
    return f"{header}{ctx}\n{chunk['text']}"


def build_chunks(turns: list[Turn], units: list[list[int]], unit_vecs: np.ndarray | None,
                 meeting_id: str, title: str, cfg: Settings) -> list[dict]:
    infos = []
    for u in units:
        ts = [turns[i] for i in u]
        infos.append({"start": ts[0].seconds, "end": ts[-1].seconds,
                      "head": " ".join(t.text for t in ts)[:160]})

    sims, thr = None, -1.0
    if unit_vecs is not None and len(units) >= 5:
        sims = np.einsum("ij,ij->i", unit_vecs[1:], unit_vecs[:-1])      # normalised vectors
        thr = float(sims.mean() - 0.75 * sims.std())

    def boundary_score(ui: int) -> float:
        s = 0.0
        if sims is not None and sims[ui - 1] < thr:
            s += 0.5
        if infos[ui]["start"] - infos[ui - 1]["end"] > cfg.gap_seconds:
            s += 0.4
        if CUE_RE.search(infos[ui]["head"]):
            s += 0.3
        return s

    chunks: list[dict] = []
    cur: list[tuple[int, str]] = []
    cur_tok, topic = 0, 0

    def flush():
        nonlocal cur, cur_tok
        if not cur:
            return
        ids = sorted({i for i, _ in cur})
        c = {
            "chunk_id": f"{meeting_id}:c{len(chunks)}",
            "meeting_id": meeting_id,
            "turn_ids": ids, "turn_start": ids[0], "turn_end": ids[-1],
            "start_ts": turns[ids[0]].ts, "end_ts": turns[ids[-1]].ts,
            "speakers": sorted({turns[i].speaker for i in ids}),
            "orgs": sorted({turns[i].org for i in ids}),
            "topic_id": topic,
            "text": render(turns, cur),
            "context": "",
        }
        c["embed_text"] = make_embed_text(c, title)
        chunks.append(c)
        cur, cur_tok = [], 0

    for ui, u in enumerate(units):
        pieces = [p for i in u for p in _split_turn(turns[i], cfg.max_chunk_tokens)]
        utok = sum(est_tokens(t) for _, t in pieces)
        if cur:
            score = boundary_score(ui)
            hard = cur_tok + utok > cfg.max_chunk_tokens
            soft = cur_tok >= cfg.target_chunk_tokens and score >= 0.3     # any one signal
            shift = cur_tok >= cfg.min_chunk_tokens and score >= 0.7        # needs 2+ signals
            if hard or soft or shift:
                flush()
                if score >= 0.7:
                    topic += 1
        for p in pieces:                       # piece loop only splits an oversize unit
            t = est_tokens(p[1])
            if cur and cur_tok + t > cfg.max_chunk_tokens:
                flush()
            cur.append(p)
            cur_tok += t
    flush()
    return chunks
