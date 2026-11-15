from __future__ import annotations
import re
from dataclasses import asdict
from pathlib import Path

from .chunking import build_chunks, build_units
from .config import Settings, SETTINGS
from .parsing import clean_text, parse_transcript, pdf_to_text
from .store import Corpus, get_embedder


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip("-") or "meeting"


def ingest_pdf(path: str | Path, cfg: Settings = SETTINGS,
               meeting_id: str | None = None) -> dict:
    path = Path(path)
    text = clean_text(pdf_to_text(path))                       # Docling
    title, turns = parse_transcript(text, default_title=path.stem)
    mid = meeting_id or slug(path.stem)
    emb = get_embedder(cfg)

    units = build_units(turns, cfg)
    unit_vecs, _ = emb.encode([" ".join(turns[i].text for i in u) for u in units])
    chunks = build_chunks(turns, units, unit_vecs, mid, title, cfg)
    dense, sparse = emb.encode([c["embed_text"] for c in chunks])
    Corpus(cfg).add_meeting(mid, [asdict(t) for t in turns], chunks, dense, sparse)
    return {"meeting_id": mid, "title": title, "turns": len(turns),
            "units": len(units), "chunks": len(chunks)}
