"""Docling PDF extraction + speaker-turn parsing."""
from __future__ import annotations
import re
from dataclasses import dataclass
from pathlib import Path

TURN_RE = re.compile(
    r"\[(\d{1,2}:\d{2}:\d{2})\]\s*([^\[\]():]{1,80}?)\s*\(([^()]{1,60})\)\s*:"
)
TITLE_RE = re.compile(r"Meeting Title:\s*(.+?)(?:\s+Duration:|\n|$)", re.I)


@dataclass
class Turn:
    idx: int
    ts: str
    seconds: int
    speaker: str
    org: str
    text: str

    @property
    def is_question(self) -> bool:
        return "?" in self.text


def pdf_to_text(path: str | Path) -> str:
    from docling.document_converter import DocumentConverter

    result = DocumentConverter().convert(str(path))
    return result.document.export_to_markdown()


def clean_text(raw: str) -> str:
    t = re.sub(r"<!--.*?-->", " ", raw, flags=re.S)          # docling image placeholders
    t = re.sub(r"\\([\[\]()_*#])", r"\1", t)                  # markdown escapes: \[ \] ...
    t = re.sub(r"^\s*#{1,6}\s*", "", t, flags=re.M)           # headings
    t = re.sub(r"\*{1,2}|`", "", t)                           # bold/code markers
    t = re.sub(r"^\s*(?:page\s+)?\d+(?:\s*(?:/|of)\s*\d+)?\s*$", " ", t, flags=re.M | re.I)  # page numbers
    return re.sub(r"\s+", " ", t).strip()


def _seconds(ts: str) -> int:
    h, m, s = (int(x) for x in ts.split(":"))
    return h * 3600 + m * 60 + s


def parse_transcript(text: str, default_title: str = "Untitled meeting") -> tuple[str, list[Turn]]:
    matches = list(TURN_RE.finditer(text))
    if not matches:
        raise ValueError("No '[hh:mm:ss] Speaker (Org):' turns found - is this a transcript?")
    m = TITLE_RE.search(text[: matches[0].start()])
    title = m.group(1).strip() if m else default_title
    turns: list[Turn] = []
    for i, mt in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[mt.end():end].strip()
        if not body:
            continue
        turns.append(Turn(len(turns), mt.group(1), _seconds(mt.group(1)),
                          mt.group(2).strip(), mt.group(3).strip(), body))
    # re-index after dropping empties
    for i, t in enumerate(turns):
        t.idx = i
    return title, turns
