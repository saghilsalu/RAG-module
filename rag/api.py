import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel

from .config import SETTINGS
from .ingest import ingest_pdf

app = FastAPI(title="Transcript retrieval")
_r = None


def retriever():
    global _r
    if _r is None:
        from .retriever import ChunkRetriever
        _r = ChunkRetriever(SETTINGS)
    return _r


class Query(BaseModel):
    query: str
    meeting_id: str | None = None
    top_k: int = 8
    hops: int = 1
    expand: bool = True


@app.post("/ingest")
async def ingest(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / file.filename
        with p.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        info = ingest_pdf(p, SETTINGS)
    retriever().corpus.reload()
    return info


@app.post("/search")
def search(q: Query):
    return retriever().search(q.query, q.meeting_id, q.top_k, q.hops, q.expand)


class Ask(BaseModel):
    question: str
    meeting_id: str | None = None


_agent = None


@app.post("/ask")
def ask(body: Ask):
    global _agent
    if _agent is None:
        from .graph import TranscriptRAG
        _agent = TranscriptRAG(SETTINGS)
    return _agent.ask(body.question, body.meeting_id)
