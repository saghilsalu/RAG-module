"""LangGraph agent with a local Ollama model (qwen3.5:4b-q4_K_M): bounded multi-hop retrieval.

understand -> decompose -> retrieve -> evaluate
evaluate: insufficient & budget left -> retrieve again (follow-up queries built from evidence)
evaluate: sufficient or budget spent -> generate -> validate
validate: unfaithful -> generate again (bounded retries) ; else END
Hard limits: max_iterations, max_retrieval_calls, max_revisions, recursion_limit.
Every LLM call has a safe fallback, so one malformed 4B reply cannot crash the graph.
"""
from __future__ import annotations
import re
from typing import Literal, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from .config import Settings, SETTINGS
from .llm import get_llm, structured, text_of
from .store import Corpus, get_reranker


class Understanding(BaseModel):
    standalone_question: str = Field(description="Question rewritten to be self-contained")
    intent: Literal["lookup", "multi_hop", "summary"]


class SubQueries(BaseModel):
    queries: list[str] = Field(description="Independent search queries covering distinct facets")


class Sufficiency(BaseModel):
    sufficient: bool
    covered: str = Field(description="What the evidence already establishes")
    missing: list[str] = Field(description="Facts still needed")
    follow_up_queries: list[str] = Field(
        description="Targeted queries for the missing facts, using entities/numbers found in the evidence")


class Faithfulness(BaseModel):
    faithful: bool
    unsupported_claims: list[str]
    feedback: str


class RAGState(TypedDict, total=False):
    question: str
    meeting_id: Optional[str]
    standalone: str
    intent: str
    pending_queries: list[str]
    done_queries: list[str]
    evidence: dict            # "meeting#turn" -> {meeting_id, idx, score}
    iteration: int
    retrieval_calls: int
    sufficient: bool
    missing: list[str]
    answer: str
    revisions: int
    feedback: str
    faithful: bool
    citations: list[dict]
    trace: list[str]


class TranscriptRAG:
    def __init__(self, cfg: Settings = SETTINGS):
        self.cfg = cfg
        self.corpus = Corpus(cfg)
        self.reranker = get_reranker(cfg)
        self.llm = get_llm(cfg)
        self.graph = self._build()

    def _log(self, s: RAGState, msg: str) -> list[str]:
        return s.get("trace", []) + [msg]

    def _view(self, evidence: dict) -> tuple[str, dict]:
        """Chronological evidence text with stable aliases [E1].. and an alias->turn map."""
        items = sorted(evidence.values(), key=lambda e: e["score"], reverse=True)[: self.cfg.max_evidence_turns]
        items.sort(key=lambda e: (e["meeting_id"], e["idx"]))
        multi = len({e["meeting_id"] for e in items}) > 1
        alias, lines, prev = {}, [], None
        for n, e in enumerate(items, 1):
            t = self.corpus.turns[e["meeting_id"]][e["idx"]]
            if prev and (prev["meeting_id"] != e["meeting_id"] or prev["idx"] + 1 != e["idx"]):
                lines.append("---")
            tag = f"E{n}"
            alias[tag] = {"meeting_id": e["meeting_id"], **{k: t[k] for k in ("idx", "ts", "speaker", "org", "text")}}
            where = f"{e['meeting_id']} | " if multi else ""
            lines.append(f"[{tag} | {where}{t['ts']} | {t['speaker']} ({t['org']})] {t['text']}")
            prev = e
        return "\n".join(lines), alias

    # ------------------------------------------------------------ nodes
    def n_understand(self, s: RAGState) -> dict:
        u = structured(
            self.llm, Understanding,
            "You analyse questions about a meeting transcript.\n"
            "Intent rules: 'lookup' = one fact (e.g. 'Which CPU was proposed?'); "
            "'multi_hop' = needs linked evidence from different parts (e.g. 'How does the proposal fix the "
            "current bottleneck?'); 'summary' = broad overview.\n"
            f"Question: {s['question']}\n"
            "Rewrite it as a standalone question and classify its intent.",
            Understanding(standalone_question=s["question"], intent="multi_hop"))
        return {"standalone": u.standalone_question or s["question"], "intent": u.intent, "iteration": 0,
                "retrieval_calls": 0, "revisions": 0, "evidence": {}, "done_queries": [],
                "trace": self._log(s, f"understand: intent={u.intent}")}

    def n_decompose(self, s: RAGState) -> dict:
        if s["intent"] == "lookup":
            qs = [s["standalone"]]
        else:
            r = structured(
                self.llm, SubQueries,
                f"Break this question into at most {self.cfg.max_subqueries} short, independent search "
                "queries over a meeting transcript, each targeting a distinct facet (e.g. the problem, "
                "the proposed solution, the numbers). Keep exact technical terms.\n"
                f"Question: {s['standalone']}",
                SubQueries(queries=[s["standalone"]]))
            qs = [q for q in r.queries if q.strip()][: self.cfg.max_subqueries] or [s["standalone"]]
        return {"pending_queries": qs, "trace": self._log(s, f"decompose: {qs}")}

    def n_retrieve(self, s: RAGState) -> dict:
        cfg = self.cfg
        budget = max(cfg.max_retrieval_calls - s.get("retrieval_calls", 0), 0)
        queries = s["pending_queries"][:budget]
        evidence = dict(s.get("evidence", {}))
        for q in queries:
            hits = self.corpus.hybrid_search(q, s.get("meeting_id"))
            ranked = self.reranker.rerank(q, [self.corpus.chunks[cid] for cid, _ in hits], cfg.rerank_k)
            for chunk, score in ranked:
                for idx, core in self.corpus.expand(chunk, cfg.neighbor_window):
                    key = f"{chunk['meeting_id']}#{idx}"
                    val = float(score) if core else float(score) * 0.5
                    if key not in evidence or evidence[key]["score"] < val:
                        evidence[key] = {"meeting_id": chunk["meeting_id"], "idx": idx, "score": val}
        return {"evidence": evidence, "done_queries": s.get("done_queries", []) + queries,
                "retrieval_calls": s.get("retrieval_calls", 0) + len(queries),
                "iteration": s.get("iteration", 0) + 1, "pending_queries": [],
                "trace": self._log(s, f"retrieve: {queries} -> {len(evidence)} evidence turns")}

    def n_evaluate(self, s: RAGState) -> dict:
        text, _ = self._view(s["evidence"])
        done = s.get("done_queries", [])
        r = structured(
            self.llm, Sufficiency,
            "Decide whether the evidence is enough to fully answer the question.\n"
            "For cause/solution, comparison or before/after questions, BOTH sides must be covered, "
            "and each claimed link between them must be supported.\n"
            "If not sufficient, write follow-up queries that use specific entities, numbers or terms "
            f"DISCOVERED in the evidence. Do not repeat earlier queries: {done}\n\n"
            f"Question: {s['standalone']}\n\nEvidence:\n{text}",
            Sufficiency(sufficient=False, covered="", missing=[], follow_up_queries=[]))
        seen = {q.lower().strip() for q in done}
        new = []
        for q in r.follow_up_queries:
            if q.strip() and q.lower().strip() not in seen:
                new.append(q)
                seen.add(q.lower().strip())
        return {"sufficient": r.sufficient, "missing": r.missing,
                "pending_queries": new[: self.cfg.max_subqueries],
                "trace": self._log(s, f"evaluate: sufficient={r.sufficient} missing={r.missing} follow_ups={new}")}

    def route_after_eval(self, s: RAGState) -> str:
        cfg = self.cfg
        if (s["sufficient"] or s["iteration"] >= cfg.max_iterations
                or s["retrieval_calls"] >= cfg.max_retrieval_calls or not s["pending_queries"]):
            return "generate"
        return "retrieve"

    def n_generate(self, s: RAGState) -> dict:
        text, _ = self._view(s["evidence"])
        gaps = "" if s.get("sufficient") else (
            f"\nNOTE: retrieval budget ended with these gaps; state them plainly: {s.get('missing')}")
        fb = f"\nFix these problems from the previous draft: {s['feedback']}" if s.get("feedback") else ""
        r = self.llm.invoke(
            "Answer the question using ONLY the transcript evidence below.\n"
            "- Answer exactly what was asked. Do not add background or restate unrelated numbers.\n"
            "- Copy numbers and comparison words exactly ('at least', 'at or below'). Never paraphrase them.\n"
            "- A TARGET or REQUIREMENT is not a CURRENT value. If asked what something is today and the "
            "evidence only gives a target, say the current value is not stated, then mention the target.\n"
            "- Attribute statements to the speaker who said them. Do not expand acronyms.\n"
            "- Do not add words like 'exactly', 'eliminate' or 'immediately' unless the evidence says them.\n"
            "- Cite each claim with the single tag whose text contains it.\n"
            "- If the evidence lacks the answer, say 'Not stated in the transcript.'\n"
            f"{gaps}{fb}\n\nQuestion: {s['standalone']}\n\nEvidence:\n{text}")
        return {"answer": text_of(r), "trace": self._log(s, "generate")}

    def n_validate(self, s: RAGState) -> dict:
        text, alias = self._view(s["evidence"])
        groups = re.findall(r"\[([^\]]*)\]", s["answer"])
        cited = sorted({t for g in groups for t in re.findall(r"E\d+", g)}, key=lambda x: int(x[1:]))
        bad = [t for t in cited if t not in alias]
        ok = Faithfulness(faithful=True, unsupported_claims=[], feedback="")
        r = ok if not self.cfg.llm_faithfulness_check else structured(
            self.llm, Faithfulness,
            "Check the answer against the evidence. List every claim that is not directly "
            "supported by the turn it cites (or is uncited).\n\n"
            f"Question: {s['standalone']}\n\nAnswer:\n{s['answer']}\n\nEvidence:\n{text}", ok)
        faithful = r.faithful and not bad
        fb = "; ".join(r.unsupported_claims + [f"invalid citation {b}" for b in bad]) or r.feedback
        out = {"faithful": faithful, "feedback": "" if faithful else fb,
               "citations": [{"tag": t, **alias[t]} for t in cited if t in alias],
               "trace": self._log(s, f"validate: faithful={faithful}")}
        if not faithful:
            out["revisions"] = s.get("revisions", 0) + 1
        return out

    def route_after_validate(self, s: RAGState) -> str:
        return "generate" if (not s["faithful"] and s["revisions"] <= self.cfg.max_revisions) else END

    def _build(self):
        g = StateGraph(RAGState)
        for name, fn in [("understand", self.n_understand), ("decompose", self.n_decompose),
                         ("retrieve", self.n_retrieve), ("evaluate", self.n_evaluate),
                         ("generate", self.n_generate), ("validate", self.n_validate)]:
            g.add_node(name, fn)
        g.add_edge(START, "understand")
        g.add_edge("understand", "decompose")
        g.add_edge("decompose", "retrieve")
        g.add_edge("retrieve", "evaluate")
        g.add_conditional_edges("evaluate", self.route_after_eval, {"retrieve": "retrieve", "generate": "generate"})
        g.add_edge("generate", "validate")
        g.add_conditional_edges("validate", self.route_after_validate, {"generate": "generate", END: END})
        return g.compile()

    def ask(self, question: str, meeting_id: str | None = None) -> dict:
        out = self.graph.invoke({"question": question, "meeting_id": meeting_id},
                                config={"recursion_limit": 30})
        answer = out["answer"]
        if not out.get("faithful", True):
            answer += "\n\n(Warning: some statements could not be fully verified against the transcript.)"
        return {"answer": answer, "citations": out.get("citations", []), "faithful": out.get("faithful"),
                "iterations": out.get("iteration"), "retrieval_calls": out.get("retrieval_calls"),
                "trace": out.get("trace", [])}
