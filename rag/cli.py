import argparse
import json

from .config import SETTINGS


def main():
    p = argparse.ArgumentParser(prog="rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ingest")
    i.add_argument("pdf")
    i.add_argument("--meeting-id")
    s = sub.add_parser("search")
    s.add_argument("query")
    s.add_argument("--meeting")
    s.add_argument("-k", type=int, default=SETTINGS.rerank_k)
    s.add_argument("--hops", type=int, default=1, help="0 = single pass, 1 = one feedback hop")
    s.add_argument("--json", action="store_true")
    a = sub.add_parser("ask", help="LLM agent (Ollama) with multi-hop retrieval + citations")
    a.add_argument("question")
    a.add_argument("--meeting")
    a.add_argument("--trace", action="store_true")
    args = p.parse_args()

    if args.cmd == "ingest":
        from .ingest import ingest_pdf
        print(json.dumps(ingest_pdf(args.pdf, SETTINGS, args.meeting_id), indent=2))
    elif args.cmd == "ask":
        from .graph import TranscriptRAG
        r = TranscriptRAG(SETTINGS).ask(args.question, args.meeting)
        print(r["answer"], "\n")
        for c in r["citations"]:
            print(f"  [{c['tag']}] {c['ts']} {c['speaker']} ({c['org']}): {c['text'][:110]}...")
        if args.trace:
            print("\n".join(["", *r["trace"]]))
    else:
        from .retriever import ChunkRetriever
        res = ChunkRetriever(SETTINGS).search(args.query, args.meeting, args.k, args.hops)
        if args.json:
            print(json.dumps(res, indent=2))
        else:
            for r in res:
                print(f"\n=== {r['chunk_id']}  {r['start_ts']}-{r['end_ts']}  score={r['score']}")
                print(r["expanded_text"])


if __name__ == "__main__":
    main()
