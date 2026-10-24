"""Command line: sales-agent research | ingest | prefs | serve"""

from __future__ import annotations

import argparse
import json
import os
import sys

from .config import Settings


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sales-agent", description="Sales Research Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    research = sub.add_parser("research", help="Research a company and print a sales briefing")
    research.add_argument("question", nargs="+", help="e.g. Research Infosys for Monday's meeting")
    research.add_argument("--json", action="store_true", help="Print JSON instead of text")

    sub.add_parser("ingest", help="Build the search index from data/docs")

    prefs = sub.add_parser("prefs", help="See or change your saved preferences")
    prefs.add_argument("action", choices=["list", "add", "clear"])
    prefs.add_argument("text", nargs="*")

    serve = sub.add_parser("serve", help="Start the HTTP API")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8080)

    args = parser.parse_args(argv)
    settings = Settings.from_env()

    if args.command in ("research", "ingest", "serve") and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY is not set. Copy .env.example to .env and add your key.", file=sys.stderr)
        return 2

    if args.command == "research":
        return _research(settings, " ".join(args.question), args.json)
    if args.command == "ingest":
        return _ingest(settings)
    if args.command == "prefs":
        return _prefs(settings, args.action, " ".join(args.text))
    return _serve(args.host, args.port)


def _research(settings: Settings, question: str, as_json: bool) -> int:
    from .factory import build_agent

    result = build_agent(settings).run(question)
    if result.refused:
        print(f"Not done: {result.refusal_reason}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(result.briefing.model_dump(), indent=2, ensure_ascii=False))
    else:
        print(result.briefing.to_text())
    t = result.trace
    note = " (stopped at step limit)" if result.stopped_early else ""
    print(
        f"\n[{t['steps']} steps{note}, {t['tool_calls']} tool calls, {t['seconds']}s, "
        f"~${t['cost_usd']}]  trace: {t['trace_file']}",
        file=sys.stderr,
    )
    return 0


def _ingest(settings: Settings) -> int:
    from .llm import OpenAILLM
    from .rag.ingest import ingest
    from .rag.store import VectorStore

    llm = OpenAILLM(settings.model, settings.embedding_model)
    files, chunks = ingest(settings.docs_dir, VectorStore(settings.index_path), llm, settings.embedding_model)
    print(f"Indexed {files} files into {chunks} chunks -> {settings.index_path}")
    return 0


def _prefs(settings: Settings, action: str, text: str) -> int:
    from .memory import Memory

    memory = Memory(settings.memory_path)
    if action == "add":
        if not text:
            print("Give the preference text, e.g.: sales-agent prefs add Keep talking points short", file=sys.stderr)
            return 2
        memory.add_preference(text)
    elif action == "clear":
        memory.clear_preferences()
    for i, pref in enumerate(memory.preferences(), 1):
        print(f"{i}. {pref}")
    return 0


def _serve(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("sales_agent.api:app", host=host, port=port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
