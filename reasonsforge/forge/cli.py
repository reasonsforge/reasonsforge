"""Forge CLI subcommands for reasonsforge."""

import importlib


def _lazy(module_name, func_name):
    mod = importlib.import_module(f".{module_name}", package="reasonsforge.forge")
    return getattr(mod, func_name)


def register_forge_commands(parent_subparsers):
    """Register 'forge' subcommand group on the main CLI parser."""
    forge_parser = parent_subparsers.add_parser(
        "forge", help="Build belief networks from source material"
    )
    sub = forge_parser.add_subparsers(dest="forge_command")

    # init
    p = sub.add_parser("init", help="Initialize a forge project")
    p.add_argument("name", help="Project name")
    p.add_argument("--domain", help="One-line domain description")

    # chunk-pdf
    p = sub.add_parser("chunk-pdf", help="Chunk a PDF into section entries")
    p.add_argument("pdf", help="Path to PDF file")
    p.add_argument("--prefix", help="Entry filename prefix")
    p.add_argument("--source-label", help="Citation label")
    p.add_argument("--dry-run", action="store_true")

    # chunk-docs
    p = sub.add_parser("chunk-docs", help="Chunk large documents")
    p.add_argument("--input-dir", default="sources")
    p.add_argument("--threshold", type=int, default=30000)
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--dry-run", action="store_true")

    # summarize
    p = sub.add_parser("summarize", help="Generate entries from source documents")
    p.add_argument("--input-dir", default="sources")
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--limit", type=int)
    p.add_argument("--model", default="claude")

    # propose-beliefs
    p = sub.add_parser("propose-beliefs",
                       help="Extract candidate beliefs from entries")
    p.add_argument("--input-dir", default="entries")
    p.add_argument("--output", default="proposed-beliefs.md")
    p.add_argument("--model", default="claude")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=5)
    p.add_argument("--entry", action="append")
    p.add_argument("--all", action="store_true")

    # accept-beliefs
    p = sub.add_parser("accept-beliefs",
                       help="Import accepted beliefs from proposals")
    p.add_argument("--file", default="proposed-beliefs.md")

    # pipeline
    p = sub.add_parser("pipeline",
                       help="Run end-to-end belief construction pipeline")
    p.add_argument("--pdf", action="append")
    p.add_argument("--sources-dir", default="sources")
    p.add_argument("--model", default="claude")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-derive-rounds", type=int, default=10)
    p.add_argument("--no-auto-accept", action="store_true")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--domain", help="Domain description for derive context")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--namespace", default=None)

    # derive-review-repair
    p = sub.add_parser("derive-review-repair",
                       help="Run derive/review/repair convergence loop")
    p.add_argument("--model", default="claude")
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--max-derive-rounds", type=int, default=10)
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--domain", help="Domain description for derive context")
    p.add_argument("--namespace", default=None)

    # index-sources
    p = sub.add_parser("index-sources", help="Build FTS5 search index")
    p.add_argument("--input-dir", default="sources")
    p.add_argument("--recursive", "-r", action="store_true")
    p.add_argument("--db", default="rag_fts.db")
    p.add_argument("--type", default="source",
                   choices=["source", "summary", "chunked-summary"])
    p.add_argument("--chunk-size", type=int, default=2000)
    p.add_argument("--rebuild", action="store_true")

    # status
    sub.add_parser("status", help="Show forge pipeline progress")

    return {
        "init": lambda a: _lazy("init_cmd", "cmd_init")(a),
        "chunk-pdf": lambda a: _lazy("chunk_pdf", "cmd_chunk_pdf")(a),
        "chunk-docs": lambda a: _lazy("chunk_docs", "cmd_chunk_docs")(a),
        "summarize": lambda a: _lazy("summarize", "cmd_summarize")(a),
        "propose-beliefs": lambda a: _lazy("propose", "cmd_propose_beliefs")(a),
        "accept-beliefs": lambda a: _lazy("propose", "cmd_accept_beliefs")(a),
        "pipeline": lambda a: _lazy("pipeline", "cmd_pipeline")(a),
        "derive-review-repair": lambda a: _lazy("pipeline",
                                                 "cmd_derive_review_repair")(a),
        "index-sources": lambda a: _lazy("index_sources",
                                         "cmd_index_sources")(a),
        "status": lambda a: _lazy("init_cmd", "cmd_status")(a),
    }
