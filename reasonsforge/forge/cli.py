"""Forge CLI subcommands for reasonsforge."""

import importlib
import sys


def _lazy(module_name, func_name):
    mod = importlib.import_module(f".{module_name}", package="reasonsforge.forge")
    return getattr(mod, func_name)


def _add_common_pipeline_args(p):
    """Add args shared by all forge type commands."""
    p.add_argument("--model", default="claude", help="LLM model to use")
    p.add_argument("--rounds", type=int, default=3,
                   help="Convergence loop cycles")
    p.add_argument("--max-derive-rounds", type=int, default=10,
                   help="Max derive rounds per cycle")
    p.add_argument("--timeout", type=int, default=600, help="LLM timeout (s)")
    p.add_argument("--output", default="reasons.db",
                   help="Output database path")
    p.add_argument("--no-auto-accept", action="store_true",
                   help="Pause after proposing beliefs for manual review")
    p.add_argument("--resume", action="store_true",
                   help="Resume a previously interrupted pipeline")
    p.add_argument("--parallel", type=int, default=1,
                   help="Parallel LLM calls")


def register_forge_type_commands(parent_subparsers):
    """Register top-level forge type commands (code, product, project, paper, document).

    Returns a dict mapping command name → handler function.
    """

    # document — general document/PDF ingestion (the base forge)
    p = parent_subparsers.add_parser(
        "document",
        help="Build beliefs from documents (PDFs, markdown, code files)")
    p.add_argument("--sources-dir", default="sources",
                   help="Directory containing source documents")
    p.add_argument("--pdf", action="append",
                   help="PDF files to ingest (repeatable)")
    p.add_argument("--domain", help="Domain description for derive context")
    p.add_argument("--recursive", "-r", action="store_true",
                   help="Recurse into subdirectories")
    p.add_argument("--namespace", default=None)
    _add_common_pipeline_args(p)

    # code — codebase analysis
    p = parent_subparsers.add_parser(
        "code",
        help="Analyze a codebase and extract architectural beliefs")
    p.add_argument("--repo", default=".", help="Path to git repository")
    p.add_argument("--domain", help="Domain description")
    p.add_argument("--since", help="Analyze commits since this date or SHA")
    _add_common_pipeline_args(p)

    # product — product data from issue trackers
    p = parent_subparsers.add_parser(
        "product",
        help="Analyze product data from issue trackers")
    p.add_argument("--github", metavar="OWNER/REPO",
                   help="GitHub repository")
    p.add_argument("--gitlab", metavar="OWNER/REPO",
                   help="GitLab repository")
    p.add_argument("--jira", metavar="PROJECT_KEY",
                   help="Jira project key")
    p.add_argument("--jira-url", help="Jira base URL")
    p.add_argument("--domain", help="Domain description")
    p.add_argument("--since", help="Analyze issues since this date")
    _add_common_pipeline_args(p)

    # project — project management from issue trackers
    p = parent_subparsers.add_parser(
        "project",
        help="Analyze project state from issue trackers")
    p.add_argument("--github", metavar="OWNER/REPO",
                   help="GitHub repository")
    p.add_argument("--gitlab", metavar="OWNER/REPO",
                   help="GitLab repository")
    p.add_argument("--jira", metavar="PROJECT_KEY",
                   help="Jira project key")
    p.add_argument("--jira-url", help="Jira base URL")
    p.add_argument("--domain", help="Domain description")
    p.add_argument("--since", help="Analyze issues since this date")
    _add_common_pipeline_args(p)

    # paper — academic papers
    p = parent_subparsers.add_parser(
        "paper",
        help="Process academic papers and extract claims")
    p.add_argument("--arxiv", metavar="ID",
                   help="arXiv paper ID (e.g., 2301.12345)")
    p.add_argument("--pdf", action="append",
                   help="PDF files to process (repeatable)")
    p.add_argument("--domain", help="Domain description")
    _add_common_pipeline_args(p)

    # run — sandbox wrapper (Phase 3)
    p = parent_subparsers.add_parser(
        "run",
        help="Run a forge in a sandbox environment")
    p.add_argument("--sandbox", default="none",
                   choices=["none", "container", "vm", "lightweight"],
                   help="Sandbox tier")
    p.add_argument("forge_type",
                   choices=["document", "code", "product", "project", "paper"],
                   help="Forge type to run")
    p.add_argument("forge_args", nargs="*",
                   help="Arguments passed to the forge")

    return {
        "document": _cmd_document,
        "code": _cmd_code,
        "product": _cmd_product,
        "project": _cmd_project,
        "paper": _cmd_paper,
        "run": _cmd_run,
    }


def _cmd_document(args):
    """Run the document forge pipeline."""
    from . import REASONS_DB
    import reasonsforge.forge as forge

    if args.output != REASONS_DB:
        forge.REASONS_DB = args.output

    from .pipeline import cmd_pipeline
    args.sources_dir = getattr(args, "sources_dir", "sources")
    args.index_db = "rag_fts.db"
    cmd_pipeline(args)


def _cmd_code(args):
    """Run the code forge pipeline."""
    print(f"Code forge: analyzing {args.repo}", file=sys.stderr)
    print("Not yet implemented — coming in a future release.", file=sys.stderr)
    print("Use 'reasonsforge document' with source files for now.",
          file=sys.stderr)
    sys.exit(1)


def _cmd_product(args):
    """Run the product forge pipeline."""
    source = args.github or args.gitlab or args.jira
    if not source:
        print("Error: specify --github, --gitlab, or --jira", file=sys.stderr)
        sys.exit(1)
    print(f"Product forge: analyzing {source}", file=sys.stderr)
    print("Not yet implemented — coming in a future release.", file=sys.stderr)
    sys.exit(1)


def _cmd_project(args):
    """Run the project forge pipeline."""
    source = args.github or args.gitlab or args.jira
    if not source:
        print("Error: specify --github, --gitlab, or --jira", file=sys.stderr)
        sys.exit(1)
    print(f"Project forge: analyzing {source}", file=sys.stderr)
    print("Not yet implemented — coming in a future release.", file=sys.stderr)
    sys.exit(1)


def _cmd_paper(args):
    """Run the paper forge pipeline."""
    source = args.arxiv or (args.pdf and args.pdf[0])
    if not source:
        print("Error: specify --arxiv or --pdf", file=sys.stderr)
        sys.exit(1)
    print(f"Paper forge: processing {source}", file=sys.stderr)
    print("Not yet implemented — coming in a future release.", file=sys.stderr)
    sys.exit(1)


def _cmd_run(args):
    """Run a forge inside a sandbox."""
    if args.sandbox == "none":
        print("Error: --sandbox=none is the default; use the forge "
              "command directly instead of 'run'", file=sys.stderr)
        sys.exit(1)
    print(f"Sandbox ({args.sandbox}): {args.forge_type} "
          f"{' '.join(args.forge_args)}", file=sys.stderr)
    print("Not yet implemented — coming in Phase 3.", file=sys.stderr)
    sys.exit(1)


def register_forge_commands(parent_subparsers):
    """Register 'forge' subcommand group for individual pipeline steps."""
    forge_parser = parent_subparsers.add_parser(
        "forge", help="Individual forge pipeline steps"
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
