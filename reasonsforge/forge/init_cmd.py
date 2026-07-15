"""Init and status commands for forge projects."""

import json
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path

from reasonsforge.api import init_db, get_status as reasons_status

from . import PROJECT_DIR, REASONS_DB


def cmd_init(args):
    """Bootstrap a new forge project directory."""
    name = args.name
    domain = getattr(args, "domain", None) or name
    cwd = Path.cwd()

    if not shutil.which("git"):
        print("Error: git not found")
        sys.exit(1)

    if not getattr(args, "no_git", False) and not (cwd / ".git").exists():
        subprocess.run(["git", "init"], check=True)
        print("Initialized git repo")

    for d in ["entries", "sources"]:
        (cwd / d).mkdir(exist_ok=True)

    forge_dir = cwd / PROJECT_DIR
    forge_dir.mkdir(exist_ok=True)

    if not (cwd / REASONS_DB).exists():
        init_db(db_path=REASONS_DB)
        print("Initialized reasons database")
    else:
        print(f"{REASONS_DB} already exists, skipping reasons init")

    gitignore = cwd / ".gitignore"
    if not gitignore.exists():
        gitignore.write_text(
            "reasons.db\n"
            "rag_fts.db\n"
        )
        print("Created .gitignore")

    config_path = forge_dir / "config.json"
    if not config_path.exists():
        config = {
            "name": name,
            "domain": domain,
            "created": date.today().isoformat(),
        }
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        print(f"Created {config_path}")

    print(f"\nForge project initialized: {name}")
    print(f"Next: add documents to sources/ and run reasonsforge forge summarize")


def cmd_status(args):
    """Show forge pipeline progress."""
    cwd = Path.cwd()

    sources_dir = cwd / "sources"
    source_count = len(list(sources_dir.glob("*.md"))) if sources_dir.exists() else 0

    entries_dir = cwd / "entries"
    entry_count = 0
    if entries_dir.exists():
        entry_count = len(list(entries_dir.rglob("*.md")))

    belief_count = 0
    nogood_count = 0
    reasons_db = cwd / REASONS_DB
    if reasons_db.exists():
        try:
            from reasonsforge.api import export_network
            status = reasons_status(db_path=REASONS_DB)
            belief_count = status["in_count"]
            network = export_network(db_path=REASONS_DB)
            nogood_count = len(network.get("nogoods", []))
        except Exception:
            pass

    proposed_file = cwd / "proposed-beliefs.md"
    proposed = 0
    accepted = 0
    if proposed_file.exists():
        import re
        text = proposed_file.read_text()
        proposed = len(re.findall(r"^### \[", text, re.MULTILINE))
        accepted = len(re.findall(r"^### \[ACCEPT\]", text, re.MULTILINE))

    config_path = cwd / PROJECT_DIR / "config.json"
    domain = "unknown"
    if config_path.exists():
        try:
            config = json.loads(config_path.read_text())
            domain = config.get("domain", config.get("name", "unknown"))
        except (json.JSONDecodeError, ValueError):
            pass

    print(f"=== Forge Status: {domain} ===")
    print(f"Sources:     {source_count} documents")
    print(f"Entries:     {entry_count} entries")
    print(f"Beliefs:     {belief_count} IN")
    print(f"Nogoods:     {nogood_count} recorded")
    if proposed:
        print(f"Proposed:    {proposed} candidates ({accepted} accepted)")
