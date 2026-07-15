"""Summarize source documents into entries using an LLM."""

import asyncio
import sys
from datetime import date
from pathlib import Path

from .llm import check_model_available, invoke
from .prompts import SUMMARIZE, SUMMARIZE_CODE


def _prepare_source(source_path):
    """Read source file, strip frontmatter, truncate if needed.

    Returns (source_url, source_id, prompt) or None if skipped.
    """
    content = source_path.read_text()

    source_url = None
    source_id = None
    if content.startswith("---"):
        end = content.find("---", 3)
        if end != -1:
            frontmatter = content[3:end]
            for line in frontmatter.splitlines():
                if line.startswith("source_url:"):
                    source_url = line.split(":", 1)[1].strip()
                elif line.startswith("source:"):
                    val = line.split(":", 1)[1].strip()
                    if not source_url and val.startswith(("http://", "https://")):
                        source_url = val
                elif line.startswith("source_id:"):
                    source_id = line.split(":", 1)[1].strip()
            content = content[end + 3:].strip()

    if not content.strip():
        return None

    if len(content) > 30000:
        original_len = len(content)
        content = content[:30000] + "\n\n[Truncated — original was longer]"
        if source_path.suffix == ".pdf":
            print(f"  WARN: truncated from {original_len} to 30000 chars. "
                  f"Consider: reasonsforge forge chunk-pdf {source_path}")
        else:
            print(f"  WARN: truncated from {original_len} to 30000 chars. "
                  f"Consider: reasonsforge forge chunk-docs")

    template = SUMMARIZE_CODE if source_path.suffix == ".py" else SUMMARIZE
    prompt = template.format(content=content)

    return source_url, source_id, prompt


def _write_entry(source_path, summary, source_url, source_id):
    """Write entry file with provenance frontmatter."""
    topic = source_path.stem
    today = date.today()
    entry_dir = Path("entries") / str(today.year) / f"{today.month:02d}" / f"{today.day:02d}"
    entry_dir.mkdir(parents=True, exist_ok=True)
    entry_path = entry_dir / f"{topic}.md"

    fm_lines = [f"source: {source_path}"]
    if source_url:
        fm_lines.append(f"source_url: {source_url}")
    if source_id:
        fm_lines.append(f"source_id: {source_id}")
    frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n\n"

    entry_path.write_text(frontmatter + summary + "\n")
    return entry_path


async def _summarize_one(source_path, model, semaphore, manifest, done):
    """Summarize a single source file under concurrency limit."""
    async with semaphore:
        prepared = _prepare_source(source_path)
        if prepared is None:
            print(f"  SKIP (empty): {source_path.name}")
            return False

        source_url, source_id, prompt = prepared
        print(f"Summarizing: {source_path.name}")

        try:
            summary = await invoke(prompt, model=model)
        except Exception as e:
            print(f"  ERROR ({source_path.name}): {e}")
            return False

        entry_path = _write_entry(source_path, summary, source_url, source_id)
        print(f"  -> Created {entry_path}")

        with manifest.open("a") as f:
            f.write(f"{source_path}\n")
        done.add(str(source_path))
        return True


def cmd_summarize(args):
    """Generate entries from source documents."""
    from .caffeinate import hold as _caffeinate
    _caffeinate()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Source directory not found: {input_dir}")
        print("Add documents to sources/ first")
        sys.exit(1)

    if not check_model_available(args.model):
        print(f"Model not available: {args.model}")
        print("Install claude CLI or specify --model")
        sys.exit(1)

    glob = input_dir.rglob if getattr(args, "recursive", False) else input_dir.glob
    sources = sorted(
        [*glob("*.md"), *glob("*.py"), *glob("*.txt")],
        key=lambda p: p.name,
    )
    if not sources:
        print(f"No .md, .py, or .txt files in {input_dir}")
        return

    if args.limit:
        sources = sources[:args.limit]

    manifest = Path(".summarized")
    done = set()
    if manifest.exists():
        done = set(manifest.read_text().strip().split("\n"))

    to_process = [s for s in sources if str(s) not in done]
    skipped = len(sources) - len(to_process)

    if not to_process:
        print(f"\nSummarized 0 sources ({skipped} already done)")
        return

    parallel = max(1, getattr(args, "parallel", 1))
    semaphore = asyncio.Semaphore(parallel)

    async def run_all():
        tasks = [
            _summarize_one(s, args.model, semaphore, manifest, done)
            for s in to_process
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(run_all())
    processed = sum(1 for r in results if r)

    print(f"\nSummarized {processed} sources ({skipped} already done)")
