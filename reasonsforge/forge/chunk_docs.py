"""Chunk large documents into entry-sized pieces."""

import re
import sys
from pathlib import Path


def chunk_markdown(text, max_chars=25000):
    """Split markdown by heading boundaries, merging small sections."""
    parts = re.split(r"(?=^#{1,2} )", text, flags=re.MULTILINE)
    parts = [p for p in parts if p.strip()]

    if len(parts) <= 1:
        return chunk_fixed(text, max_chars)

    chunks = []
    current = ""
    for part in parts:
        if len(current) + len(part) > max_chars and current:
            chunks.append(current)
            current = part
        else:
            current += part
    if current:
        chunks.append(current)

    return chunks


def chunk_python(text, max_chars=25000):
    """Split Python by top-level class/def boundaries, keeping imports."""
    lines = text.split("\n")

    preamble_end = 0
    for i, line in enumerate(lines):
        if re.match(r"^(class |def )", line) or (line.startswith("@") and i + 1 < len(lines) and re.match(r"^(class |def |@)", lines[i + 1])):
            preamble_end = i
            break
    else:
        return chunk_fixed(text, max_chars)

    preamble = "\n".join(lines[:preamble_end]).rstrip() + "\n\n"

    boundaries = []
    for i, line in enumerate(lines[preamble_end:], start=preamble_end):
        if re.match(r"^(class |def )", line):
            # Include preceding decorator lines
            start = i
            while start > preamble_end and lines[start - 1].startswith("@"):
                start -= 1
            if not boundaries or boundaries[-1] != start:
                boundaries.append(start)

    if not boundaries:
        return chunk_fixed(text, max_chars)

    sections = []
    for j, start in enumerate(boundaries):
        end = boundaries[j + 1] if j + 1 < len(boundaries) else len(lines)
        section = "\n".join(lines[start:end])
        sections.append(section)

    chunks = []
    current = preamble
    for section in sections:
        if len(current) + len(section) > max_chars and current.strip() != preamble.strip():
            chunks.append(current)
            current = preamble + section
        else:
            current += section + "\n"
    if current.strip():
        chunks.append(current)

    return chunks


def chunk_fixed(text, max_chars=25000, overlap=500):
    """Split text into fixed-size windows with overlap."""
    if len(text) <= max_chars:
        return [text]

    overlap = min(overlap, max_chars // 4)
    step = max(max_chars - overlap, 1)

    chunks = []
    start = 0
    while start < len(text):
        end = start + max_chars
        chunks.append(text[start:end])
        start += step
    return chunks


def _strip_frontmatter(text):
    """Strip YAML frontmatter and return (frontmatter_dict, content)."""
    meta = {}
    content = text
    if text.startswith("---"):
        end = text.find("---", 3)
        if end != -1:
            for line in text[3:end].splitlines():
                if ":" in line:
                    key, _, val = line.partition(":")
                    meta[key.strip()] = val.strip()
            content = text[end + 3:].strip()
    return meta, content


def cmd_chunk_docs(args):
    """Chunk large documents into entry-sized pieces."""
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Source directory not found: {input_dir}")
        sys.exit(1)

    threshold = args.threshold

    glob = input_dir.rglob if getattr(args, "recursive", False) else input_dir.glob
    sources = sorted(
        [*glob("*.md"), *glob("*.py")],
        key=lambda p: p.name,
    )
    if not sources:
        print(f"No .md or .py files in {input_dir}")
        return

    manifest = Path(".chunked-docs")
    done = set()
    if manifest.exists():
        done = set(manifest.read_text().strip().split("\n"))

    total_chunked = 0
    total_skipped = 0

    for source_path in sources:
        if str(source_path) in done:
            total_skipped += 1
            continue

        raw = source_path.read_text()
        meta, content = _strip_frontmatter(raw)

        if len(content) <= threshold:
            continue

        print(f"Chunking: {source_path.name} ({len(content)} chars)")

        max_chars = threshold
        if source_path.suffix == ".py":
            chunks = chunk_python(content, max_chars=max_chars)
        elif source_path.suffix == ".md":
            chunks = chunk_markdown(content, max_chars=max_chars)
        else:
            chunks = chunk_fixed(content, max_chars=max_chars)

        if args.dry_run:
            for i, chunk in enumerate(chunks, 1):
                print(f"  chunk {i}: {len(chunk)} chars")
            continue

        chunk_dir = Path("sources") / "chunks"
        chunk_dir.mkdir(parents=True, exist_ok=True)

        source_url = meta.get("source_url") or meta.get("source", "")
        if source_url and not source_url.startswith(("http://", "https://")):
            source_url = ""
        source_id = meta.get("source_id", "")

        for i, chunk in enumerate(chunks, 1):
            chunk_name = f"{source_path.stem}-chunk-{i}.md"
            chunk_path = chunk_dir / chunk_name

            fm_lines = [f"source: {source_path}"]
            if source_url:
                fm_lines.append(f"source_url: {source_url}")
            if source_id:
                fm_lines.append(f"source_id: {source_id}")
            fm_lines.append(f"chunk: {i}/{len(chunks)}")
            frontmatter = "---\n" + "\n".join(fm_lines) + "\n---\n\n"

            chunk_path.write_text(frontmatter + chunk + "\n")
            print(f"  -> {chunk_path}")

        total_chunked += 1

        with manifest.open("a") as f:
            f.write(f"{source_path}\n")
        done.add(str(source_path))

    print(f"\nChunked {total_chunked} files ({total_skipped} already done)")
