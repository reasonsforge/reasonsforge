"""Propose and accept beliefs from entries."""

import asyncio
import hashlib
import json
import re
import sys
from datetime import date
from pathlib import Path

from reasonsforge.api import add_node, list_nodes

from .llm import check_model_available, extract_json, invoke, RETRY_JSON
from .prompts import PROPOSE_BELIEFS

from . import PROJECT_DIR, REASONS_DB


def _has_embeddings() -> bool:
    """Check if fastembed is available."""
    try:
        import numpy  # noqa: F401
        from fastembed import TextEmbedding  # noqa: F401
        return True
    except ImportError:
        return False


_embed_model = None


def _get_embed_model():
    """Lazy-load the fastembed model."""
    global _embed_model
    if _embed_model is None:
        from fastembed import TextEmbedding
        _embed_model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5")
    return _embed_model


def _load_existing_beliefs(db_path: str = REASONS_DB) -> list[dict]:
    """Load existing beliefs from the reasons database."""
    if not Path(db_path).exists():
        return []
    try:
        from reasonsforge.api import export_network
        network = export_network(db_path=db_path)
        beliefs = []
        for nid, ndata in network.get("nodes", {}).items():
            beliefs.append({
                "id": nid,
                "text": ndata.get("text", ""),
                "source": ndata.get("source", ""),
            })
        return beliefs
    except Exception:
        return []


def _load_processed(path: Path) -> dict[str, str]:
    """Load processed entries tracking {path: content_hash}."""
    if path.exists():
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_processed(path: Path, new_entries: list[Path], existing: dict[str, str]):
    """Record new entries as processed by content hash and write to disk."""
    for entry_path in new_entries:
        content = entry_path.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        existing[str(entry_path)] = content_hash
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2) + "\n")


def _filter_unprocessed(entries: list[Path], processed: dict[str, str]) -> list[Path]:
    """Return entries that are new or modified since last propose."""
    unprocessed = []
    for entry_path in entries:
        key = str(entry_path)
        if key not in processed:
            unprocessed.append(entry_path)
            continue
        content = entry_path.read_text()
        content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        if content_hash != processed[key]:
            unprocessed.append(entry_path)
    return unprocessed


# --- Embedding support ---


def _load_belief_vectors(cache_path: Path) -> dict[str, list[float]]:
    """Load cached belief vectors from JSON."""
    if cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


def _save_belief_vectors(cache_path: Path, vectors: dict[str, list[float]]):
    """Save belief vectors to JSON cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(vectors))


def _get_belief_embeddings(
    beliefs: list[dict], cache_path: Path,
) -> dict[str, list[float]]:
    """Get embeddings for all beliefs, using cache for known ones."""
    model = _get_embed_model()
    cached = _load_belief_vectors(cache_path)

    def _cache_key(belief):
        text_hash = hashlib.sha256(belief["text"].encode()).hexdigest()[:8]
        return f"{belief['id']}:{text_hash}"

    needed = []
    needed_keys = []
    result = {}
    for belief in beliefs:
        key = _cache_key(belief)
        if key in cached:
            result[belief["id"]] = cached[key]
        else:
            needed.append(belief)
            needed_keys.append(key)

    if needed:
        texts = [b["text"] for b in needed]
        vectors = list(model.embed(texts))
        for belief, key, vec in zip(needed, needed_keys, vectors):
            vec_list = vec.tolist()
            cached[key] = vec_list
            result[belief["id"]] = vec_list

        current_keys = {_cache_key(b) for b in beliefs}
        cached = {k: v for k, v in cached.items() if k in current_keys}
        _save_belief_vectors(cache_path, cached)

    return result


def _score_by_embedding(
    beliefs: list[dict],
    belief_vectors: dict[str, list[float]],
    batch_text: str,
    batch_entry_paths: list[str],
) -> list[tuple[float, dict]]:
    """Score beliefs by embedding similarity to batch content."""
    import numpy as np

    model = _get_embed_model()
    batch_summary = batch_text[:4000]
    query_vec = np.array(list(model.embed([batch_summary]))[0], dtype=np.float32)

    scored = []
    for belief in beliefs:
        vec = belief_vectors.get(belief["id"])
        if vec is None:
            scored.append((0.0, belief))
            continue
        belief_vec = np.array(vec, dtype=np.float32)
        dot = np.dot(query_vec, belief_vec)
        norm = np.linalg.norm(query_vec) * np.linalg.norm(belief_vec)
        similarity = float(dot / norm) if norm > 0 else 0.0
        if belief["source"] and any(belief["source"] in p or p in belief["source"]
                                     for p in batch_entry_paths):
            similarity += 1.0
        scored.append((similarity, belief))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _score_by_keywords(
    beliefs: list[dict],
    batch_text: str,
    batch_entry_paths: list[str],
) -> list[tuple[float, dict]]:
    """Score beliefs by keyword overlap (fallback when embeddings unavailable)."""
    batch_words = set(re.findall(r'[a-z]{3,}', batch_text.lower()))

    scored = []
    for belief in beliefs:
        score = 0.0
        if belief["source"] and any(belief["source"] in p or p in belief["source"]
                                     for p in batch_entry_paths):
            score += 1000
        belief_words = set(re.findall(r'[a-z]{3,}', belief["text"].lower()))
        belief_words |= set(belief["id"].replace("-", " ").lower().split())
        overlap = len(batch_words & belief_words)
        score += overlap
        scored.append((score, belief))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored


def _build_dedup_context(
    existing_beliefs: list[dict],
    batch_entry_paths: list[str],
    batch_text: str,
    max_detailed: int = 50,
    max_compact: int = 200,
    belief_vectors: dict[str, list[float]] | None = None,
) -> str:
    """Build per-batch dedup context: relevant beliefs with text, rest as compact IDs."""
    if not existing_beliefs:
        return ""

    if belief_vectors:
        scored = _score_by_embedding(
            existing_beliefs, belief_vectors, batch_text, batch_entry_paths,
        )
    else:
        scored = _score_by_keywords(
            existing_beliefs, batch_text, batch_entry_paths,
        )

    detailed = scored[:max_detailed]
    compact = scored[max_detailed:max_detailed + max_compact]

    parts = [
        "\n\n## Already Accepted Beliefs\n\n"
        "The following beliefs already exist. Do NOT propose beliefs with these IDs "
        "or that duplicate their meaning under different names.\n"
    ]

    if detailed:
        parts.append("\nRelevant existing beliefs:")
        for _, belief in detailed:
            parts.append(f"- `{belief['id']}`: {belief['text']}")

    if compact:
        compact_ids = ", ".join(b["id"] for _, b in compact)
        parts.append(f"\nOther existing IDs: {compact_ids}")

    return "\n".join(parts) + "\n"


# --- Commands ---


def auto_accept_proposals(filepath: str):
    """Rewrite all [ACCEPT/REJECT] and [REJECT] markers to [ACCEPT] in a proposals file."""
    path = Path(filepath)
    text = path.read_text()
    text = re.sub(r'\[ACCEPT/REJECT\]', '[ACCEPT]', text)
    text = re.sub(r'\[REJECT\]', '[ACCEPT]', text)
    path.write_text(text)


def cmd_propose_beliefs(args):
    """Extract candidate beliefs from entries for human review."""
    from .caffeinate import hold as _caffeinate
    _caffeinate()
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"Entries directory not found: {input_dir}")
        sys.exit(1)

    if not check_model_available(args.model):
        print(f"Model not available: {args.model}")
        sys.exit(1)

    # Collect entries
    if hasattr(args, 'entry') and args.entry:
        entries = [Path(p) for p in args.entry]
    else:
        entries = sorted(input_dir.rglob("*.md"))

    if not entries:
        print(f"No .md files found.")
        return

    # Filter out already-processed entries (unless --all or --entry)
    processed_path = Path(PROJECT_DIR) / "proposed-entries.json"
    processed = _load_processed(processed_path)
    process_all = getattr(args, 'all', False)
    has_entry_flag = hasattr(args, 'entry') and args.entry

    if not process_all and not has_entry_flag:
        total = len(entries)
        entries = _filter_unprocessed(entries, processed)
        skipped = total - len(entries)
        if skipped:
            print(f"Skipping {skipped} already-processed entries (use --all to reprocess)")
        if not entries:
            print("No new entries to process.")
            return

    # Load existing beliefs for dedup context
    existing_beliefs = _load_existing_beliefs()
    existing_ids = {b["id"] for b in existing_beliefs}

    if existing_ids:
        print(f"Found {len(existing_ids)} existing beliefs (will skip duplicates)")

    # Compute belief embeddings once (if fastembed available)
    belief_vectors = None
    if existing_beliefs and _has_embeddings():
        print("Computing belief embeddings for semantic dedup...")
        cache_path = Path(PROJECT_DIR) / "belief-vectors.json"
        belief_vectors = _get_belief_embeddings(existing_beliefs, cache_path)
        print(f"  {len(belief_vectors)} belief vectors ready")
    elif existing_beliefs:
        print("(install fastembed for semantic dedup: uv pip install 'expert-agent-builder[embeddings]')")

    print(f"Reading {len(entries)} entries...")

    # Batch entries — track paths per batch for relevance scoring
    batches = []
    batch_paths = []
    current_batch = []
    current_paths = []
    for entry_path in entries:
        content = entry_path.read_text()
        if len(content) > 10000:
            content = content[:10000] + "\n[Truncated]"
        source_url = ""
        if content.startswith("---"):
            end = content.find("---", 3)
            if end != -1:
                for line in content[3:end].splitlines():
                    if line.startswith("source_url:"):
                        source_url = line.split(":", 1)[1].strip()
                    elif line.startswith("source:"):
                        val = line.split(":", 1)[1].strip()
                        if not source_url and val.startswith(("http://", "https://")):
                            source_url = val
        header = f"--- FILE: {entry_path}"
        if source_url:
            header += f" | SOURCE_URL: {source_url}"
        header += " ---"
        current_batch.append(f"{header}\n{content}")
        current_paths.append(str(entry_path))
        if len(current_batch) >= args.batch_size:
            batches.append("\n\n".join(current_batch))
            batch_paths.append(current_paths)
            current_batch = []
            current_paths = []
    if current_batch:
        batches.append("\n\n".join(current_batch))
        batch_paths.append(current_paths)

    print(f"Processing {len(batches)} batches (batch size: {args.batch_size})...")

    source_desc = (", ".join(str(e) for e in entries)
                   if has_entry_flag
                   else f"{len(entries)} entries from {input_dir}/")
    output = Path(args.output)

    # Write header before first batch if starting a new file
    appended = output.exists() and output.stat().st_size > 0
    if not appended:
        with output.open("w") as f:
            f.write("# Proposed Beliefs\n\n")
            f.write("Review each entry: change `[REJECT]` to `[ACCEPT]` to keep, or vice versa.\n")
            f.write("Then run: `reasonsforge forge accept-beliefs`\n\n")
            f.write("---\n\n")
            f.write(f"**Generated:** {date.today().isoformat()}\n")
            f.write(f"**Source:** {source_desc}\n")
            f.write(f"**Model:** {args.model}\n\n")
    else:
        with output.open("a") as f:
            f.write(f"\n---\n\n")
            f.write(f"**Generated:** {date.today().isoformat()}\n")
            f.write(f"**Source:** {source_desc}\n")
            f.write(f"**Model:** {args.model}\n\n")

    total_skipped = 0
    write_lock = asyncio.Lock()

    async def _process_batch(i, batch_text, semaphore):
        """Process one batch and write results immediately."""
        nonlocal total_skipped
        async with semaphore:
            print(f"  Batch {i + 1}/{len(batches)}...")
            existing_context = _build_dedup_context(
                existing_beliefs, batch_paths[i], batch_text,
                belief_vectors=belief_vectors,
            )
            prompt = PROPOSE_BELIEFS.format(entries=batch_text) + existing_context
            try:
                result = await invoke(prompt, model=args.model, timeout=600)
            except Exception as e:
                print(f"  ERROR: {e}")
                return

            beliefs = extract_json(result)
            if not isinstance(beliefs, list):
                print("    WARN: response not valid JSON, retrying...", file=sys.stderr)
                try:
                    retry_response = await invoke(
                        prompt + "\n\n" + result + "\n\n" + RETRY_JSON,
                        model=args.model, timeout=600,
                    )
                    beliefs = extract_json(retry_response)
                except Exception:
                    pass
            if not isinstance(beliefs, list):
                print("    WARN: could not parse beliefs JSON, skipping batch", file=sys.stderr)
                return

            filtered = []
            skipped = 0
            for b in beliefs:
                bid = b.get("id", "")
                if bid in existing_ids:
                    skipped += 1
                    continue
                filtered.append(b)

        async with write_lock:
            total_skipped += skipped
            with output.open("a") as f:
                for b in filtered:
                    bid = b.get("id", "unknown")
                    claim = b.get("claim", "")
                    source = b.get("source", "")
                    source_url = b.get("source_url", "")
                    verdict = "[ACCEPT]" if b.get("accept", True) else "[REJECT]"
                    f.write(f"### {verdict} {bid}\n")
                    f.write(f"{claim}\n")
                    f.write(f"- Source: {source}\n")
                    f.write(f"- Source URL: {source_url or 'none'}\n\n")

            batch_entries = [Path(p) for p in batch_paths[i]]
            _save_processed(processed_path, batch_entries, processed)

    parallel = max(1, getattr(args, "parallel", 1))
    semaphore = asyncio.Semaphore(parallel)

    async def run_batches():
        tasks = [_process_batch(i, bt, semaphore) for i, bt in enumerate(batches)]
        await asyncio.gather(*tasks)

    asyncio.run(run_batches())

    if total_skipped:
        print(f"  Filtered {total_skipped} already-accepted beliefs")

    print(f"\n{'Appended to' if appended else 'Wrote'} {output}")

    print("Review the file, mark entries as [ACCEPT] or [REJECT], then run:")
    print("  reasonsforge forge accept-beliefs")


def cmd_accept_beliefs(args):
    """Import accepted beliefs from proposals file."""
    proposals_file = Path(args.file)
    if not proposals_file.exists():
        print(f"Proposals file not found: {proposals_file}")
        print("Run: reasonsforge forge propose-beliefs")
        sys.exit(1)

    text = proposals_file.read_text()

    # Parse accepted beliefs — tolerate both ### [ACCEPT] and ### ACCEPT
    pattern = re.compile(
        r"### \[?ACCEPT\]? (\S+)\n"
        r"(.+?)\n"
        r"- Source: (.+?)\n"
        r"(?:- Source URL: (.+?)\n)?"
    )
    matches = pattern.findall(text)

    if not matches:
        print("No [ACCEPT] entries found in proposals file.")
        print("Edit the file and change [REJECT] to [ACCEPT] for beliefs to keep.")
        return

    print(f"Found {len(matches)} accepted beliefs")

    added = 0
    failed = 0
    for match in matches:
        belief_id, claim_text, source = match[0], match[1], match[2]
        source_url = match[3] if len(match) > 3 else ""
        if source_url and source_url.lower() == "none":
            source_url = ""
        try:
            add_node(
                node_id=belief_id,
                text=claim_text.strip(),
                source=source.strip(),
                source_url=source_url.strip() if source_url else "",
                db_path=REASONS_DB,
            )
            print(f"  Added: {belief_id}")
            added += 1
        except Exception as e:
            err = str(e)
            if "already exists" in err.lower() or "duplicate" in err.lower():
                print(f"  EXISTS: {belief_id}")
            else:
                print(f"  FAIL: {belief_id}: {err}")
                failed += 1

    print(f"\nAccepted {added} beliefs ({failed} failed)")
