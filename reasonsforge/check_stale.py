"""Detect stale nodes by comparing source file hashes.

A node is stale when the file it was sourced from has changed since
the node was created. This is detected by comparing the stored
source_hash against the current SHA-256 hash of the source file.
"""

import hashlib
import re
import subprocess
from datetime import date
from pathlib import Path

from .network import Network


def hash_file(path: Path) -> str:
    """Full SHA-256 hash of file content (64 hex chars)."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_source_path(
    source: str,
    repos: dict[str, Path] | None = None,
    db_dir: Path | None = None,
    agent: str | None = None,
) -> Path | None:
    """Resolve a source string like 'repo-name/path/to/file.md' to an absolute path.

    Tries agent repo first (for agent-imported beliefs where source is relative
    to the agent's repo), then db_dir (for expert repos where sources live next
    to reasons.db), then repos dict by first path component, then ~/git/ fallback.
    """
    if not source:
        return None

    if agent and repos and agent in repos:
        p = repos[agent] / source
        if p.exists():
            return p

    if db_dir:
        p = db_dir / source
        if p.exists():
            return p

    parts = source.split("/", 1)
    if len(parts) < 2:
        p = Path(source)
        return p if p.exists() else None

    repo_name, rel_path = parts

    if repos and repo_name in repos:
        p = repos[repo_name] / rel_path
    else:
        p = Path.home() / "git" / repo_name / rel_path

    return p if p.exists() else None


def check_stale(
    network: Network,
    repos: dict[str, Path] | None = None,
    db_dir: Path | None = None,
    upgrade_hashes: bool = False,
    git_aware: bool = False,
) -> tuple[list[dict], int, int]:
    """Check all IN nodes for source staleness.

    If upgrade_hashes=True, truncated hashes that are a prefix of the
    current full hash are upgraded in place (caller must save the network).

    If git_aware=True, nodes with pinned_sha in metadata skip content
    hashing when no commits touched the file since the pinned SHA.

    Returns (stale_results, upgraded_count, sha_bumped_count).
    """
    if repos is None and network.repos:
        repos = {k: Path(v) for k, v in network.repos.items()}

    results = []
    upgraded = 0
    sha_bumped = 0

    for nid, node in sorted(network.nodes.items()):
        if node.truth_value != "IN":
            continue
        if not node.source or not node.source_hash:
            continue

        agent = node.metadata.get("agent") if node.metadata else None
        path = resolve_source_path(node.source, repos, db_dir, agent=agent)
        if path is None:
            results.append({
                "node_id": nid,
                "old_hash": node.source_hash,
                "new_hash": None,
                "source": node.source,
                "source_path": None,
                "reason": "source_deleted",
            })
            continue

        pinned_sha = node.metadata.get("pinned_sha") if node.metadata else None
        if git_aware and pinned_sha:
            pinned_lines = node.metadata.get("pinned_lines") if node.metadata else None
            if pinned_lines:
                try:
                    parts = pinned_lines.split("-")
                    ls, le = int(parts[0]), int(parts[1])
                except (ValueError, IndexError):
                    pass  # malformed pinned_lines — fall through to content hash
                else:
                    if not file_lines_changed_since(path, pinned_sha, ls, le):
                        new_sha = get_file_commit_sha(path)
                        if new_sha and new_sha != pinned_sha:
                            node.metadata["pinned_sha"] = new_sha
                            node.metadata["verified_at"] = date.today().isoformat()
                            node.source_hash = hash_file(path)
                            sha_bumped += 1
                        continue
            elif not file_changed_since(path, pinned_sha):
                continue

        current_hash = hash_file(path)
        if current_hash != node.source_hash:
            if len(node.source_hash) == 16 and current_hash.startswith(node.source_hash):
                if upgrade_hashes:
                    node.source_hash = current_hash
                    upgraded += 1
                    continue
                results.append({
                    "node_id": nid,
                    "old_hash": node.source_hash,
                    "new_hash": current_hash,
                    "source": node.source,
                    "source_path": str(path),
                    "reason": "truncated_hash",
                })
                continue
            results.append({
                "node_id": nid,
                "old_hash": node.source_hash,
                "new_hash": current_hash,
                "source": node.source,
                "source_path": str(path),
                "reason": "content_changed",
            })
        elif git_aware and pinned_sha:
            new_sha = get_file_commit_sha(path)
            if new_sha and new_sha != pinned_sha:
                node.metadata["pinned_sha"] = new_sha
                node.metadata["verified_at"] = date.today().isoformat()
                sha_bumped += 1

    return results, upgraded, sha_bumped


def hash_sources(
    network: Network,
    repos: dict[str, Path] | None = None,
    force: bool = False,
    db_dir: Path | None = None,
) -> list[dict]:
    """Backfill source hashes for nodes that have a source path but no stored hash.

    If force=True, re-hashes all nodes with sources (even those that already
    have a hash). Use after confirming a source change is expected.

    Returns a list of dicts for each node that was hashed:
        {"node_id": str, "source": str, "hash": str, "was_empty": bool}
    """
    if repos is None and network.repos:
        repos = {k: Path(v) for k, v in network.repos.items()}

    results = []

    for nid, node in sorted(network.nodes.items()):
        if not node.source:
            continue
        if node.source_hash and not force:
            continue

        agent = node.metadata.get("agent") if node.metadata else None
        path = resolve_source_path(node.source, repos, db_dir, agent=agent)
        if path is None:
            continue

        new_hash = hash_file(path)
        was_empty = not node.source_hash
        node.source_hash = new_hash
        results.append({
            "node_id": nid,
            "source": node.source,
            "hash": new_hash,
            "was_empty": was_empty,
        })

    return results


# --- Git-aware pinning ---


def get_file_commit_sha(filepath: Path) -> str | None:
    """Get the last commit SHA that touched this file."""
    result = subprocess.run(
        ["git", "log", "-1", "--format=%H", "--", str(filepath.name)],
        capture_output=True, text=True,
        cwd=str(filepath.parent),
        timeout=30,
    )
    if result.returncode == 0 and result.stdout.strip():
        return result.stdout.strip()
    return None


def file_changed_since(filepath: Path, since_sha: str) -> bool:
    """Check if file was modified since since_sha (committed or uncommitted).

    Returns True on git errors to force fallback to content hashing.
    """
    try:
        # Check committed changes
        result = subprocess.run(
            ["git", "log", "--format=%H", f"{since_sha}..HEAD",
             "--", str(filepath.name)],
            capture_output=True, text=True,
            cwd=str(filepath.parent),
            timeout=30,
        )
        if result.returncode != 0:
            return True
        if result.stdout.strip():
            return True
        # Check uncommitted changes (staged + unstaged)
        result = subprocess.run(
            ["git", "diff", "--quiet", "HEAD", "--", str(filepath.name)],
            capture_output=True, text=True,
            cwd=str(filepath.parent),
            timeout=30,
        )
        if result.returncode != 0:
            return True
        return False
    except subprocess.TimeoutExpired:
        return True


def parse_diff_hunks(diff_output: str) -> list[tuple[int, int]]:
    """Parse unified diff hunk headers, return OLD-side (start, end) ranges.

    Skips hunks with count=0 (pure insertions that don't affect old lines).
    """
    hunks = []
    for m in re.finditer(r'^@@ -(\d+)(?:,(\d+))? \+\d+(?:,\d+)? @@', diff_output, re.MULTILINE):
        start = int(m.group(1))
        count = int(m.group(2)) if m.group(2) is not None else 1
        if count == 0:
            continue
        hunks.append((start, start + count - 1))
    return hunks


def file_lines_changed_since(
    filepath: Path, since_sha: str, line_start: int, line_end: int,
) -> bool:
    """Check if specific lines were modified since since_sha.

    Diffs since_sha against the working tree (committed + uncommitted) in
    a single call. Uses the OLD side of the diff (stored lines reference the
    file at pinned_sha). Returns True on git errors to force content hash fallback.
    """
    try:
        result = subprocess.run(
            ["git", "diff", "-U0", since_sha, "--", str(filepath.name)],
            capture_output=True, text=True,
            cwd=str(filepath.parent),
            timeout=30,
        )
        if result.returncode != 0:
            return True
        for hunk_start, hunk_end in parse_diff_hunks(result.stdout):
            if hunk_end >= line_start and hunk_start <= line_end:
                return True
        return False
    except subprocess.TimeoutExpired:
        return True


def pin_source_url(url: str, sha: str, source_path: str = "") -> str:
    """Rewrite GitHub blob URL from branch ref to commit SHA.

    If source_path is provided, uses it to locate the file path portion
    of the URL (handles branch names containing slashes).
    """
    m = re.match(r'(https://github\.com/[^/]+/[^/]+/blob/).+', url)
    if not m:
        return url
    prefix = m.group(1)
    if source_path and url.endswith(source_path):
        return f"{prefix}{sha}/{source_path}"
    rest = url[len(prefix):]
    parts = rest.split("/", 1)
    if len(parts) == 2:
        return f"{prefix}{sha}/{parts[1]}"
    return url


def pin_sources(
    network: Network,
    repos: dict[str, Path] | None = None,
    db_dir: Path | None = None,
    force: bool = False,
    pin_urls: bool = False,
) -> list[dict]:
    """Pin IN nodes to their current git commit SHA.

    Stores pinned_sha and verified_at in node metadata. Also backfills
    source_hash if missing.

    Returns list of dicts: {"node_id", "source", "pinned_sha", "was_empty"}
    """
    if repos is None and network.repos:
        repos = {k: Path(v) for k, v in network.repos.items()}

    today = date.today().isoformat()
    results = []

    for nid, node in sorted(network.nodes.items()):
        if node.truth_value != "IN":
            continue
        if not node.source:
            continue
        if not force and node.metadata.get("pinned_sha"):
            continue

        agent = node.metadata.get("agent") if node.metadata else None
        path = resolve_source_path(node.source, repos, db_dir, agent=agent)
        if path is None:
            continue

        sha = get_file_commit_sha(path)
        if sha is None:
            continue

        was_empty = not node.metadata.get("pinned_sha")
        node.metadata["pinned_sha"] = sha
        node.metadata["verified_at"] = today

        if not node.source_hash:
            node.source_hash = hash_file(path)

        if pin_urls and node.source_url:
            node.source_url = pin_source_url(node.source_url, sha, node.source)

        results.append({
            "node_id": nid,
            "source": node.source,
            "pinned_sha": sha,
            "was_empty": was_empty,
        })

    return results


def pin_update(
    network: Network,
    node_ids: list[str],
    repos: dict[str, Path] | None = None,
    db_dir: Path | None = None,
) -> list[dict]:
    """Bump pinned_sha to current HEAD for specified nodes.

    Returns list of dicts: {"node_id", "old_sha", "new_sha", "source"}
    """
    if repos is None and network.repos:
        repos = {k: Path(v) for k, v in network.repos.items()}

    today = date.today().isoformat()
    results = []

    for nid in node_ids:
        node = network.nodes.get(nid)
        if node is None:
            results.append({
                "node_id": nid, "error": "not found",
            })
            continue

        if not node.source:
            results.append({
                "node_id": nid, "error": "no source",
            })
            continue

        agent = node.metadata.get("agent") if node.metadata else None
        path = resolve_source_path(node.source, repos, db_dir, agent=agent)
        if path is None:
            results.append({
                "node_id": nid, "error": "source not found",
            })
            continue

        sha = get_file_commit_sha(path)
        if sha is None:
            results.append({
                "node_id": nid, "error": "not in git repo",
            })
            continue

        old_sha = node.metadata.get("pinned_sha", "")
        node.metadata["pinned_sha"] = sha
        node.metadata["verified_at"] = today
        node.source_hash = hash_file(path)

        results.append({
            "node_id": nid,
            "old_sha": old_sha,
            "new_sha": sha,
            "source": node.source,
        })

    return results


def pin_lines(
    network: Network,
    node_id: str,
    line_start: int,
    line_end: int,
    repos: dict[str, Path] | None = None,
    db_dir: Path | None = None,
) -> dict:
    """Set pinned_lines on a node, auto-pinning SHA if not already set."""
    if line_start < 1:
        raise ValueError(f"line_start must be >= 1, got {line_start}")
    if line_end < line_start:
        raise ValueError(f"line_end ({line_end}) must be >= line_start ({line_start})")

    node = network.nodes.get(node_id)
    if node is None:
        raise KeyError(f"node {node_id!r} not found")
    if not node.source:
        raise ValueError(f"node {node_id!r} has no source")

    if repos is None and network.repos:
        repos = {k: Path(v) for k, v in network.repos.items()}

    if not node.metadata:
        node.metadata = {}
    node.metadata["pinned_lines"] = f"{line_start}-{line_end}"

    auto_pinned = False
    if not node.metadata.get("pinned_sha"):
        agent = node.metadata.get("agent") if node.metadata else None
        path = resolve_source_path(node.source, repos, db_dir, agent=agent)
        if path is not None:
            sha = get_file_commit_sha(path)
            if sha:
                node.metadata["pinned_sha"] = sha
                node.metadata["verified_at"] = date.today().isoformat()
                auto_pinned = True
            node.source_hash = hash_file(path)

    return {
        "node_id": node_id,
        "pinned_lines": node.metadata["pinned_lines"],
        "pinned_sha": node.metadata.get("pinned_sha", ""),
        "auto_pinned": auto_pinned,
    }
