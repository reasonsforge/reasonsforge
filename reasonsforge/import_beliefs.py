"""Import beliefs.md and nogoods.md into an RMS network.

Parses the markdown format used by the beliefs CLI and converts claims
to RMS nodes with SL justifications for depends_on relationships.

Mapping:
- IN claims with no depends_on → premises
- IN claims with depends_on → SL-justified nodes
- STALE claims → OUT nodes (retracted)
- OUT claims → OUT nodes (retracted)
- nogoods → RMS nogoods
"""

import re
from pathlib import Path

from . import Justification, Node, Nogood
from .network import Network


def strip_frontmatter(text: str) -> tuple[str, dict[str, str]]:
    """Strip YAML frontmatter from markdown text if present.

    Returns (body_text, frontmatter_dict). Frontmatter is a simple
    key: value parse (no nested structures, no YAML library needed).
    If no frontmatter is present, returns (text, {}).
    """
    if not text.startswith("---"):
        return text, {}
    lines = text.split("\n")
    end = -1
    for i, line in enumerate(lines[1:], 1):
        if line.strip() == "---":
            end = i
            break
    if end < 0:
        return text, {}
    fm = {}
    for line in lines[1:end]:
        if ":" in line:
            key, _, val = line.partition(":")
            val = val.strip().strip('"').strip("'")
            fm[key.strip()] = val
    body = "\n".join(lines[end + 1:])
    return body, fm


def parse_repos(text: str) -> dict[str, str]:
    """Parse the ## Repos section from a beliefs.md file.

    Returns a dict of repo name → path.
    """
    repos = {}
    in_repos = False
    for line in text.split("\n"):
        if line.strip() == "## Repos":
            in_repos = True
            continue
        if in_repos and line.startswith("## "):
            break
        if in_repos and line.startswith("- "):
            parts = line[2:].split(":", 1)
            if len(parts) == 2:
                repos[parts[0].strip()] = parts[1].strip()
    return repos


def parse_beliefs(text: str) -> list[dict]:
    """Parse a beliefs.md file into a list of claim dicts."""
    claims = []
    current = None

    for line in text.split("\n"):
        # Claim header: ### claim-id [STATUS] TYPE
        m = re.match(r"^### (\S+) \[(IN|OUT|STALE)\]\s*(.*)", line)
        if m:
            if current:
                claims.append(current)
            current = {
                "id": m.group(1),
                "status": m.group(2),
                "type": m.group(3).strip(),
                "text": "",
                "source": "",
                "source_url": "",
                "source_type": "",
                "source_hash": "",
                "date": "",
                "depends_on": [],
                "unless": [],
                "stale_reason": "",
                "superseded_by": "",
                "accepted_pr": "",
            }
            continue

        if current is None:
            continue

        if line.startswith("- Source: "):
            current["source"] = line[len("- Source: "):].strip()
        elif line.startswith("- Source URL: "):
            current["source_url"] = line[len("- Source URL: "):].strip()
        elif line.startswith("- Source hash: "):
            current["source_hash"] = line[len("- Source hash: "):].strip()
        elif line.startswith("- Date: "):
            current["date"] = line[len("- Date: "):].strip()
        elif line.startswith("- Depends on: "):
            deps = line[len("- Depends on: "):].strip()
            current["depends_on"] = [d.strip() for d in deps.split(",") if d.strip()]
        elif line.startswith("- Unless: "):
            unless = line[len("- Unless: "):].strip()
            current["unless"] = [u.strip() for u in unless.split(",") if u.strip()]
        elif line.startswith("- Source type: "):
            current["source_type"] = line[len("- Source type: "):].strip()
        elif line.lower().startswith("- stale reason: "):
            current["stale_reason"] = line[line.index(": ") + 2:].strip()
        elif line.lower().startswith("- superseded by: "):
            current["superseded_by"] = line[line.index(": ") + 2:].strip()
        elif line.startswith("- Accepted PR: "):
            current["accepted_pr"] = line[len("- Accepted PR: "):].strip()
        elif line.startswith("- "):
            pass  # other metadata lines — skip
        elif line.startswith("### "):
            pass  # next claim header handled above
        elif line.strip() and not current["text"]:
            current["text"] = line.strip()

    if current:
        claims.append(current)

    return claims


def parse_nogoods(text: str) -> list[dict]:
    """Parse a nogoods.md file into a list of nogood dicts."""
    nogoods = []
    current = None

    for line in text.split("\n"):
        m = re.match(r"^### (nogood-\d+):\s*(.*)", line)
        if m:
            if current:
                nogoods.append(current)
            current = {
                "id": m.group(1),
                "label": m.group(2).strip(),
                "discovered": "",
                "resolution": "",
                "affects": [],
            }
            continue

        if current is None:
            continue

        if line.startswith("- Discovered: "):
            current["discovered"] = line[len("- Discovered: "):].strip()
        elif line.startswith("- Resolution: "):
            current["resolution"] = line[len("- Resolution: "):].strip()
        elif line.startswith("- Affects: "):
            affects = line[len("- Affects: "):].strip()
            current["affects"] = [a.strip() for a in affects.split(",") if a.strip()]

    if current:
        nogoods.append(current)

    return nogoods


def import_into_network(
    network: Network,
    beliefs_text: str,
    nogoods_text: str | None = None,
) -> dict:
    """Import parsed beliefs and nogoods into an existing network.

    Returns a summary dict with counts of what was imported.
    """
    beliefs_text, frontmatter = strip_frontmatter(beliefs_text)

    if frontmatter:
        for key in ("schema_version", "project_name", "created_at"):
            if key in frontmatter and frontmatter[key]:
                network.meta[key] = frontmatter[key]

    # Parse and store repos
    repos = parse_repos(beliefs_text)
    network.repos.update(repos)

    claims = parse_beliefs(beliefs_text)

    # Sort claims so that dependencies are added before dependents.
    # Topological sort: claims with no depends_on first, then claims
    # whose dependencies are all already processed.
    claim_by_id = {c["id"]: c for c in claims}
    ordered = []
    added = set()
    remaining = list(claims)

    # Iterative topological sort (handles missing deps gracefully)
    max_passes = len(remaining) + 1
    for _ in range(max_passes):
        if not remaining:
            break
        next_remaining = []
        for c in remaining:
            # A claim is ready if all its depends_on are either already added
            # or not present in this registry (external dependency)
            deps_in_registry = [d for d in c["depends_on"] if d in claim_by_id]
            if all(d in added for d in deps_in_registry):
                ordered.append(c)
                added.add(c["id"])
            else:
                next_remaining.append(c)
        if len(next_remaining) == len(remaining):
            # No progress — remaining claims have circular deps or missing deps
            # Add them anyway (they'll be computed as OUT if deps are missing)
            ordered.extend(next_remaining)
            break
        remaining = next_remaining

    imported = 0
    skipped = 0
    retracted = 0

    for claim in ordered:
        if claim["id"] in network.nodes:
            skipped += 1
            continue

        # Build justifications from depends_on and unless
        justifications = None
        deps_in_network = [d for d in claim["depends_on"] if d in claim_by_id]
        unless_in_network = [u for u in claim.get("unless", []) if u in claim_by_id]
        if deps_in_network or unless_in_network:
            justifications = [
                Justification(
                    type="SL",
                    antecedents=deps_in_network,
                    outlist=unless_in_network,
                    label=f"imported from beliefs: {claim['type']}",
                )
            ]

        metadata = {}
        if claim["type"]:
            metadata["beliefs_type"] = claim["type"]
        if claim.get("source_type"):
            metadata["source_type"] = claim["source_type"]
        if claim["stale_reason"]:
            metadata["stale_reason"] = claim["stale_reason"]
        if claim["superseded_by"]:
            metadata["superseded_by"] = claim["superseded_by"]
        if claim.get("accepted_pr"):
            metadata["accepted_pr"] = claim["accepted_pr"]

        network.add_node(
            id=claim["id"],
            text=claim["text"],
            justifications=justifications,
            source=claim["source"],
            source_url=claim.get("source_url", ""),
            source_hash=claim["source_hash"],
            date=claim["date"],
            metadata=metadata,
        )
        imported += 1

        # STALE and OUT claims get retracted after adding
        if claim["status"] in ("STALE", "OUT"):
            network.retract(claim["id"])
            retracted += 1

    # Import nogoods
    nogoods_imported = 0
    if nogoods_text:
        nogoods = parse_nogoods(nogoods_text)
        for ng in nogoods:
            # Only import if all affected nodes exist in the network
            valid_nodes = [a for a in ng["affects"] if a in network.nodes]
            if len(valid_nodes) >= 2:
                nogood = Nogood(
                    id=ng["id"],
                    nodes=valid_nodes,
                    discovered=ng["discovered"],
                    resolution=ng["resolution"],
                )
                network.nogoods.append(nogood)
                m = re.fullmatch(r"nogood-(\d+)", nogood.id)
                if m:
                    network._next_nogood_id = max(
                        network._next_nogood_id, int(m.group(1)) + 1
                    )
                nogoods_imported += 1

    return {
        "claims_imported": imported,
        "claims_skipped": skipped,
        "claims_retracted": retracted,
        "nogoods_imported": nogoods_imported,
    }
