"""Export a reasons network as a HuggingFace-compatible EEM card (README.md)."""

from .metadata import build_meta
from .network import Network


def _node_depth(nid, nodes, memo=None):
    if memo is None:
        memo = {}
    if nid in memo:
        return memo[nid]
    node = nodes.get(nid)
    if not node or not node.justifications:
        memo[nid] = 0
        return 0
    memo[nid] = 0
    si = node.supporting_justification
    if si is not None and 0 <= si < len(node.justifications):
        justifications = [node.justifications[si]]
    else:
        justifications = node.justifications
    max_d = 0
    for j in justifications:
        for a in j.antecedents:
            max_d = max(max_d, _node_depth(a, nodes, memo))
    memo[nid] = max_d + 1
    return max_d + 1


def export_card(
    network: Network,
    domain: list[str] | None = None,
    license: str = "mit",
    base_network: str | None = None,
    source_repos: list[str] | None = None,
) -> str:
    """Generate a HuggingFace EEM card from the network.

    Args:
        network: The network to export
        domain: Domain tags for the card frontmatter
        license: License identifier (default: mit)
        base_network: Parent EEM this was derived from
        source_repos: Source repository identifiers
    """
    meta = build_meta(
        project_name=network.meta.get("project_name", ""),
        node_count=len(network.nodes),
        created_at=network.meta.get("created_at", ""),
    )

    project_name = meta["project_name"] or "eem"
    total = len(network.nodes)
    in_count = sum(1 for n in network.nodes.values() if n.truth_value == "IN")
    out_count = total - in_count
    premises = sum(1 for n in network.nodes.values() if not n.justifications)
    derived = total - premises
    nogood_count = len(network.nogoods)

    memo = {}
    max_depth = max((_node_depth(nid, network.nodes, memo) for nid in network.nodes), default=0)

    retraction_pct = f"{out_count / total * 100:.0f}%" if total > 0 else "0%"

    # --- YAML frontmatter ---
    lines = [
        "---",
        f'schema_version: "{meta["schema_version"]}"',
        "type: eem",
        f'project_name: "{project_name}"',
    ]

    if domain:
        lines.append("domain:")
        for d in domain:
            lines.append(f"  - {d}")
    else:
        lines.append("domain: []")

    lines.append(f"license: {license}")

    if base_network:
        lines.append(f'base_network: "{base_network}"')
    else:
        lines.append("base_network: null")

    if source_repos:
        lines.append("source_repos:")
        for sr in source_repos:
            lines.append(f"  - {sr}")
    else:
        lines.append("source_repos: []")

    lines.extend([
        f"beliefs_total: {total}",
        f"beliefs_in: {in_count}",
        f"beliefs_out: {out_count}",
        f"premises: {premises}",
        f"derived: {derived}",
        f"nogoods: {nogood_count}",
        f"generator: {meta['generator']}",
        "---",
        "",
    ])

    # --- Markdown body ---
    title = project_name.replace("-", " ").replace("_", " ").title()
    lines.extend([
        f"# {title}",
        "",
        "## Stats",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total beliefs | {total} |",
        f"| Status | {in_count} IN / {out_count} OUT |",
        f"| Premises (observations) | {premises} |",
        f"| Derived (justified conclusions) | {derived} |",
        f"| Nogoods (contradictions) | {nogood_count} |",
        f"| Retraction rate | {retraction_pct} |",
        f"| Max derivation depth | {max_depth} |",
        "",
        "## How to Use",
        "",
        "### Import into a reasons database",
        "",
        "```bash",
        "reasons init",
        "reasons import-json network.json",
        "```",
        "",
        "### Query beliefs",
        "",
        "```bash",
        'reasons search "your query"',
        "reasons explain <node-id>",
        "reasons show <node-id>",
        "```",
        "",
        "## Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `network.json` | Full belief network (machine-readable, portable) |",
        "| `reasons.db` | SQLite database (gitignored, regenerate with `reasons import-json network.json`) |",
        "| `README.md` | This EEM card |",
        "",
        "## Quality",
        "",
        f"- {in_count} beliefs IN, {out_count} OUT",
        f"- {premises} premises grounded in direct observations",
        f"- {derived} derived beliefs justified via SL justifications",
        f"- {nogood_count} nogoods detected",
        "",
        "## Limitations",
        "",
        "- Auto-generated card — review and customize for your use case",
        "",
        "## License",
        "",
        license,
        "",
    ])

    return "\n".join(lines)
