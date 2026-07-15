"""Token-budgeted belief state summary for context injection.

Produces a compact summary of the network state suitable for inclusion
in CLAUDE.md files or LLM context windows. Prioritizes nogoods and
OUT nodes, then fills remaining budget with IN nodes.
"""

from datetime import date

from .network import Network


def estimate_tokens(text: str) -> int:
    """Token estimate using chars/4 heuristic (standard BPE approximation)."""
    return max(1, len(text) // 4)


def compact(
    network: Network,
    budget: int = 500,
    truncate: bool = True,
) -> str:
    """Generate a token-budgeted belief state summary.

    Priority order (all sections count against the budget):
    1. Nogoods (highest priority)
    2. OUT nodes (need review)
    3. IN nodes by dependent count (most-depended-on first)

    Args:
        network: The RMS network
        budget: Maximum token budget (chars/4 BPE approximation)
        truncate: If True, truncate node text to 80 chars
    """
    in_nodes = []
    out_nodes = []
    for node in network.nodes.values():
        if node.truth_value == "IN":
            in_nodes.append(node)
        else:
            out_nodes.append(node)

    # Sort IN nodes by dependent count (most depended on first)
    in_nodes.sort(key=lambda n: len(n.dependents), reverse=True)

    today = date.today().isoformat()
    in_count = len(in_nodes)
    out_count = len(out_nodes)
    total = in_count + out_count
    nogood_count = len(network.nogoods)

    lines = [
        f"# Belief State Summary ({today})",
        f"# {total} nodes tracked | {nogood_count} nogoods | {in_count} IN | {out_count} OUT",
        "",
    ]

    def _text(node):
        t = node.text
        if truncate and len(t) > 80:
            t = t[:77] + "..."
        return t

    footer_tokens = estimate_tokens(f"Token count: ~{budget} / {budget} budget")
    # Track total chars to derive tokens in O(1) instead of rejoining
    _char_count = sum(len(l) for l in lines) + len(lines) - 1  # +newlines

    def _add_line(line):
        nonlocal _char_count
        lines.append(line)
        _char_count += 1 + len(line)  # +1 for \n separator

    def _current_tokens():
        return max(1, _char_count // 4)

    def _over_budget(line):
        return _current_tokens() + estimate_tokens(line) + footer_tokens > budget

    # Section 1: Nogoods (highest priority, but counted against budget)
    if network.nogoods and not _over_budget("## Nogoods"):
        _add_line("## Nogoods")
        added_nogoods = 0
        for ng in network.nogoods:
            res = f" — {ng.resolution}" if ng.resolution else ""
            line = f"- {ng.id}: {', '.join(ng.nodes)}{res}"
            if _over_budget(line):
                remaining = len(network.nogoods) - added_nogoods
                _add_line(f"  ... ({remaining} more nogoods omitted)")
                break
            _add_line(line)
            added_nogoods += 1
        _add_line("")

    # Section 2: OUT nodes (budget-limited)
    if out_nodes and not _over_budget("## OUT (retracted)"):
        _add_line("## OUT (retracted)")
        added_out = 0
        for node in out_nodes:
            reason = ""
            retract_reason = node.metadata.get("retract_reason") or node.metadata.get("stale_reason")
            if retract_reason:
                reason = f" (stale: {retract_reason[:60]})"
            elif node.metadata.get("superseded_by"):
                reason = f" (superseded by: {node.metadata['superseded_by']})"
            line = f"- {node.id}: {_text(node)}{reason}"
            if _over_budget(line):
                remaining = len(out_nodes) - added_out
                _add_line(f"  ... ({remaining} more OUT nodes omitted)")
                break
            _add_line(line)
            added_out += 1
        _add_line("")

    # Section 3: IN nodes (budget-limited)
    if in_nodes and not _over_budget("## IN (active)"):
        covered_by_summary: set[str] = set()
        summary_nodes = []
        regular_nodes = []
        for node in in_nodes:
            summarizes = node.metadata.get("summarizes")
            if summarizes:
                summary_nodes.append(node)
                for covered_id in summarizes:
                    covered_by_summary.add(covered_id)
            else:
                regular_nodes.append(node)

        visible_nodes = summary_nodes + [
            n for n in regular_nodes if n.id not in covered_by_summary
        ]
        visible_nodes.sort(key=lambda n: len(n.dependents), reverse=True)

        hidden_count = len(in_nodes) - len(visible_nodes)

        _add_line("## IN (active)")
        added = 0

        for node in visible_nodes:
            is_summary = bool(node.metadata.get("summarizes"))
            prefix = "[summary] " if is_summary else ""
            deps = ""
            for j in node.justifications:
                if j.antecedents and j.label != "summarizes":
                    deps = f" <- {', '.join(j.antecedents)}"
                    break
            dep_count = len(node.dependents)
            dep_info = f" ({dep_count} dependents)" if dep_count else ""
            summarizes = node.metadata.get("summarizes", [])
            sum_info = f" (covers {len(summarizes)} nodes)" if summarizes else ""
            line = f"- {prefix}{node.id}: {_text(node)}{deps}{dep_info}{sum_info}"

            if _over_budget(line):
                remaining = len(visible_nodes) - added
                _add_line(f"  ... ({remaining} more IN nodes omitted)")
                break

            _add_line(line)
            added += 1

        if hidden_count:
            _add_line(f"  ({hidden_count} nodes hidden by summaries)")
        _add_line("")

    lines.append(f"Token count: ~{_current_tokens()} / {budget} budget")

    return "\n".join(lines)
