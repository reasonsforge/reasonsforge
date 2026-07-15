"""Generate a problems/open-issues report from gated beliefs.

Identifies retracted premises (fixed defects) and active IN blockers
(open problems), then generates a structured markdown report.  Optionally
uses an LLM to synthesize a narrative version.
"""

from datetime import date


REPORT_GATED_PROMPT = """\
You are generating a problems and open issues report for a belief network \
(Truth Maintenance System).  The report should be a clear, professional \
markdown document suitable for stakeholders who want to understand the \
current state of known issues and what has already been fixed.

## Guidelines

- Write in clear, professional prose
- Use markdown headers (##, ###) to organize sections
- Start with a brief Summary section with overall statistics
- Include a "Fixed Defects" section for retracted premises — these are \
  solved problems.  For each, mention the belief ID in backticks, its \
  description, how it was resolved, and its downstream impact (number of \
  dependent beliefs affected)
- Include an "Open Problems" section for active blockers, numbered.  For \
  each, include its description, the number of gated beliefs it blocks, \
  and the IDs of those gated beliefs
- Include a brief "Impact Analysis" section discussing themes and priorities
- Reference belief IDs in backticks when mentioned
- Do NOT invent information not present in the data
- Do NOT include a title — the report already has one

## Network Data

{data}

Write the report now.
"""


def _build_structured_report(in_count, out_count, gated_data, retracted_premises,
                             blocker_details):
    """Generate a structured markdown report without an LLM."""
    lines = []
    lines.append("# Gated Beliefs Report")
    lines.append("")

    blocker_count = gated_data["blocker_count"]
    gated_count = gated_data["gated_count"]
    today = date.today().isoformat()
    lines.append(
        f"*Generated {today} — {in_count} IN / {out_count} OUT beliefs, "
        f"{blocker_count} active blocker(s) gating {gated_count} belief(s)*"
    )
    lines.append("")

    if retracted_premises:
        lines.append("## Fixed Defects (Retracted Premises)")
        lines.append("")
        lines.append("| Defect | Impact | Resolution |")
        lines.append("|--------|--------|------------|")
        for rp in sorted(retracted_premises, key=lambda r: -r["dependent_count"]):
            dep = rp["dependent_count"]
            impact = f"{dep} dependent(s)" if dep else "no dependents"
            text = rp["text"][:80].replace("|", "\\|")
            reason = rp["retract_reason"].replace("|", "\\|")
            lines.append(
                f"| `{rp['id']}` — {text} | {impact} | {reason} |"
            )
        lines.append("")

    blockers = gated_data.get("blockers", {})
    if blockers:
        lines.append("## Active Blockers")
        lines.append("")
        for i, (bid, bdata) in enumerate(
            sorted(blockers.items(), key=lambda kv: -len(kv[1]["gated"])), 1
        ):
            detail = blocker_details.get(bid, {})
            dep_count = detail.get("dependent_count", 0)
            gated = bdata["gated"]
            lines.append(f"### {i}. `{bid}`")
            lines.append("")
            lines.append(bdata["text"])
            lines.append("")
            lines.append(
                f"**Gates {len(gated)} belief(s)** "
                f"| {dep_count} total dependent(s)"
            )
            lines.append("")
            for g in sorted(gated, key=lambda g: g["id"]):
                lines.append(f"- `{g['id']}`: {g['text'][:100]}")
            lines.append("")
    elif not retracted_premises:
        lines.append("No blockers or retracted premises found.")
        lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append(
        "This report was generated from the belief network using "
        "`reasons list-gated` (active blockers), retracted premises with "
        "`retract_reason` metadata (fixed defects), and `reasons show` "
        "for impact counts."
    )
    lines.append("")

    return "\n".join(lines)


def _format_data_for_prompt(in_count, out_count, gated_data, retracted_premises,
                            blocker_details):
    """Format all gathered data into structured text for the LLM prompt."""
    lines = []
    lines.append(f"Total beliefs: {in_count} IN, {out_count} OUT")
    lines.append(
        f"Blockers: {gated_data['blocker_count']} gating "
        f"{gated_data['gated_count']} belief(s)"
    )
    lines.append(f"Retracted premises (fixed defects): {len(retracted_premises)}")
    lines.append("")

    if retracted_premises:
        lines.append("### Fixed Defects")
        lines.append("")
        for rp in sorted(retracted_premises, key=lambda r: -r["dependent_count"]):
            lines.append(f"ID: {rp['id']}")
            lines.append(f"Text: {rp['text']}")
            lines.append(f"Retract reason: {rp['retract_reason']}")
            lines.append(f"Dependent count: {rp['dependent_count']}")
            lines.append("")

    blockers = gated_data.get("blockers", {})
    if blockers:
        lines.append("### Active Blockers")
        lines.append("")
        for bid, bdata in sorted(
            blockers.items(), key=lambda kv: -len(kv[1]["gated"])
        ):
            detail = blocker_details.get(bid, {})
            lines.append(f"Blocker ID: {bid}")
            lines.append(f"Blocker text: {bdata['text']}")
            lines.append(f"Total dependents: {detail.get('dependent_count', 0)}")
            lines.append(f"Gates {len(bdata['gated'])} belief(s):")
            for g in sorted(bdata["gated"], key=lambda g: g["id"]):
                lines.append(f"  - {g['id']}: {g['text']}")
            lines.append("")

    return "\n".join(lines)


def generate_report(data_text, model, timeout):
    """Generate a narrative report using an LLM."""
    from .llm import invoke_model

    prompt = REPORT_GATED_PROMPT.format(data=data_text)
    return invoke_model(prompt, model=model, timeout=timeout)


def report_gated(gated_data, retracted_premises, blocker_details,
                 in_count, out_count, model="", timeout=300):
    """Generate a gated-beliefs report.

    Returns a markdown string — either structured (default) or
    LLM-synthesized (when model is set).
    """
    if model:
        data_text = _format_data_for_prompt(
            in_count, out_count, gated_data, retracted_premises,
            blocker_details,
        )
        content = generate_report(data_text, model, timeout)
        lines = ["# Gated Beliefs Report", ""]
        today = date.today().isoformat()
        lines.append(
            f"*Generated {today} — {in_count} IN / {out_count} OUT beliefs, "
            f"{gated_data['blocker_count']} active blocker(s) gating "
            f"{gated_data['gated_count']} belief(s)*"
        )
        lines.append("")
        lines.append(content)
        lines.append("")
        return "\n".join(lines)

    return _build_structured_report(
        in_count, out_count, gated_data, retracted_premises,
        blocker_details,
    )
