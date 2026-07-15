"""Generate a report tracing a belief back to its root premises with source evidence."""

from datetime import date


REPORT_BELIEF_PROMPT = """\
You are generating an evidence report for a single belief in a Truth Maintenance \
System.  The report should explain why this belief is held (or not held), tracing \
it back to its root premises and the source evidence supporting each premise.

## Guidelines

- Write in clear, professional prose
- Start with a brief summary of the belief and its current status
- Walk through the justification chain, explaining how each intermediate \
  belief derives from its antecedents
- For each root premise, present the source evidence that supports it
- Highlight any premises that are OUT or lack source evidence
- Do NOT invent information not present in the data
- Do NOT include a title — the report already has one

## Belief Data

{data}

Write the report now.
"""


def _build_structured_report(node_id, node_detail, explain_steps, premises_data,
                             source_chunks):
    """Generate a structured markdown report without an LLM."""
    lines = []
    lines.append(f"# Belief Report: `{node_id}`")
    lines.append("")

    today = date.today().isoformat()
    status = node_detail.get("truth_value", "UNKNOWN")
    lines.append(f"*Generated {today} — status: {status}*")
    lines.append("")

    lines.append("## Belief")
    lines.append("")
    lines.append(f"> {node_detail.get('text', '')}")
    lines.append("")

    if node_detail.get("source"):
        lines.append(f"**Source:** `{node_detail['source']}`")
        lines.append("")

    lines.append("## Justification Chain")
    lines.append("")

    if not explain_steps:
        lines.append("No justification chain available.")
        lines.append("")
    else:
        for step in explain_steps:
            nid = step.get("node", "")
            tv = step.get("truth_value", "")
            reason = step.get("reason", "")
            marker = "+" if tv == "IN" else "-"
            lines.append(f"  [{marker}] `{nid}`: {reason}")

            ants = step.get("antecedents", [])
            if ants:
                lines.append(f"      antecedents: {', '.join(f'`{a}`' for a in ants)}")
            label = step.get("label", "")
            if label:
                lines.append(f"      label: {label}")
            outlist = step.get("outlist", [])
            if outlist:
                lines.append(f"      outlist: {', '.join(f'`{o}`' for o in outlist)}")
        lines.append("")

    lines.append(f"## Root Premises ({len(premises_data)})")
    lines.append("")

    if not premises_data:
        lines.append("No root premises found.")
        lines.append("")
    else:
        for pd in premises_data:
            pid = pd["id"]
            tv = pd.get("truth_value", "UNKNOWN")
            marker = "+" if tv == "IN" else "-"
            lines.append(f"### [{marker}] `{pid}`")
            lines.append("")
            lines.append(pd.get("text", ""))
            lines.append("")

            source = pd.get("source", "")
            if source:
                lines.append(f"**Source:** `{source}`")
                lines.append("")

            chunks = source_chunks.get(pid, [])
            if chunks:
                lines.append("**Source Evidence:**")
                lines.append("")
                for i, chunk in enumerate(chunks, 1):
                    header = chunk.get("filename", "")
                    section = chunk.get("section", "")
                    if section:
                        header += f" > {section}"
                    lines.append(f"<details><summary>[{i}] {header}</summary>")
                    lines.append("")
                    lines.append(chunk.get("text", ""))
                    lines.append("")
                    lines.append("</details>")
                    lines.append("")
            elif source_chunks is not None and pid in source_chunks:
                lines.append("*No matching source chunks found.*")
                lines.append("")

    return "\n".join(lines)


def _format_data_for_prompt(node_id, node_detail, explain_steps, premises_data,
                            source_chunks):
    """Format all gathered data into structured text for the LLM prompt."""
    lines = []
    status = node_detail.get("truth_value", "UNKNOWN")
    lines.append(f"Belief: `{node_id}` ({status})")
    lines.append(f"Text: {node_detail.get('text', '')}")
    if node_detail.get("source"):
        lines.append(f"Source: {node_detail['source']}")
    lines.append("")

    lines.append("### Justification Chain")
    lines.append("")
    for step in explain_steps:
        nid = step.get("node", "")
        tv = step.get("truth_value", "")
        reason = step.get("reason", "")
        lines.append(f"- `{nid}` ({tv}): {reason}")
        ants = step.get("antecedents", [])
        if ants:
            lines.append(f"  antecedents: {', '.join(ants)}")
        label = step.get("label", "")
        if label:
            lines.append(f"  label: {label}")
    lines.append("")

    lines.append(f"### Root Premises ({len(premises_data)})")
    lines.append("")
    for pd in premises_data:
        pid = pd["id"]
        tv = pd.get("truth_value", "UNKNOWN")
        lines.append(f"Premise: `{pid}` ({tv})")
        lines.append(f"Text: {pd.get('text', '')}")
        source = pd.get("source", "")
        if source:
            lines.append(f"Source file: {source}")

        chunks = source_chunks.get(pid, [])
        if chunks:
            lines.append("Source evidence:")
            for i, chunk in enumerate(chunks, 1):
                header = chunk.get("filename", "")
                section = chunk.get("section", "")
                if section:
                    header += f" > {section}"
                lines.append(f"  [{i}] {header}")
                lines.append(f"  {chunk.get('text', '')}")
        lines.append("")

    return "\n".join(lines)


def generate_report(data_text, model, timeout):
    """Generate a narrative report using an LLM."""
    from .llm import invoke_model

    prompt = REPORT_BELIEF_PROMPT.format(data=data_text)
    return invoke_model(prompt, model=model, timeout=timeout)


def report_belief(node_id, node_detail, explain_steps, premises_data,
                  source_chunks, model="", timeout=300):
    """Generate a belief evidence report.

    Returns a markdown string — either structured (default) or
    LLM-synthesized (when model is set).
    """
    if model:
        data_text = _format_data_for_prompt(
            node_id, node_detail, explain_steps, premises_data,
            source_chunks,
        )
        content = generate_report(data_text, model, timeout)
        lines = [f"# Belief Report: `{node_id}`", ""]
        today = date.today().isoformat()
        status = node_detail.get("truth_value", "UNKNOWN")
        lines.append(f"*Generated {today} — status: {status}*")
        lines.append("")
        lines.append(content)
        lines.append("")
        return "\n".join(lines)

    return _build_structured_report(
        node_id, node_detail, explain_steps, premises_data,
        source_chunks,
    )
