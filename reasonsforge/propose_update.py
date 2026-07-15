"""Propose structured updates to existing beliefs.

Feeds beliefs to an LLM with their justification chains, source info,
and dependents. Returns structured proposals with failure mode taxonomy,
update basis, evidence, and cascade impact predictions.
"""

import json
import sys

from .llm import invoke_model

PROPOSE_UPDATE_BATCH_SIZE = 10

FAILURE_MODES = {
    "smuggled-premise",
    "unsupported-superlative",
    "false-causal-link",
    "domain-conflation",
    "contradicted-by-source",
    "stale",
}

BASES = {
    "source-divergence",
    "detected-contradiction",
    "prior-knowledge",
}

PROPOSE_UPDATE_PROMPT = """\
You are auditing beliefs in a Truth Maintenance System (TMS).
Each belief below may need updating or retracting based on its current
text, source material, justification chain, and dependents.

For each belief, decide: does it need an update, a retraction, or is it fine?
Only propose changes for beliefs that genuinely need them. If a belief is
accurate and well-supported, skip it (do not include it in the output).

For each belief that needs a change, classify:

1. **action**: "update" (amend the text) or "retract" (remove the belief)

2. **failure_mode**: one of these categories:
   - smuggled-premise: unstated assumption baked into the claim
   - unsupported-superlative: "always", "never", "all" without evidence
   - false-causal-link: correlation presented as causation
   - domain-conflation: mixing concepts from different domains
   - contradicted-by-source: source material disagrees with the claim
   - stale: source material has changed, belief is outdated

3. **basis**: why this update is being proposed:
   - source-divergence: the source file/URL changed (mechanically verifiable, cheap to accept)
   - detected-contradiction: conflicts with another belief (nogood)
   - prior-knowledge: your own assessment (needs scrutiny — reviewer may share your blind spot)

4. **proposed_text**: the corrected text (null if retracting)

5. **evidence**: explanation with URLs where available

6. **comment**: one-sentence summary of the most important finding

Return ONLY a JSON array. For each belief that needs a change, one object:

```json
[
  {{
    "id": "belief-id",
    "action": "update",
    "proposed_text": "The corrected belief text.",
    "failure_mode": "smuggled-premise",
    "basis": "prior-knowledge",
    "evidence": "The original claim assumes X, but the source shows Y.",
    "comment": "Removes unstated assumption about X."
  }}
]
```

If no beliefs need changes, return an empty array: []

Rules:
- Only propose changes for beliefs that genuinely need them.
- Be conservative: when in doubt, don't propose a change.
- For "update" actions, proposed_text must be meaningfully different from current text.
- For "retract" actions, set proposed_text to null.
- Use the failure_mode and basis taxonomies exactly as defined above.
- Include source URLs in evidence when the belief has a source_url.

## Beliefs to review

{beliefs}"""


def format_belief_for_update(node_id, nodes):
    """Format one belief with source info and dependents for LLM review."""
    node = nodes.get(node_id)
    if not node:
        return ""

    lines = [f"### {node_id}"]
    lines.append(f"Text: {node.get('text', '')}")
    lines.append(f"Status: {node.get('truth_value', 'unknown')}")

    source = node.get("source", "")
    if source:
        lines.append(f"Source: {source}")

    source_url = node.get("source_url", "")
    if source_url:
        lines.append(f"Source URL: {source_url}")

    meta = node.get("metadata", {}) or {}
    if meta.get("stale_reason"):
        lines.append(f"Stale reason: {meta['stale_reason']}")

    justs = node.get("justifications", [])
    if justs:
        for ji, j in enumerate(justs):
            antecedents = j.get("antecedents", [])
            outlist = j.get("outlist", [])
            label = j.get("label", "")

            if len(justs) > 1:
                lines.append(f"Justification {ji + 1}/{len(justs)}:")

            if antecedents:
                lines.append("Antecedents:")
                for ant_id in antecedents:
                    ant_node = nodes.get(ant_id)
                    if ant_node:
                        lines.append(f"- {ant_id}: {ant_node.get('text', '')}")
                    else:
                        lines.append(f"- {ant_id}: (not found)")

            if outlist:
                lines.append("Unless (must be OUT):")
                for out_id in outlist:
                    out_node = nodes.get(out_id)
                    if out_node:
                        lines.append(f"- {out_id}: {out_node.get('text', '')}")
                    else:
                        lines.append(f"- {out_id}: (not found)")

            if label:
                lines.append(f"Label: {label}")
    else:
        lines.append("Type: premise (no justifications)")

    dependents = node.get("dependents", [])
    if dependents:
        dep_summaries = []
        for dep_id in sorted(dependents) if isinstance(dependents, (set, list)) else []:
            dep_node = nodes.get(dep_id)
            if dep_node:
                dep_summaries.append(f"- {dep_id}: {dep_node.get('text', '')}")
            else:
                dep_summaries.append(f"- {dep_id}: (not found)")
        if dep_summaries:
            lines.append(f"Dependents ({len(dep_summaries)}):")
            lines.extend(dep_summaries)

    return "\n".join(lines)


def parse_update_proposals(response):
    """Extract update proposals JSON array from LLM response.

    Uses json.JSONDecoder.raw_decode at each '[' position to handle
    prose around the JSON array.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch != "[":
            continue
        try:
            items, _ = decoder.raw_decode(response, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(items, list):
            continue
        results = []
        for item in items:
            if not isinstance(item, dict) or "id" not in item:
                continue
            action = item.get("action", "update")
            if action not in ("update", "retract"):
                action = "update"
            failure_mode = item.get("failure_mode", "")
            if failure_mode not in FAILURE_MODES:
                failure_mode = ""
            basis = item.get("basis", "prior-knowledge")
            if basis not in BASES:
                basis = "prior-knowledge"
            results.append({
                "id": item["id"],
                "action": action,
                "proposed_text": item.get("proposed_text"),
                "failure_mode": failure_mode,
                "basis": basis,
                "evidence": item.get("evidence", ""),
                "comment": item.get("comment", ""),
            })
        if results:
            return results
    return []


def format_proposal_markdown(proposal, nodes=None, cascade=None):
    """Render one proposal as a markdown section."""
    nid = proposal["id"]
    action = proposal["action"].upper()
    lines = [f"## {action}: {nid}", ""]

    current_text = ""
    if nodes:
        node = nodes.get(nid, {})
        current_text = node.get("text", "")
    lines.append(f"**Current text:** {current_text}")

    if action == "UPDATE" and proposal.get("proposed_text"):
        lines.append(f"\n**Proposed text:** {proposal['proposed_text']}")

    lines.append(f"\n**Failure mode:** {proposal.get('failure_mode', 'unknown')}")
    lines.append(f"**Basis:** {proposal.get('basis', 'unknown')}")

    evidence = proposal.get("evidence", "")
    if evidence:
        lines.append(f"**Evidence:** {evidence}")

    if cascade:
        retracted = cascade.get("retracted", [])
        restored = cascade.get("restored", [])
        total = cascade.get("total_affected", 0)
        if total > 0:
            if action == "RETRACT":
                lines.append(f"\n**Cascade impact:** {total} dependent(s) affected")
                for item in retracted:
                    text = item.get("text", "")[:80]
                    lines.append(f"- {item['id']}: \"{text}\" — will go OUT")
                for item in restored:
                    text = item.get("text", "")[:80]
                    lines.append(f"- {item['id']}: \"{text}\" — will go IN")
            else:
                lines.append(f"\n**Cascade impact:** {total} dependent(s) need re-review")
                for item in retracted:
                    text = item.get("text", "")[:80]
                    lines.append(f"- {item['id']}: \"{text}\"")
                for item in restored:
                    text = item.get("text", "")[:80]
                    lines.append(f"- {item['id']}: \"{text}\"")
        else:
            lines.append("\n**Cascade impact:** none")
    else:
        lines.append("\n**Cascade impact:** not computed")

    if action == "UPDATE" and proposal.get("proposed_text"):
        escaped = proposal["proposed_text"].replace('"', '\\"')
        lines.append(f"\n**Command:** `reasons update {nid} \"{escaped}\"`")
    elif action == "RETRACT":
        fm = proposal.get("failure_mode", "unknown")
        lines.append(
            f"\n**Command:** `reasons retract {nid} "
            f"--reason \"{fm}: {proposal.get('comment', '')}\"`"
        )

    return "\n".join(lines)


def format_proposals_file(proposals, nodes=None, cascades=None):
    """Render all proposals as a markdown document."""
    lines = [
        "# Proposed Updates",
        "",
        "Review each proposal below. Accept by running the suggested command.",
        "",
    ]
    if not proposals:
        lines.append("No updates proposed.")
        return "\n".join(lines)

    for proposal in proposals:
        nid = proposal["id"]
        cascade = cascades.get(nid) if cascades else None
        lines.append("---")
        lines.append("")
        lines.append(format_proposal_markdown(proposal, nodes, cascade))
        lines.append("")

    return "\n".join(lines)


def propose_updates(nodes, belief_ids=None, model="claude", timeout=300,
                    batch_size=PROPOSE_UPDATE_BATCH_SIZE, on_batch=None):
    """Propose updates for beliefs by sending them to an LLM in batches.

    Returns list of proposal dicts.
    """
    if belief_ids is None:
        belief_ids = [
            nid for nid, node in sorted(nodes.items())
            if node.get("truth_value") == "IN"
        ]
    else:
        belief_ids = [nid for nid in belief_ids if nid in nodes]

    if not belief_ids:
        return []

    all_results = []
    total_batches = (len(belief_ids) + batch_size - 1) // batch_size

    for i in range(0, len(belief_ids), batch_size):
        batch = belief_ids[i:i + batch_size]
        batch_num = i // batch_size + 1
        print(f"  Reviewing batch {batch_num}/{total_batches} "
              f"({len(batch)} beliefs)...", file=sys.stderr)

        beliefs_text = "\n\n".join(
            format_belief_for_update(nid, nodes) for nid in batch
        )
        prompt = PROPOSE_UPDATE_PROMPT.format(beliefs=beliefs_text)

        try:
            response = invoke_model(prompt, model=model, timeout=timeout)
            results = parse_update_proposals(response)
            all_results.extend(results)
            if on_batch is not None:
                on_batch(all_results)
        except Exception as e:
            print(f"  WARN: batch {batch_num} failed: {e}", file=sys.stderr)

    return all_results
