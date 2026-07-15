"""Repair inaccurate premises by rewriting from source or retracting.

Takes premises flagged by review-premises and either rewrites them to
accurately reflect the source material, or retracts them if the source
doesn't support any version of the claim.
"""

import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import invoke_model

REPAIR_PREMISES_PROMPT = """\
You are repairing inaccurate premises in a Truth Maintenance System (TMS).
Each premise below was flagged as inaccurate by a prior review step.
For each premise, the source document and the review comment explaining
the error are provided.

For each premise, decide:
- **rewrite**: if the source supports a corrected version of the claim,
  produce the corrected text that accurately reflects the source.
- **retract**: if the source does not support any version of the claim,
  mark it for retraction.

Return ONLY a JSON array. For each premise, one object:

```json
[
  {{
    "id": "premise-id",
    "action": "rewrite",
    "corrected_text": "The corrected claim that matches the source.",
    "rationale": "Brief explanation of what was wrong and how it was fixed."
  }}
]
```

Rules:
- Return one object per premise, in the same order as presented.
- "action" must be either "rewrite" or "retract".
- If action is "rewrite", "corrected_text" must be a complete replacement
  for the original claim. Keep the same scope and style.
- If action is "retract", "corrected_text" should be null or omitted.
- "rationale" should be a single sentence.
- Only use information from the source document. Do not add external knowledge.

## Premises to repair

{premises}"""


def format_premise_for_repair(node_id, nodes, source_content, review_result):
    """Format one premise with source text and review comment for repair."""
    node = nodes.get(node_id)
    if not node:
        return ""

    lines = [f"### {node_id}"]
    lines.append(f"Original claim: {node.get('text', '')}")

    error_type = review_result.get("error_type", "inaccurate")
    comment = review_result.get("comment", "")
    lines.append(f"Error type: {error_type}")
    if comment:
        lines.append(f"Review comment: {comment}")

    source = node.get("source", "")
    if source:
        lines.append(f"Source reference: {source}")

    lines.append("")
    lines.append("Source document content:")
    lines.append("```")
    lines.append(source_content)
    lines.append("```")

    return "\n".join(lines)


def parse_repair_response(response):
    """Extract repair results JSON array from LLM response."""
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
            results.append({
                "id": item["id"],
                "action": item.get("action", "retract"),
                "corrected_text": item.get("corrected_text"),
                "rationale": item.get("rationale", ""),
            })
        if results:
            return results
    return []


def _repair_one(node_id, nodes, source_contents, review_results, model, timeout):
    """Repair a single premise. Returns a result dict."""
    review = review_results.get(node_id, {})
    source = nodes[node_id].get("source", "")
    source_text = source_contents.get(source, "(source not available)")

    prompt_text = format_premise_for_repair(node_id, nodes, source_text, review)
    prompt = REPAIR_PREMISES_PROMPT.format(premises=prompt_text)

    try:
        response = invoke_model(prompt, model=model, timeout=timeout)
        parsed = parse_repair_response(response)
        if parsed:
            return parsed[0]
        return {"id": node_id, "action": "retract", "corrected_text": None,
                "rationale": "Could not parse repair response"}
    except Exception as e:
        return {"id": node_id, "action": "error", "corrected_text": None,
                "rationale": str(e)}


def repair_premises(nodes, premise_ids, source_contents, review_results,
                    model="claude", timeout=300, parallel=0, on_result=None):
    """Repair inaccurate premises by rewriting or retracting.

    Args:
        nodes: Dict of node_id -> node data from export_network().
        premise_ids: List of premise IDs to repair.
        source_contents: Dict of source_path -> file content string.
        review_results: Dict of node_id -> review result dict.
        model: LLM model to use.
        timeout: LLM timeout in seconds.
        parallel: Number of concurrent workers (0 = sequential).
        on_result: Callback(all_results) after each result.

    Returns: list of repair result dicts.
    """
    if not premise_ids:
        return []

    all_results = []

    if parallel > 0:
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    _repair_one, pid, nodes, source_contents,
                    review_results, model, timeout
                ): pid
                for pid in premise_ids
            }
            for i, future in enumerate(as_completed(futures), 1):
                pid = futures[future]
                print(f"  Repaired {i}/{len(premise_ids)}: {pid}", file=sys.stderr)
                result = future.result()
                all_results.append(result)
                if on_result is not None:
                    on_result(all_results)
    else:
        for i, pid in enumerate(premise_ids, 1):
            print(f"  Repairing {i}/{len(premise_ids)}: {pid}...", file=sys.stderr)
            result = _repair_one(pid, nodes, source_contents,
                                 review_results, model, timeout)
            all_results.append(result)
            if on_result is not None:
                on_result(all_results)

    return all_results
