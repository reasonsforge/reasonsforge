"""Review premises against their source material for factual accuracy.

Premises are taken as ground truth in the TMS — they have no antecedents
to validate against. This module checks whether premises accurately reflect
what their source documents actually say, catching fabricated details,
overgeneralizations, and unsupported claims.
"""

import json
import sys

from .llm import invoke_model

REVIEW_BATCH_SIZE = 5

REVIEW_PREMISES_PROMPT = """\
You are auditing premises in a Truth Maintenance System (TMS).
Each premise below was extracted from a source document by an earlier LLM pass.
The system stored the claim but did not verify whether the source actually
says what the premise claims.

For each premise, the source document text is provided. Evaluate two axes:

1. **Accurate**: Does the source material actually state or clearly imply
   this claim? Watch for:
   - Details added that the source never mentions (misread_source)
   - Claims that go beyond what the source supports (overgeneralized)
   - Information fabricated with no basis in the source (fabricated)
   - Claims the source does not address at all (unsupported)

2. **Well-scoped**: Is the claim appropriately scoped? A premise that says
   "X always does Y" when the source says "X sometimes does Y" is
   overgeneralized even if the core idea is present.

Return ONLY a JSON array. For each premise, one object:

```json
[
  {{
    "id": "premise-id",
    "accurate": true,
    "well_scoped": true,
    "error_type": null,
    "comment": "brief explanation"
  }}
]
```

Rules:
- Return one object per premise reviewed, in the same order as presented.
- "error_type" must be one of: "misread_source", "overgeneralized",
  "fabricated", "unsupported", or null if the premise is accurate.
- "comment" should be a single sentence explaining the most important finding.
- If accurate is false, error_type MUST be set (not null).
- If accurate is true, error_type MUST be null.
- Be rigorous: a claim that sounds reasonable but isn't in the source is NOT accurate.
- Focus on what the source document says, not on general knowledge.

## Premises to review

{premises}"""


def format_premise_for_review(node_id, nodes, source_content):
    """Format one premise with its source text for LLM review."""
    node = nodes.get(node_id)
    if not node:
        return ""

    lines = [f"### {node_id}"]
    lines.append(f"Claim: {node.get('text', '')}")

    source = node.get("source", "")
    if source:
        lines.append(f"Source reference: {source}")

    lines.append("")
    lines.append("Source document content:")
    lines.append("```")
    lines.append(source_content)
    lines.append("```")

    return "\n".join(lines)


def parse_premise_review_response(response):
    """Extract premise review results JSON array from LLM response."""
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
                "accurate": item.get("accurate", True),
                "well_scoped": item.get("well_scoped", True),
                "error_type": item.get("error_type"),
                "comment": item.get("comment", ""),
            })
        if results:
            return results
    return []


def _process_batch(batch, nodes, source_contents, model, timeout):
    """Process a single review batch. Returns (batch_results, error_or_None)."""
    premises_text = "\n\n".join(
        format_premise_for_review(
            pid, nodes,
            source_contents.get(nodes[pid].get("source", ""), "(source not available)")
        )
        for pid in batch
    )
    prompt = REVIEW_PREMISES_PROMPT.format(premises=premises_text)
    response = invoke_model(prompt, model=model, timeout=timeout)
    return parse_premise_review_response(response)


def review_premises(nodes, premise_ids, source_contents, model="claude",
                    timeout=300, batch_size=REVIEW_BATCH_SIZE, on_batch=None,
                    parallel=0):
    """Review premises against their source material.

    Args:
        nodes: Dict of node_id -> node data from export_network().
        premise_ids: List of premise IDs to review.
        source_contents: Dict of source_path -> file content string.
        model: LLM model to use for review.
        timeout: LLM timeout in seconds.
        batch_size: Number of premises per LLM call.
        on_batch: Callback(all_results) after each batch.
        parallel: Number of concurrent workers (0 = sequential).

    Returns: list of review result dicts.
    """
    if not premise_ids:
        return []

    source_for_premise = {}
    for pid in premise_ids:
        node = nodes.get(pid)
        if node:
            source_for_premise[pid] = node.get("source", "")

    by_source = {}
    for pid in premise_ids:
        src = source_for_premise.get(pid, "")
        by_source.setdefault(src, []).append(pid)

    ordered_ids = []
    for src_pids in by_source.values():
        ordered_ids.extend(src_pids)

    batches = []
    for i in range(0, len(ordered_ids), batch_size):
        batches.append(ordered_ids[i:i + batch_size])

    all_results = []
    total_batches = len(batches)

    if parallel > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    _process_batch, batch, nodes, source_contents, model, timeout
                ): batch_num
                for batch_num, batch in enumerate(batches, 1)
            }
            for future in as_completed(futures):
                batch_num = futures[future]
                try:
                    results = future.result()
                    all_results.extend(results)
                    print(f"  Batch {batch_num}/{total_batches} complete "
                          f"({len(results)} results)", file=sys.stderr)
                    if on_batch is not None:
                        on_batch(all_results)
                except Exception as e:
                    print(f"  WARN: batch {batch_num} failed: {e}", file=sys.stderr)
    else:
        for batch_num, batch in enumerate(batches, 1):
            print(f"  Reviewing batch {batch_num}/{total_batches} "
                  f"({len(batch)} premises)...", file=sys.stderr)
            try:
                results = _process_batch(batch, nodes, source_contents, model, timeout)
                all_results.extend(results)
                if on_batch is not None:
                    on_batch(all_results)
            except Exception as e:
                print(f"  WARN: batch {batch_num} failed: {e}", file=sys.stderr)

    return all_results
