"""Review SL justifications for ALL vs ANY misclassification.

Scans derived beliefs with multi-antecedent SL justifications and asks
an LLM whether each should be ALL (conjunctive) or ANY (disjunctive).
Reports candidates for conversion but does not modify the database.
"""

import json
import sys

from .llm import invoke_model
from .review import format_belief_for_review

REVIEW_BATCH_SIZE = 15

REVIEW_JUSTIFICATIONS_PROMPT = """\
You are auditing justification modes in a Truth Maintenance System (TMS).
Each belief below was derived from its antecedents. Currently, all use
ALL (conjunctive) justifications — meaning every antecedent must be IN
for the conclusion to hold. If any single antecedent is retracted, the
conclusion is retracted too.

For each belief, evaluate whether the justification mode is correct:

1. **ALL (conjunctive)**: The conclusion genuinely requires all antecedents
   together. It represents a logical chain or synthesis where removing any
   premise invalidates the conclusion.

2. **ANY (disjunctive)**: Each antecedent independently supports the
   conclusion. The conclusion represents convergent evidence — it should
   survive as long as at least one antecedent holds.

3. **MIXED**: Some antecedents are required together (a core group), while
   others are independent additional support. Identify which are which.

Return ONLY a JSON array. For each belief, one object:

```json
[
  {{
    "id": "belief-id",
    "classification": "ALL",
    "required_antecedents": ["id1", "id2"],
    "independent_antecedents": [],
    "comment": "brief explanation"
  }}
]
```

Rules:
- Return one object per belief reviewed, in the same order as presented.
- `classification` must be one of: "ALL", "ANY", "MIXED".
- For ALL: `required_antecedents` = all antecedents, `independent_antecedents` = [].
- For ANY: `required_antecedents` = [], `independent_antecedents` = all antecedents.
- For MIXED: split antecedents into the two lists. `required_antecedents` are those
  that must be present together; `independent_antecedents` are those that each
  independently support the conclusion.
- Be rigorous: if the conclusion synthesizes information from multiple sources in
  a way that requires all of them, it is ALL. If the conclusion is a general
  observation that any single antecedent would justify, it is ANY.

## Beliefs to review

{beliefs}"""


def parse_justification_review(response):
    """Extract justification review results from LLM response."""
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
            cls = item.get("classification")
            req = item.get("required_antecedents")
            ind = item.get("independent_antecedents")
            results.append({
                "id": item["id"],
                "classification": cls.upper() if isinstance(cls, str) else "ALL",
                "required_antecedents": req if isinstance(req, list) else [],
                "independent_antecedents": ind if isinstance(ind, list) else [],
                "comment": item.get("comment") or "",
            })
        if results:
            return results
    return []


def _process_batch(batch, nodes, model, timeout):
    """Process a single batch of beliefs through the LLM."""
    beliefs_text = "\n\n".join(
        format_belief_for_review(nid, nodes) for nid in batch
    )
    prompt = REVIEW_JUSTIFICATIONS_PROMPT.format(beliefs=beliefs_text)
    response = invoke_model(prompt, model=model, timeout=timeout)
    results = parse_justification_review(response)
    if not results and batch:
        print(f"  WARN: batch returned no parseable results", file=sys.stderr)
    return results


def review_justifications(nodes, belief_ids=None, model="claude", timeout=300,
                          batch_size=REVIEW_BATCH_SIZE, min_antecedents=2,
                          on_batch=None, parallel=0):
    """Review SL justifications for ALL vs ANY classification.

    Args:
        nodes: Dict of node_id -> node data from export_network().
        belief_ids: Optional list of specific IDs to review.
        model: LLM model to use.
        timeout: LLM timeout in seconds.
        batch_size: Number of beliefs per LLM call.
        min_antecedents: Only review justifications with at least this many antecedents.
        on_batch: Optional callback after each batch.

    Returns: list of review result dicts.
    """
    if belief_ids is None:
        belief_ids = sorted(
            nid for nid, node in nodes.items()
            if node.get("truth_value") == "IN"
            and _has_multi_antecedent_sl(node, min_antecedents)
        )
    else:
        belief_ids = [
            nid for nid in belief_ids
            if nid in nodes
            and nodes[nid].get("truth_value") == "IN"
            and _has_multi_antecedent_sl(nodes[nid], min_antecedents)
        ]

    if not belief_ids:
        return []

    batches = []
    for i in range(0, len(belief_ids), batch_size):
        batches.append(belief_ids[i:i + batch_size])

    all_results = []
    total_batches = len(batches)

    if parallel > 0:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=parallel) as executor:
            futures = {
                executor.submit(
                    _process_batch, batch, nodes, model, timeout
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
                  f"({len(batch)} beliefs)...", file=sys.stderr)
            try:
                results = _process_batch(batch, nodes, model, timeout)
                all_results.extend(results)
                if on_batch is not None:
                    on_batch(all_results)
            except Exception as e:
                print(f"  WARN: batch {batch_num} failed: {e}", file=sys.stderr)

    return all_results


def _has_multi_antecedent_sl(node, min_antecedents):
    """Check if node has any SL justification with enough antecedents."""
    for j in node.get("justifications", []):
        if j.get("type") == "SL" and len(j.get("antecedents", [])) >= min_antecedents:
            return True
    return False
