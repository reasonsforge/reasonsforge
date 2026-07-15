"""Research and repair flagged beliefs.

Three patterns for resolving beliefs flagged invalid by review-beliefs:

1. Search-and-link — belief is sound but missing an antecedent; find and wire it
2. Soften — belief overstates the evidence; weaken text to match antecedents
3. Abandon — dependency tree too broken to repair; retract

The `repair_beliefs` orchestrator triages each invalid belief via LLM,
then executes the appropriate pattern.
"""

import json
import sys

from .llm import invoke_model
from .review import format_belief_for_review

EXTRACT_PROMPT = """\
You are analyzing a derived belief that was flagged as invalid in a Truth Maintenance System.
The belief's conclusion introduces a factual claim not supported by its antecedents (a smuggled premise).

Your task: identify the specific factual claim that the conclusion smuggles in — the fact it
relies on that is NOT stated or implied by any antecedent.

## Invalid belief

{belief_context}

## Review finding

{review_comment}

Respond with ONLY a single concise factual claim (one sentence, no preamble) that captures
what specific knowledge the conclusion assumes but the antecedents do not provide.
Do NOT restate the conclusion. Extract the missing piece — the gap between what the
antecedents establish and what the conclusion asserts.

Smuggled claim:"""

MATCH_PROMPT = """\
You are matching a smuggled claim against existing premises in a belief network.
The smuggled claim is a factual assertion that was missing from a derivation's antecedents.
Your task: determine which (if any) of the candidate premises below directly supports
this smuggled claim.

## Smuggled claim

{smuggled_claim}

## Candidate premises

{candidates}

Rules:
- A premise "supports" the smuggled claim if it states or directly implies the same fact.
- Do NOT match premises that are merely topically related — they must actually establish
  the smuggled fact.
- If multiple premises jointly support the claim, list all of them.
- If no premise supports the claim, say "none".

Respond with ONLY a JSON object in this exact format:
{{"matched_ids": ["premise-id-1", "premise-id-2"], "rationale": "brief explanation"}}

If no match: {{"matched_ids": [], "rationale": "brief explanation"}}"""

TRIAGE_PROMPT = """\
You are triaging a derived belief that was flagged as invalid in a Truth Maintenance System.
Your task: decide which repair pattern is most appropriate.

## Invalid belief

{belief_context}

## Review finding

{review_comment}

## Dependency info

- Belief depth: {depth}
- Flagged ancestors in this batch: {flagged_ancestors}

## Patterns

1. **search_and_link** — The core claim is sound, but the derivation is missing an antecedent
   that probably exists elsewhere in the network. Choose this when the conclusion is reasonable
   but the justification chain has a gap that could be filled by finding an existing premise.

2. **soften** — The claim overstates what the antecedents support. The insight is partially
   valid but the wording is too strong (e.g., "sole mechanism" when evidence only supports
   "primary mechanism"). Choose this when weakening the text would make it follow from
   the antecedents.

3. **abandon** — The belief sits atop a broken dependency chain or makes claims too far
   removed from the evidence to repair. Choose this when neither linking nor softening
   can make the derivation sound.

4. **research** — The claim is plausible but cannot be confirmed or denied from the
   current evidence in the network. Further investigation (code reading, testing,
   documentation review) could validate or refine it. Choose this when the belief is
   worth preserving pending more information, rather than abandoning or weakening
   prematurely.

Respond with ONLY a JSON object:
{{"pattern": "search_and_link" | "soften" | "abandon" | "research", "rationale": "brief explanation"}}"""

SOFTEN_PROMPT = """\
You are rewriting a derived belief to match what its antecedents actually support.
The original belief was flagged as invalid because it overstates the evidence.

## Original belief

{belief_text}

## Antecedents (the evidence)

{antecedents}

Rewrite the belief so it follows strictly from the antecedents. Weaken any absolute claims
to qualified ones (e.g., "the mechanism" -> "a primary mechanism", "ensures" -> "supports").
Preserve the core insight but remove unsupported specificity.

Respond with ONLY a JSON object:
{{"softened_text": "the rewritten claim", "rationale": "what was weakened and why"}}"""

VALID_PATTERNS = {"search_and_link", "soften", "abandon", "research"}


def parse_extract_response(response):
    """Extract the smuggled claim string from LLM response."""
    text = response.strip()
    if not text:
        return ""
    for prefix in ("Smuggled claim:", "smuggled claim:"):
        if text.lower().startswith(prefix.lower()):
            text = text[len(prefix):].strip()
    if len(text) > 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1].strip()
    return text


def parse_match_response(response, valid_ids):
    """Extract match result JSON from LLM response.

    Uses raw_decode to find JSON object in response text.
    Filters matched_ids to only include IDs in valid_ids.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(response, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        matched = obj.get("matched_ids", [])
        if not isinstance(matched, list):
            matched = []
        filtered = [mid for mid in matched if mid in valid_ids]
        return {
            "matched_ids": filtered,
            "rationale": obj.get("rationale", ""),
        }
    return {"matched_ids": [], "rationale": ""}


def extract_smuggled_claim(belief_context, review_comment, model="claude",
                           timeout=300):
    """LLM call 1: extract the smuggled claim from an invalid belief."""
    prompt = EXTRACT_PROMPT.format(
        belief_context=belief_context,
        review_comment=review_comment,
    )
    response = invoke_model(prompt, model=model, timeout=timeout)
    return parse_extract_response(response)


def find_matching_premises(smuggled_claim, candidates, model="claude",
                           timeout=300):
    """LLM call 2: match smuggled claim against candidate premises.

    Args:
        smuggled_claim: extracted claim string
        candidates: list of {"id": str, "text": str} dicts (IN premises)
    """
    if not candidates:
        return {"matched_ids": [], "rationale": "no candidates found"}

    candidate_lines = "\n".join(
        f"- `{c['id']}`: {c['text']}" for c in candidates
    )
    prompt = MATCH_PROMPT.format(
        smuggled_claim=smuggled_claim,
        candidates=candidate_lines,
    )
    response = invoke_model(prompt, model=model, timeout=timeout)
    valid_ids = {c["id"] for c in candidates}
    return parse_match_response(response, valid_ids)


def parse_triage_response(response):
    """Extract triage result JSON from LLM response."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(response, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        pattern = obj.get("pattern", "")
        if pattern not in VALID_PATTERNS:
            continue
        return {
            "pattern": pattern,
            "rationale": obj.get("rationale", ""),
        }
    return {"pattern": "", "rationale": ""}


def parse_soften_response(response):
    """Extract softened text JSON from LLM response."""
    decoder = json.JSONDecoder()
    for i, ch in enumerate(response):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(response, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        text = obj.get("softened_text", "")
        if text:
            return {
                "softened_text": text,
                "rationale": obj.get("rationale", ""),
            }
    return {"softened_text": "", "rationale": ""}


def triage_belief(belief_context, review_comment, depth=0,
                  flagged_ancestors=0, model="claude", timeout=300):
    """LLM call: triage an invalid belief into a repair pattern."""
    prompt = TRIAGE_PROMPT.format(
        belief_context=belief_context,
        review_comment=review_comment,
        depth=depth,
        flagged_ancestors=flagged_ancestors,
    )
    response = invoke_model(prompt, model=model, timeout=timeout)
    return parse_triage_response(response)


def soften_belief(belief_text, antecedents_text, model="claude", timeout=300):
    """LLM call: produce a softened version of an overstated belief."""
    prompt = SOFTEN_PROMPT.format(
        belief_text=belief_text,
        antecedents=antecedents_text,
    )
    response = invoke_model(prompt, model=model, timeout=timeout)
    return parse_soften_response(response)


def _compute_depth(node_id, nodes, memo=None):
    """Compute derivation depth: 0 for premises, max(antecedent depths)+1."""
    if memo is None:
        memo = {}
    if node_id in memo:
        return memo[node_id]
    node = nodes.get(node_id)
    if not node or not node.get("justifications"):
        memo[node_id] = 0
        return 0
    memo[node_id] = 0
    max_d = 0
    for j in node["justifications"]:
        for a in j.get("antecedents", []):
            max_d = max(max_d, _compute_depth(a, nodes, memo))
    memo[node_id] = max_d + 1
    return max_d + 1


def _format_antecedents(node, nodes):
    """Format antecedent texts for soften prompt."""
    lines = []
    for j in node.get("justifications", []):
        for ant_id in j.get("antecedents", []):
            ant = nodes.get(ant_id)
            if ant:
                lines.append(f"- {ant_id}: {ant.get('text', '')}")
    return "\n".join(lines) if lines else "(no antecedents)"


def _do_search_and_link(belief_id, node, nodes, comment, model, timeout,
                        db_path, dry_run, search_fn):
    """Execute Pattern 1: search-and-link for a single belief."""
    from . import api

    belief_context = format_belief_for_review(belief_id, nodes)

    claim = extract_smuggled_claim(belief_context, comment,
                                   model=model, timeout=timeout)
    if not claim:
        return "extraction_failed", claim, [], ""

    existing_ants = set()
    for j in node.get("justifications", []):
        existing_ants.update(j.get("antecedents", []))

    raw = search_fn(claim, format="json", db_path=db_path)
    try:
        search_results = json.loads(raw) if isinstance(raw, str) else raw
    except (json.JSONDecodeError, TypeError):
        search_results = []

    candidates = []
    for sr in search_results:
        sid = sr.get("id", "")
        if sid == belief_id or sid in existing_ants:
            continue
        if sr.get("truth_value") != "IN":
            continue
        sr_node = nodes.get(sid)
        if sr_node and sr_node.get("justifications"):
            continue
        candidates.append({"id": sid, "text": sr.get("text", "")})

    if not candidates:
        return "no_candidates", claim, [], ""

    match = find_matching_premises(claim, candidates,
                                   model=model, timeout=timeout)
    matched_ids = match.get("matched_ids", [])

    if not matched_ids:
        return "no_match", claim, [], match.get("rationale", "")

    if not dry_run:
        first_just = node.get("justifications", [{}])[0]
        original_ants = first_just.get("antecedents", [])
        original_outlist = first_just.get("outlist", [])
        new_ants = list(original_ants) + matched_ids
        api.add_justification(
            belief_id,
            sl=",".join(new_ants),
            unless=",".join(original_outlist) if original_outlist else "",
            label=f"repair: linked {claim[:50]}",
            db_path=db_path,
        )

    return "linked", claim, matched_ids, match.get("rationale", "")


def repair_beliefs(review_results, nodes, model="claude",
                     timeout=300, db_path=None, dry_run=False,
                     search_fn=None):
    """Orchestrate triage and repair for invalid beliefs.

    Triages each invalid belief into search_and_link, soften, abandon, or
    research, then executes the appropriate pattern.

    Returns list of repair result dicts.
    """
    from . import api

    if search_fn is None:
        search_fn = lambda query, **kw: api.search(query, **kw)

    invalid = [r for r in review_results if not r.get("valid", True)]
    invalid_ids = {r["id"] for r in invalid}
    depth_memo = {}
    results = []

    for r in invalid:
        belief_id = r["id"]
        result = {
            "id": belief_id,
            "pattern": None,
            "status": "error",
            "rationale": None,
            "smuggled_claim": None,
            "matched_premises": [],
            "softened_text": None,
            "error": None,
        }

        try:
            node = nodes.get(belief_id)
            if not node:
                result["error"] = "belief not found in network"
                results.append(result)
                continue

            belief_context = format_belief_for_review(belief_id, nodes)
            comment = r.get("comment", "")
            depth = _compute_depth(belief_id, nodes, depth_memo)

            flagged_ancestors = 0
            for j in node.get("justifications", []):
                for ant_id in j.get("antecedents", []):
                    if ant_id in invalid_ids:
                        flagged_ancestors += 1

            print(f"  Triaging {belief_id} (depth={depth})...",
                  file=sys.stderr)
            triage = triage_belief(belief_context, comment,
                                   depth=depth,
                                   flagged_ancestors=flagged_ancestors,
                                   model=model, timeout=timeout)
            pattern = triage.get("pattern", "")
            result["pattern"] = pattern
            result["rationale"] = triage.get("rationale", "")

            if not pattern:
                result["status"] = "triage_failed"
                results.append(result)
                continue

            if pattern == "search_and_link":
                print(f"  Pattern: search-and-link for {belief_id}...",
                      file=sys.stderr)
                status, claim, matched, rationale = _do_search_and_link(
                    belief_id, node, nodes, comment, model, timeout,
                    db_path, dry_run, search_fn,
                )
                result["status"] = status
                result["smuggled_claim"] = claim
                result["matched_premises"] = matched
                if rationale:
                    result["rationale"] = rationale
                if not dry_run and status == "linked":
                    api.set_metadata(belief_id, "repair_action", "search_and_link", db_path=db_path)

            elif pattern == "soften":
                print(f"  Pattern: soften for {belief_id}...",
                      file=sys.stderr)
                ant_text = _format_antecedents(node, nodes)
                soften_result = soften_belief(
                    node.get("text", ""), ant_text,
                    model=model, timeout=timeout,
                )
                softened = soften_result.get("softened_text", "")
                if not softened:
                    result["status"] = "soften_failed"
                    results.append(result)
                    continue
                result["softened_text"] = softened
                result["rationale"] = soften_result.get("rationale", "")
                if not dry_run:
                    sup = api.supersede_with_text(belief_id, softened, db_path=db_path)
                    result["new_id"] = sup["new_id"]
                    api.set_metadata(sup["new_id"], "repair_action", "softened", db_path=db_path)
                result["status"] = "softened"

            elif pattern == "abandon":
                print(f"  Pattern: abandon for {belief_id}...",
                      file=sys.stderr)
                if not dry_run:
                    api.retract_node(
                        belief_id,
                        reason=f"repair: abandoned — {triage.get('rationale', '')}",
                        db_path=db_path,
                    )
                    api.set_metadata(belief_id, "repair_action", "abandoned", db_path=db_path)
                result["status"] = "abandoned"

            elif pattern == "research":
                print(f"  Pattern: research for {belief_id}...",
                      file=sys.stderr)
                if not dry_run:
                    api.set_metadata(
                        belief_id, "repair_research",
                        triage.get("rationale", ""),
                        db_path=db_path,
                    )
                    api.set_metadata(belief_id, "repair_action", "research", db_path=db_path)
                result["status"] = "needs_research"

        except Exception as exc:
            result["error"] = str(exc)

        results.append(result)

    return results


research_beliefs = repair_beliefs


def repair_smuggled_beliefs(review_results, nodes, model="claude",
                            timeout=300, db_path=None, dry_run=False,
                            search_fn=None):
    """Orchestrate search-and-link repair for invalid beliefs.

    Args:
        review_results: list of review dicts with valid=False
        nodes: full nodes dict from export_network()
        model: LLM model for extract/match calls
        timeout: LLM timeout
        db_path: database path for add_justification
        dry_run: if True, report without applying
        search_fn: callable(query, format, db_path) for search (injectable for tests)

    Returns:
        list of repair result dicts
    """
    from . import api

    if search_fn is None:
        search_fn = lambda query, **kw: api.search(query, **kw)

    invalid = [r for r in review_results if not r.get("valid", True)]
    repairs = []

    for r in invalid:
        belief_id = r["id"]
        result = {
            "id": belief_id,
            "status": "error",
            "smuggled_claim": None,
            "matched_premises": [],
            "rationale": None,
            "error": None,
        }

        try:
            node = nodes.get(belief_id)
            if not node:
                result["error"] = "belief not found in network"
                repairs.append(result)
                continue

            belief_context = format_belief_for_review(belief_id, nodes)
            comment = r.get("comment", "")

            print(f"  Extracting smuggled claim for {belief_id}...",
                  file=sys.stderr)
            claim = extract_smuggled_claim(belief_context, comment,
                                           model=model, timeout=timeout)
            result["smuggled_claim"] = claim

            if not claim:
                result["status"] = "extraction_failed"
                repairs.append(result)
                continue

            existing_ants = set()
            for j in node.get("justifications", []):
                existing_ants.update(j.get("antecedents", []))

            print(f"  Searching for: {claim[:80]}...", file=sys.stderr)
            raw = search_fn(claim, format="json", db_path=db_path)
            try:
                search_results = json.loads(raw) if isinstance(raw, str) else raw
            except (json.JSONDecodeError, TypeError):
                search_results = []

            candidates = []
            for sr in search_results:
                sid = sr.get("id", "")
                if sid == belief_id:
                    continue
                if sid in existing_ants:
                    continue
                if sr.get("truth_value") != "IN":
                    continue
                sr_node = nodes.get(sid)
                if sr_node and sr_node.get("justifications"):
                    continue
                candidates.append({"id": sid, "text": sr.get("text", "")})

            if not candidates:
                result["status"] = "no_candidates"
                repairs.append(result)
                continue

            print(f"  Matching against {len(candidates)} candidate(s)...",
                  file=sys.stderr)
            match = find_matching_premises(claim, candidates,
                                           model=model, timeout=timeout)
            matched_ids = match.get("matched_ids", [])
            result["rationale"] = match.get("rationale", "")

            if not matched_ids:
                result["status"] = "no_match"
                repairs.append(result)
                continue

            result["matched_premises"] = matched_ids

            if not dry_run:
                first_just = node.get("justifications", [{}])[0]
                original_ants = first_just.get("antecedents", [])
                original_outlist = first_just.get("outlist", [])
                new_ants = list(original_ants) + matched_ids
                api.add_justification(
                    belief_id,
                    sl=",".join(new_ants),
                    unless=",".join(original_outlist) if original_outlist else "",
                    label=f"repair-smuggled: {claim[:60]}",
                    db_path=db_path,
                )

            result["status"] = "repaired"

        except Exception as exc:
            result["error"] = str(exc)

        repairs.append(result)

    return repairs
