"""Verify beliefs against their source documents.

Reads the actual source file for each belief, sends it with the claim to
an LLM, and returns CONFIRMED/STALE/PARTIAL/INCONCLUSIVE verdicts with
verbatim quotes.  Designed for rigorous source-checking, not narrative
synthesis (see report_belief.py for that).
"""

import json
import re
from pathlib import Path

MAX_SOURCE_CHARS = 30_000

VERIFY_PROMPT = """\
You are verifying whether beliefs in a knowledge base are supported by their \
source documents.

For each belief below, the source document text is provided. Evaluate whether \
the source document actually states or clearly implies the claim.

For each belief, return:
- **CONFIRMED** — the source document clearly supports this claim
- **STALE** — the source document does not support or contradicts this claim
- **PARTIAL** — the source partially supports the claim but key details differ
- **INCONCLUSIVE** — insufficient source material to determine

Return ONLY a JSON object mapping each belief ID to an object with "verdict", \
"reason", and "quote" (the most relevant quote from the source, or null):

```json
{{
  "belief-id": {{
    "verdict": "CONFIRMED",
    "reason": "The source explicitly states this figure",
    "quote": "exact quote from the source document"
  }}
}}
```

Rules:
- Be rigorous: a claim that sounds reasonable but isn't in the source is NOT confirmed.
- The quote must be copied verbatim from the source document.
- If the source is unavailable, verdict must be INCONCLUSIVE.
- Focus on what the source document says, not general knowledge.

## Beliefs to verify

{beliefs}"""


def read_source(source, db_path=None):
    """Read a source file, resolving the path via check_stale infrastructure.

    Returns the file content (truncated to MAX_SOURCE_CHARS) or None.
    """
    from .check_stale import resolve_source_path

    db_dir = Path(db_path).parent if db_path else None
    path = resolve_source_path(source, db_dir=db_dir)
    if not path or not path.exists():
        return None

    try:
        content = path.read_text(errors="replace")
    except (OSError, UnicodeDecodeError):
        return None

    if len(content) > MAX_SOURCE_CHARS:
        content = content[:MAX_SOURCE_CHARS] + "\n\n[... truncated ...]"
    return content


def format_belief_for_verify(belief, source_content):
    """Format a belief and its source for the LLM prompt."""
    lines = [f"### `{belief['id']}`"]
    lines.append(f"**Claim:** {belief['text']}")
    if belief.get("source"):
        lines.append(f"**Source file:** {belief['source']}")
    if belief.get("source_url"):
        lines.append(f"**Source URL:** {belief['source_url']}")
    lines.append("")
    lines.append("**Source document content:**")
    lines.append("```")
    lines.append(source_content or "(source not available)")
    lines.append("```")
    return "\n".join(lines)


def parse_verify_response(response):
    """Parse LLM JSON response into verdict dicts.

    Returns dict mapping belief_id -> {"verdict", "reason", "quote"}.
    """
    if not response:
        return {}
    m = re.search(r"\{.*\}", response, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except (json.JSONDecodeError, TypeError):
        return {}
    results = {}
    for k, v in data.items():
        if not isinstance(v, dict) or "verdict" not in v:
            continue
        raw_verdict = v["verdict"]
        if not isinstance(raw_verdict, str):
            continue
        results[k] = {
            "verdict": raw_verdict.upper(),
            "reason": v.get("reason", ""),
            "quote": v.get("quote"),
        }
    return results


def verify_beliefs(beliefs_with_sources, model="claude", timeout=120):
    """Call LLM to verify beliefs against their sources.

    Args:
        beliefs_with_sources: list of (belief_dict, source_content_or_None)
        model: LLM model identifier
        timeout: LLM timeout in seconds

    Returns dict mapping belief_id -> {"verdict", "reason", "quote"}.
    """
    from .llm import invoke_model

    beliefs_text = "\n\n---\n\n".join(
        format_belief_for_verify(b, src)
        for b, src in beliefs_with_sources
    )
    prompt = VERIFY_PROMPT.format(beliefs=beliefs_text)
    response = invoke_model(prompt, model=model, timeout=timeout)
    return parse_verify_response(response)
