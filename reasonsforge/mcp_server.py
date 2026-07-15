"""MCP server for reasonsforge — exposes belief network tools over stdio.

Usage:
    reasons-mcp                              # auto-discover reasons.db
    REASONSFORGE_DB=/path/to/reasons.db reasons-mcp  # explicit path

Add to Claude Code:
    claude mcp add reasons -- reasons-mcp
    claude mcp add reasons -e REASONSFORGE_DB=/path/to/reasons.db -- reasons-mcp
"""

import json
import os

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    raise ImportError(
        "The 'mcp' package is required for the MCP server. "
        "Install with: pip install 'reasonsforge[mcp]'"
    )

from reasonsforge import api

mcp = FastMCP("reasonsforge")

_db: str | None = None


def _find_db() -> str:
    """Walk up from cwd looking for reasons.db, matching CLI behavior."""
    if os.environ.get("REASONSFORGE_DB"):
        return os.environ["REASONSFORGE_DB"]
    d = os.getcwd()
    while True:
        candidate = os.path.join(d, "reasons.db")
        if os.path.exists(candidate):
            return candidate
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return api.DEFAULT_DB


def _get_db() -> str:
    """Return the resolved database path, discovering lazily if needed."""
    global _db
    if _db is None:
        _db = _find_db()
    return _db


# --- Tier 1: Core ---


@mcp.tool()
def search(query: str, output_format: str = "markdown", depth: int = 1) -> str:
    """Search beliefs by text with neighbor expansion.

    Args:
        query: Search terms (matches all terms in any order)
        output_format: Output format — "markdown", "json", or "minimal"
        depth: Hops to expand along justification chains (default 1)
    """
    return api.search(query, db_path=_get_db(), format=output_format, depth=depth)


@mcp.tool()
def show(node_id: str) -> str:
    """Get full details for a belief: text, truth value, justifications, and dependents.

    Args:
        node_id: The belief identifier (e.g. "ansible-is-declarative")
    """
    try:
        return json.dumps(api.show_node(node_id, db_path=_get_db()), indent=2)
    except KeyError:
        return json.dumps({"error": f"Node '{node_id}' not found"})


@mcp.tool()
def explain(node_id: str) -> str:
    """Trace why a belief is IN or OUT by walking its justification chain.

    Args:
        node_id: The belief identifier to explain
    """
    try:
        return json.dumps(api.explain_node(node_id, db_path=_get_db()), indent=2)
    except KeyError:
        return json.dumps({"error": f"Node '{node_id}' not found"})


@mcp.tool()
def list_beliefs(status: str = "", premises_only: bool = False, namespace: str = "") -> str:
    """List beliefs in the network with optional filters.

    Args:
        status: Filter by truth value — "IN", "OUT", or empty for all
        premises_only: Only show premise nodes (no derived beliefs)
        namespace: Filter by namespace prefix
    """
    result = api.list_nodes(
        status=status or None,
        premises_only=premises_only,
        namespace=namespace or None,
        db_path=_get_db(),
    )
    return json.dumps(result, indent=2)


@mcp.tool()
def add(node_id: str, text: str, sl: str = "", unless: str = "", label: str = "") -> str:
    """Add a belief to the network. Without sl, adds as a premise. With sl, adds as derived.

    Args:
        node_id: Identifier for the new belief (e.g. "ansible-is-declarative")
        text: The belief text
        sl: Comma-separated antecedent node IDs for SL justification (empty = premise)
        unless: Comma-separated outlist node IDs (must be OUT for justification to hold)
        label: Optional justification label
    """
    try:
        result = api.add_node(node_id, text, sl=sl, unless=unless, label=label, db_path=_get_db())
        return json.dumps(result, indent=2)
    except (ValueError, KeyError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def retract(node_id: str, reason: str = "") -> str:
    """Retract a belief, cascading OUT to all dependents.

    Args:
        node_id: The belief to retract
        reason: Why this belief is being retracted
    """
    try:
        return json.dumps(api.retract_node(node_id, reason=reason, db_path=_get_db()), indent=2)
    except KeyError:
        return json.dumps({"error": f"Node '{node_id}' not found"})


@mcp.tool()
def assert_belief(node_id: str) -> str:
    """Assert a retracted belief back to IN, cascading restoration to dependents.

    Args:
        node_id: The belief to assert
    """
    try:
        return json.dumps(api.assert_node(node_id, db_path=_get_db()), indent=2)
    except KeyError:
        return json.dumps({"error": f"Node '{node_id}' not found"})


# --- Tier 2: Reasoning ---


@mcp.tool()
def what_if(node_id: str, action: str = "retract") -> str:
    """Simulate retracting or asserting a belief without modifying the database.

    Shows the cascade: which beliefs would change truth values.

    Args:
        node_id: The belief to simulate
        action: "retract" or "assert"
    """
    try:
        if action == "assert":
            result = api.what_if_assert(node_id, db_path=_get_db())
        else:
            result = api.what_if_retract(node_id, db_path=_get_db())
        return json.dumps(result, indent=2)
    except KeyError:
        return json.dumps({"error": f"Node '{node_id}' not found"})


@mcp.tool()
def add_justification(node_id: str, sl: str = "", unless: str = "", label: str = "") -> str:
    """Add an additional justification to an existing belief.

    Can bring an OUT belief back to IN if the new justification is satisfied.

    Args:
        node_id: The belief to add a justification to
        sl: Comma-separated antecedent node IDs
        unless: Comma-separated outlist node IDs
        label: Optional justification label
    """
    try:
        result = api.add_justification(node_id, sl=sl, unless=unless, label=label, db_path=_get_db())
        return json.dumps(result, indent=2)
    except (ValueError, KeyError) as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def nogood(node_ids: list[str]) -> str:
    """Record a contradiction between beliefs and trigger backtracking.

    The TMS will retract the weakest premise to resolve the contradiction.

    Args:
        node_ids: List of belief IDs that form a contradiction
    """
    try:
        return json.dumps(api.add_nogood(node_ids, db_path=_get_db()), indent=2)
    except KeyError as e:
        return json.dumps({"error": str(e)})


@mcp.tool()
def trace(node_id: str) -> str:
    """Find all root premises a belief ultimately depends on.

    Args:
        node_id: The belief to trace
    """
    try:
        return json.dumps(api.trace_assumptions(node_id, db_path=_get_db()), indent=2)
    except KeyError:
        return json.dumps({"error": f"Node '{node_id}' not found"})


@mcp.tool()
def compact(budget: int = 500) -> str:
    """Get a token-budgeted summary of the entire belief network.

    Args:
        budget: Maximum token budget for the summary
    """
    return api.compact(budget=budget, db_path=_get_db())


# --- Tier 3: Data management ---


@mcp.tool()
def status() -> str:
    """Get a network overview: all nodes with truth values and counts."""
    return json.dumps(api.get_status(db_path=_get_db()), indent=2)


@mcp.tool()
def list_gated() -> str:
    """Find OUT beliefs that are blocked by active outlist gates.

    These are beliefs that would come IN if their blocker was retracted.
    """
    return json.dumps(api.list_gated(db_path=_get_db()), indent=2)


@mcp.tool()
def export_markdown() -> str:
    """Export the entire belief network as markdown (beliefs.md format)."""
    return api.export_markdown(db_path=_get_db())


@mcp.tool()
def topics(limit: int = 20) -> str:
    """Extract topic clusters from belief identifiers.

    Args:
        limit: Maximum number of topics to return
    """
    return json.dumps(api.topics(limit=limit, db_path=_get_db()), indent=2)


def main():
    global _db
    _db = _find_db()
    mcp.run()


if __name__ == "__main__":
    main()
