"""Metadata helpers for export formats.

Builds a consistent metadata dict used by JSON export, markdown export,
and SQLite/PostgreSQL storage. Single source of truth for field names
and the generator string.
"""

from datetime import datetime, timezone


SCHEMA_VERSION = "1.0"


def _get_generator() -> str:
    """Return the generator string, e.g. 'reasonsforge/0.40.0'."""
    try:
        from importlib.metadata import version
        v = version("reasonsforge")
    except Exception:
        v = "unknown"
    return f"reasonsforge/{v}"


def build_meta(
    project_name: str = "",
    node_count: int = 0,
    created_at: str = "",
) -> dict:
    """Build a metadata dict for export/storage.

    Args:
        project_name: Name of the belief network/project.
        node_count: Total nodes at time of export.
        created_at: ISO 8601 timestamp of initial creation (preserved if set).

    Returns:
        Dict with schema_version, project_name, created_at, updated_at,
        node_count, and generator fields.

    Example:
        >>> meta = build_meta("my-project", node_count=42)
        >>> meta["schema_version"]
        '1.0'
        >>> "reasonsforge/" in meta["generator"]
        True
    """
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return {
        "schema_version": SCHEMA_VERSION,
        "project_name": project_name,
        "created_at": created_at or now,
        "updated_at": now,
        "node_count": node_count,
        "generator": _get_generator(),
    }
