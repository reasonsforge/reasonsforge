"""HTTP client for agentic-mind-service belief sync."""

import json
import os
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def _resolve_config(
    url: str | None = None,
    agent_id: str | None = None,
    api_key: str | None = None,
) -> tuple[str, str, str]:
    """Resolve service config: explicit params > environment variables."""
    url = url or os.environ.get("MIND_SERVICE_URL", "")
    agent_id = agent_id or os.environ.get("MIND_AGENT_ID", "")
    api_key = api_key or os.environ.get("MIND_API_KEY", "")

    missing = []
    if not url:
        missing.append("url (--url or MIND_SERVICE_URL)")
    if not agent_id:
        missing.append("agent-id (--agent-id or MIND_AGENT_ID)")
    if not api_key:
        missing.append("api-key (--api-key or MIND_API_KEY)")
    if missing:
        raise RuntimeError(f"Missing required config: {', '.join(missing)}")

    return url.rstrip("/"), agent_id, api_key


def fetch_export(url: str, agent_id: str, api_key: str) -> str:
    """GET /api/agents/{agent_id}/export — download full network JSON."""
    endpoint = f"{url}/api/agents/{agent_id}/export"
    req = Request(endpoint, headers={"Authorization": f"Bearer {api_key}"})
    try:
        with urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                "Authentication failed. Check your MIND_API_KEY."
            ) from e
        if e.code == 404:
            raise RuntimeError(
                f"Agent not found: {agent_id}"
            ) from e
        raise


def push_belief(
    url: str, agent_id: str, api_key: str,
    node_id: str, text: str, sl: str = "", source: str = "",
) -> dict:
    """POST /api/agents/{agent_id}/beliefs — push one belief."""
    endpoint = f"{url}/api/agents/{agent_id}/beliefs"
    body = {"node_id": node_id, "text": text}
    if sl:
        body["sl"] = sl
    if source:
        body["source"] = source

    data = json.dumps(body).encode("utf-8")
    req = Request(
        endpoint,
        data=data,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                "Authentication failed. Check your MIND_API_KEY."
            ) from e
        raise
