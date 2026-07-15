"""Download and publish belief networks on HuggingFace."""

import json
import os
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


HF_BASE = "https://huggingface.co"
DEFAULT_HF_ORG = "EEM-Hub"


def _resolve_token(token: str | None = None) -> str | None:
    """Resolve HuggingFace token: explicit > HF_TOKEN env > cached file."""
    if token:
        return token
    token = os.environ.get("HF_TOKEN")
    if token:
        return token
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.exists():
        return token_path.read_text().strip()
    return None


def resolve_repo_id(repo_id: str) -> str:
    """Resolve a repo identifier, prepending the default org if needed.

    - ``ddia-expert`` → ``EEM-Hub/ddia-expert`` (default org)
    - ``myuser/my-eem`` → ``myuser/my-eem`` (explicit org)
    - ``https://huggingface.co/org/repo`` → ``org/repo`` (URL)

    The default org is ``EEM-Hub`` unless overridden by the
    ``REASONS_HF_ORG`` environment variable.
    """
    repo_id = repo_id.strip().rstrip("/")
    if repo_id.startswith(("http://", "https://")):
        parts = repo_id.split("huggingface.co/", 1)
        if len(parts) == 2:
            return parts[1]
        raise ValueError(f"Not a HuggingFace URL: {repo_id}")
    if "/" in repo_id:
        return repo_id
    default_org = os.environ.get("REASONS_HF_ORG", DEFAULT_HF_ORG)
    return f"{default_org}/{repo_id}"


def _parse_repo_id(repo_id: str) -> str:
    """Extract user/repo from a repo ID or HuggingFace URL.

    Delegates to resolve_repo_id for default org handling.
    """
    return resolve_repo_id(repo_id)


def download_network(repo_id: str, token: str | None = None) -> str:
    """Download network.json from a HuggingFace repo.

    Args:
        repo_id: HuggingFace repo ID (user/repo, bare name, or full URL)
        token: Optional auth token (falls back to HF_TOKEN env or cached token)

    Returns:
        JSON string of the network
    """
    parsed_id = _parse_repo_id(repo_id)
    url = f"{HF_BASE}/{parsed_id}/resolve/main/network.json"

    resolved_token = _resolve_token(token)
    headers = {}
    if resolved_token:
        headers["Authorization"] = f"Bearer {resolved_token}"

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=30) as resp:
            return resp.read().decode("utf-8")
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                f"Authentication required for {parsed_id}. "
                "Run 'huggingface-cli login' or pass --token."
            ) from e
        if e.code == 404:
            raise RuntimeError(
                f"Repository or file not found: {parsed_id}/network.json"
            ) from e
        raise


def create_repo(repo_id: str, token: str, private: bool = False) -> str:
    """Create a HuggingFace repo if it doesn't exist.

    Returns the resolved repo_id. Silently succeeds if the repo already exists.
    """
    parsed_id = resolve_repo_id(repo_id)
    org, name = parsed_id.split("/", 1)

    payload = json.dumps({
        "name": name,
        "organization": org,
        "private": private,
        "type": "model",
    }).encode()

    req = Request(
        f"{HF_BASE}/api/repos/create",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(req, timeout=30) as resp:
            resp.read()
    except HTTPError as e:
        if e.code == 409:
            pass  # repo already exists
        elif e.code == 401:
            raise RuntimeError(
                "Authentication failed. Run 'huggingface-cli login' or pass --token."
            ) from e
        else:
            raise RuntimeError(
                f"Failed to create repo {parsed_id}: HTTP {e.code}"
            ) from e

    return parsed_id


def upload_file(
    repo_id: str,
    path_in_repo: str,
    content: bytes,
    token: str,
) -> None:
    """Upload a file to a HuggingFace repo.

    Uses the HuggingFace Hub upload API (single-file upload via PUT).
    """
    parsed_id = resolve_repo_id(repo_id)
    url = f"{HF_BASE}/api/models/{parsed_id}/upload/main/{path_in_repo}"

    req = Request(
        url,
        data=content,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/octet-stream",
        },
        method="PUT",
    )
    try:
        with urlopen(req, timeout=60) as resp:
            resp.read()
    except HTTPError as e:
        if e.code == 401:
            raise RuntimeError(
                "Authentication failed. Run 'huggingface-cli login' or pass --token."
            ) from e
        raise RuntimeError(
            f"Failed to upload {path_in_repo} to {parsed_id}: HTTP {e.code}"
        ) from e
