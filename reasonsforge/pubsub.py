"""In-process publish/subscribe for TMS truth-value change events.

Subscribers receive batched events after each _with_network write
transaction completes (after save). Events are never emitted for
intermediate cascade states -- only the final stable state is visible.

Events carry what changed (old/new truth values) but not why — the
_with_network hook sees only before/after snapshots, so cause and
cascade_root metadata are not available at this level.
"""

import logging
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ChangeEvent:
    """A single truth-value change observed during a write transaction.

    Does not include cause or cascade_root — the _with_network hook only
    sees before/after snapshots, not which operation triggered the change.
    """
    type: str
    node_id: str
    old_truth_value: str | None
    new_truth_value: str | None
    db_path: str


Subscriber = Callable[[list[ChangeEvent]], None]

_subscribers: dict[str | None, list[Subscriber]] = {}


def subscribe(callback: Subscriber, db_path: str | None = None) -> None:
    """Register a callback to receive change events.

    Args:
        callback: Called with a list of ChangeEvent after each write
                  transaction that changes truth values.
        db_path: If provided, only receive events for this database.
                 If None, receive events from all databases.
    """
    _subscribers.setdefault(db_path, []).append(callback)


def unsubscribe(callback: Subscriber, db_path: str | None = None) -> None:
    """Remove a previously registered callback."""
    subs = _subscribers.get(db_path, [])
    try:
        subs.remove(callback)
    except ValueError:
        pass
    if not subs:
        _subscribers.pop(db_path, None)


def has_subscribers(db_path: str | None = None) -> bool:
    """Check if any subscribers exist (global or for this db_path)."""
    if _subscribers.get(None):
        return True
    if db_path is not None and _subscribers.get(db_path):
        return True
    return False


def publish(events: list[ChangeEvent], db_path: str) -> None:
    """Dispatch events to all relevant subscribers.

    Called by _with_network.__exit__ after save completes.
    Exceptions in subscribers are logged but do not propagate.
    """
    if not events:
        return

    targets = list(_subscribers.get(None, []))
    targets.extend(_subscribers.get(db_path, []))

    for subscriber in targets:
        try:
            subscriber(events)
        except Exception:
            logger.exception("pubsub subscriber raised an exception")


def compute_changes(
    before: dict[str, str],
    after: dict[str, str],
    db_path: str,
) -> list[ChangeEvent]:
    """Diff two truth-value snapshots and produce ChangeEvents."""
    events = []

    for nid, new_tv in after.items():
        old_tv = before.get(nid)
        if old_tv != new_tv:
            events.append(ChangeEvent(
                type="node_changed",
                node_id=nid,
                old_truth_value=old_tv,
                new_truth_value=new_tv,
                db_path=db_path,
            ))

    for nid in before:
        if nid not in after:
            events.append(ChangeEvent(
                type="node_changed",
                node_id=nid,
                old_truth_value=before[nid],
                new_truth_value=None,
                db_path=db_path,
            ))

    return events


def clear() -> None:
    """Remove all subscribers. Intended for test teardown."""
    _subscribers.clear()
