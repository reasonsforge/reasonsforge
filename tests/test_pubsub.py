"""Tests for the pubsub interface — truth-value change events."""

import pytest

from reasonsforge import api
from reasonsforge.pubsub import (
    ChangeEvent,
    clear,
    compute_changes,
    has_subscribers,
    publish,
    subscribe,
    unsubscribe,
)


@pytest.fixture(autouse=True)
def cleanup_subscribers():
    yield
    clear()


class TestComputeChanges:

    def test_no_changes(self):
        before = {"a": "IN", "b": "OUT"}
        after = {"a": "IN", "b": "OUT"}
        assert compute_changes(before, after, "test.db") == []

    def test_truth_value_change(self):
        before = {"a": "IN"}
        after = {"a": "OUT"}
        events = compute_changes(before, after, "test.db")
        assert len(events) == 1
        assert events[0].node_id == "a"
        assert events[0].old_truth_value == "IN"
        assert events[0].new_truth_value == "OUT"
        assert events[0].type == "node_changed"
        assert events[0].db_path == "test.db"

    def test_new_node(self):
        before = {}
        after = {"a": "IN"}
        events = compute_changes(before, after, "test.db")
        assert len(events) == 1
        assert events[0].node_id == "a"
        assert events[0].old_truth_value is None
        assert events[0].new_truth_value == "IN"

    def test_deleted_node(self):
        before = {"a": "IN"}
        after = {}
        events = compute_changes(before, after, "test.db")
        assert len(events) == 1
        assert events[0].node_id == "a"
        assert events[0].old_truth_value == "IN"
        assert events[0].new_truth_value is None

    def test_multiple_changes(self):
        before = {"a": "IN", "b": "IN", "c": "OUT"}
        after = {"a": "OUT", "b": "IN", "c": "IN"}
        events = compute_changes(before, after, "test.db")
        changed_ids = {e.node_id for e in events}
        assert changed_ids == {"a", "c"}
        assert len(events) == 2

    def test_new_node_out(self):
        before = {}
        after = {"a": "OUT"}
        events = compute_changes(before, after, "test.db")
        assert len(events) == 1
        assert events[0].old_truth_value is None
        assert events[0].new_truth_value == "OUT"


class TestSubscriberRegistry:

    def test_subscribe_and_has_subscribers(self):
        assert not has_subscribers()
        subscribe(lambda events: None)
        assert has_subscribers()

    def test_unsubscribe(self):
        cb = lambda events: None  # noqa: E731
        subscribe(cb)
        assert has_subscribers()
        unsubscribe(cb)
        assert not has_subscribers()

    def test_global_subscriber(self):
        received = []
        subscribe(lambda events: received.extend(events))
        events = [ChangeEvent("node_changed", "a", None, "IN", "db1.db")]
        publish(events, "db1.db")
        assert len(received) == 1
        assert received[0].node_id == "a"

    def test_db_scoped_subscriber(self):
        received = []
        subscribe(lambda events: received.extend(events), db_path="my.db")
        publish(
            [ChangeEvent("node_changed", "a", None, "IN", "other.db")],
            "other.db",
        )
        assert len(received) == 0
        publish(
            [ChangeEvent("node_changed", "b", None, "IN", "my.db")],
            "my.db",
        )
        assert len(received) == 1
        assert received[0].node_id == "b"

    def test_clear(self):
        subscribe(lambda events: None)
        subscribe(lambda events: None, db_path="x.db")
        assert has_subscribers()
        clear()
        assert not has_subscribers()

    def test_multiple_subscribers(self):
        counts = [0, 0]
        subscribe(lambda events: counts.__setitem__(0, counts[0] + 1))
        subscribe(lambda events: counts.__setitem__(1, counts[1] + 1))
        publish(
            [ChangeEvent("node_changed", "a", None, "IN", "t.db")],
            "t.db",
        )
        assert counts == [1, 1]

    def test_publish_empty_list_skipped(self):
        called = []
        subscribe(lambda events: called.append(True))
        publish([], "t.db")
        assert called == []

    def test_subscriber_exception_logged(self):
        good_received = []

        def bad_sub(events):
            raise RuntimeError("boom")

        def good_sub(events):
            good_received.extend(events)

        subscribe(bad_sub)
        subscribe(good_sub)
        publish(
            [ChangeEvent("node_changed", "a", None, "IN", "t.db")],
            "t.db",
        )
        assert len(good_received) == 1


class TestApiIntegration:

    def test_add_node_emits_event(self, tmp_path):
        db = str(tmp_path / "test.db")
        received = []
        subscribe(lambda events: received.extend(events))
        api.add_node("p1", "A premise", db_path=db)
        assert len(received) == 1
        assert received[0].node_id == "p1"
        assert received[0].old_truth_value is None
        assert received[0].new_truth_value == "IN"

    def test_retract_cascade_emits_events(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)
        api.add_node("c", "Derived C", sl="b", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.retract_node("a", db_path=db)

        changed_ids = {e.node_id for e in received}
        assert "a" in changed_ids
        assert "b" in changed_ids
        assert "c" in changed_ids
        for e in received:
            assert e.new_truth_value == "OUT"
            assert e.old_truth_value == "IN"

    def test_assert_emits_events(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)
        api.retract_node("a", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.assert_node("a", db_path=db)

        changed_ids = {e.node_id for e in received}
        assert "a" in changed_ids
        assert "b" in changed_ids
        for e in received:
            assert e.old_truth_value == "OUT"
            assert e.new_truth_value == "IN"

    def test_add_justification_emits_event(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Node B", db_path=db)
        api.retract_node("b", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.add_justification("b", sl="a", db_path=db)

        b_events = [e for e in received if e.node_id == "b"]
        assert len(b_events) == 1
        assert b_events[0].old_truth_value == "OUT"
        assert b_events[0].new_truth_value == "IN"

    def test_propagate_emits_events(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Derived B", sl="a", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.propagate(db_path=db)

        # propagate on a consistent network may emit nothing
        # just verify no crash and events are well-formed
        for e in received:
            assert e.type == "node_changed"

    def test_no_event_when_no_change(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("p1", "A premise", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.set_metadata("p1", "key", "value", db_path=db)
        assert len(received) == 0

    def test_no_snapshot_without_subscribers(self, tmp_path):
        db = str(tmp_path / "test.db")
        assert not has_subscribers()
        api.add_node("p1", "A premise", db_path=db)
        # just verify it works with no subscribers, no crash

    def test_subscriber_exception_does_not_break_save(self, tmp_path):
        db = str(tmp_path / "test.db")

        def bad_sub(events):
            raise RuntimeError("boom")

        subscribe(bad_sub)
        api.add_node("p1", "A premise", db_path=db)

        node = api.show_node("p1", db_path=db)
        assert node["truth_value"] == "IN"

    def test_events_after_save(self, tmp_path):
        db = str(tmp_path / "test.db")
        db_state_at_callback = {}

        def check_db(events):
            for e in events:
                node = api.show_node(e.node_id, db_path=db)
                db_state_at_callback[e.node_id] = node["truth_value"]

        subscribe(check_db)
        api.add_node("p1", "A premise", db_path=db)

        assert db_state_at_callback["p1"] == "IN"

    def test_event_shape(self, tmp_path):
        db = str(tmp_path / "test.db")
        received = []
        subscribe(lambda events: received.extend(events))
        api.add_node("p1", "A premise", db_path=db)

        e = received[0]
        assert isinstance(e, ChangeEvent)
        assert e.type == "node_changed"
        assert isinstance(e.node_id, str)
        assert e.db_path == db

    def test_challenge_emits_events(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("target", "A belief", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.challenge("target", reason="disputed", db_path=db)

        changed_ids = {e.node_id for e in received}
        assert "target" in changed_ids

    def test_global_and_scoped_both_receive(self, tmp_path):
        db = str(tmp_path / "test.db")
        global_received = []
        scoped_received = []
        subscribe(lambda events: global_received.extend(events))
        subscribe(lambda events: scoped_received.extend(events), db_path=db)

        api.add_node("p1", "A premise", db_path=db)
        assert len(global_received) == 1
        assert len(scoped_received) == 1

    def test_defend_emits_events(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("target", "A belief", db_path=db)
        api.challenge("target", reason="disputed",
                      challenge_id="ch-1", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.defend("target", "ch-1", reason="evidence",
                   db_path=db)

        changed_ids = {e.node_id for e in received}
        assert "target" in changed_ids
        target_events = [e for e in received if e.node_id == "target"]
        assert target_events[0].new_truth_value == "IN"

    def test_add_nogood_emits_events(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("a", "Premise A", db_path=db)
        api.add_node("b", "Premise B", db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))
        api.add_nogood(["a", "b"], db_path=db)

        changed_ids = {e.node_id for e in received}
        # add_nogood backtracks by retracting one of the conflicting nodes
        assert len(changed_ids) >= 1

    def test_no_events_on_exception(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.init_db(db_path=db)

        received = []
        subscribe(lambda events: received.extend(events))

        # retract_node raises KeyError for nonexistent nodes;
        # the exception must prevent both save and event emission
        with pytest.raises(KeyError):
            api.retract_node("nonexistent", db_path=db)

        assert len(received) == 0
