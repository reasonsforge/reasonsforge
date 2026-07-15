"""Tests for monotonic nogood ID generation (issue #26).

The bug: nogood IDs were derived from len(self.nogoods) + 1, so deleting
a nogood and adding a new one would reuse the deleted ID. The fix uses a
persisted monotonic counter (_next_nogood_id) that never decreases.
"""

import json

import pytest

from reasonsforge import api, Nogood
from reasonsforge.network import Network
from reasonsforge.storage import Storage


@pytest.fixture
def db(tmp_path):
    db_path = str(tmp_path / "reasons.db")
    api.init_db(db_path=db_path)
    return db_path


class TestMonotonicCounter:

    def test_ids_increment(self):
        net = Network()
        net.add_node("a", "A")
        net.add_node("b", "B")
        net.add_node("c", "C")
        net.add_node("d", "D")
        net.add_nogood(["a", "b"])
        net.add_nogood(["c", "d"])
        assert net.nogoods[0].id == "nogood-001"
        assert net.nogoods[1].id == "nogood-002"
        assert net._next_nogood_id == 3

    def test_delete_last_does_not_reuse_id(self):
        net = Network()
        net.add_node("a", "A")
        net.add_node("b", "B")
        net.add_node("c", "C")
        net.add_node("d", "D")
        net.add_nogood(["a", "b"])
        net.add_nogood(["c", "d"])
        del net.nogoods[-1]
        net.add_nogood(["a", "c"])
        assert net.nogoods[-1].id == "nogood-003"

    def test_delete_all_does_not_reset(self):
        net = Network()
        net.add_node("a", "A")
        net.add_node("b", "B")
        net.add_nogood(["a", "b"])
        net.nogoods.clear()
        assert net._next_nogood_id == 2
        net.add_nogood(["a", "b"])
        assert net.nogoods[0].id == "nogood-002"


class TestPersistence:

    def test_counter_survives_save_load(self, tmp_path):
        db_path = str(tmp_path / "reasons.db")
        storage = Storage(db_path)
        net = Network()
        net.add_node("a", "A")
        net.add_node("b", "B")
        net.add_nogood(["a", "b"])
        assert net._next_nogood_id == 2
        storage.save(net)
        storage.close()

        storage2 = Storage(db_path)
        loaded = storage2.load()
        assert loaded._next_nogood_id == 2
        storage2.close()

    def test_counter_persists_after_deletion(self, tmp_path):
        db_path = str(tmp_path / "reasons.db")
        storage = Storage(db_path)
        net = Network()
        net.add_node("a", "A")
        net.add_node("b", "B")
        net.add_node("c", "C")
        net.add_node("d", "D")
        net.add_nogood(["a", "b"])
        net.add_nogood(["c", "d"])
        del net.nogoods[-1]
        storage.save(net)
        storage.close()

        storage2 = Storage(db_path)
        loaded = storage2.load()
        assert loaded._next_nogood_id == 3
        loaded.add_nogood(["a", "c"])
        assert loaded.nogoods[-1].id == "nogood-003"
        storage2.close()

    def test_old_db_without_meta_table(self, tmp_path):
        """Databases created before this fix should still load."""
        db_path = str(tmp_path / "reasons.db")
        storage = Storage(db_path)
        net = Network()
        net.add_node("a", "A")
        storage.save(net)
        # Drop the meta table to simulate an old database
        storage.conn.execute("DROP TABLE IF EXISTS network_meta")
        storage.conn.commit()
        storage.close()

        storage2 = Storage(db_path)
        loaded = storage2.load()
        assert loaded._next_nogood_id == 1
        storage2.close()

    def test_old_db_with_nogoods_derives_counter(self, tmp_path):
        """Old DB with existing nogoods must derive counter, not start at 1."""
        db_path = str(tmp_path / "reasons.db")
        storage = Storage(db_path)
        net = Network()
        net.add_node("a", "A")
        net.add_node("b", "B")
        net.add_node("c", "C")
        net.add_node("d", "D")
        net.add_nogood(["a", "b"])
        net.add_nogood(["c", "d"])
        storage.save(net)
        # Drop the table entirely to simulate an old database
        storage.conn.execute("DROP TABLE IF EXISTS network_meta")
        storage.conn.commit()
        storage.close()

        storage2 = Storage(db_path)
        loaded = storage2.load()
        assert loaded._next_nogood_id == 3
        loaded.add_nogood(["a", "c"])
        assert loaded.nogoods[-1].id == "nogood-003"
        # Verify save works after upgrade
        storage2.save(loaded)
        storage2.close()


class TestImportJson:

    def test_import_json_updates_counter(self, db, tmp_path):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)

        json_data = {
            "nodes": {},
            "nogoods": [
                {"id": "nogood-005", "nodes": ["a", "b"], "discovered": "", "resolution": ""},
            ],
            "repos": {},
        }
        json_file = str(tmp_path / "import.json")
        with open(json_file, "w") as f:
            json.dump(json_data, f)

        api.import_json(json_file, db_path=db)
        result = api.add_nogood(["a", "b"], db_path=db)
        assert result["nogood_id"] == "nogood-006"


class TestImportBeliefs:

    def test_import_nogoods_updates_counter(self, db, tmp_path):
        api.add_node("a", "A", db_path=db)
        api.add_node("b", "B", db_path=db)

        beliefs_text = ""
        nogoods_text = """# Nogoods

### nogood-010: a, b
- Affects: a, b
- Discovered: 2026-01-01
"""
        from reasonsforge.import_beliefs import import_into_network
        from reasonsforge.storage import Storage

        storage = Storage(db)
        net = storage.load()
        import_into_network(net, beliefs_text, nogoods_text)
        assert net._next_nogood_id == 11
        storage.save(net)
        storage.close()

        result = api.add_nogood(["a", "b"], db_path=db)
        assert result["nogood_id"] == "nogood-011"
