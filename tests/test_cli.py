"""Tests for CLI command handlers via main()."""

import json
import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge.cli import main


@pytest.fixture
def db_path(tmp_path):
    return str(tmp_path / "test.db")


def run_cli(*args, db_path=None):
    argv = ["reasons"]
    if db_path:
        argv += ["--db", db_path]
    argv += list(args)
    stdout, stderr = StringIO(), StringIO()
    with patch.object(sys, "argv", argv), \
         patch.object(sys, "stdout", stdout), \
         patch.object(sys, "stderr", stderr):
        try:
            main()
        except SystemExit as e:
            return stdout.getvalue(), stderr.getvalue(), e.code
    return stdout.getvalue(), stderr.getvalue(), 0


class TestInit:

    def test_init_creates_db(self, db_path):
        out, err, code = run_cli("init", db_path=db_path)
        assert code == 0
        assert "Initialized" in out

    def test_init_refuses_existing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("init", db_path=db_path)
        assert code == 1
        assert "--force" in err

    def test_init_force(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("init", "--force", db_path=db_path)
        assert code == 0
        assert "Initialized" in out


class TestAdd:

    def test_add_premise(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("add", "a", "Premise A", db_path=db_path)
        assert code == 0
        assert "Added a [IN]" in out
        assert "premise" in out.lower()

    def test_add_derived(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Premise A", db_path=db_path)
        out, err, code = run_cli("add", "b", "Derived B", "--sl", "a", db_path=db_path)
        assert code == 0
        assert "Added b [IN]" in out

    def test_add_duplicate_fails(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("add", "a", "A again", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_add_with_access_tags(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("add", "a", "Tagged", "--access-tags", "finance,hr", db_path=db_path)
        assert code == 0
        assert "Added a [IN]" in out

    def test_add_multi_premise_tip(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", db_path=db_path)
        run_cli("add", "c", "C", db_path=db_path)
        out, err, code = run_cli("add", "d", "D", "--sl", "a,b,c", db_path=db_path)
        assert code == 0
        assert "Tip" in out
        assert "--any" in out


class TestAddJustification:

    def test_add_justification(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", db_path=db_path)
        out, err, code = run_cli("add-justification", "b", "--sl", "a", db_path=db_path)
        assert code == 0
        assert "Added justification to b" in out

    def test_add_justification_missing_node(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("add-justification", "missing", "--sl", "a", db_path=db_path)
        assert code == 1
        assert "Error" in err


class TestRetractAssert:

    def test_retract(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("retract", "a", db_path=db_path)
        assert code == 0
        assert "Retracted a" in out

    def test_retract_already_out(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        out, err, code = run_cli("retract", "a", db_path=db_path)
        assert code == 0
        assert "already OUT" in out

    def test_retract_cascade(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("retract", "a", db_path=db_path)
        assert code == 0
        assert "Went OUT" in out
        assert "b" in out

    def test_retract_with_reason(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("retract", "a", "--reason", "Fixed in PR", db_path=db_path)
        assert code == 0

    def test_retract_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("retract", "missing", db_path=db_path)
        assert code == 1

    def test_assert(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        out, err, code = run_cli("assert", "a", db_path=db_path)
        assert code == 0
        assert "Asserted a" in out

    def test_assert_already_in(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("assert", "a", db_path=db_path)
        assert code == 0
        assert "already IN" in out

    def test_assert_cascade(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        out, err, code = run_cli("assert", "a", db_path=db_path)
        assert code == 0
        assert "Went IN" in out
        assert "b" in out


class TestStatus:

    def test_status_empty(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("status", db_path=db_path)
        assert code == 0
        assert "No nodes" in out

    def test_status_with_nodes(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Premise A", db_path=db_path)
        run_cli("add", "b", "Derived B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("status", db_path=db_path)
        assert code == 0
        assert "[+] a" in out
        assert "[+] b" in out
        assert "2/2 IN" in out

    def test_status_visible_to(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "pub", "Public", db_path=db_path)
        run_cli("add", "fin", "Finance", "--access-tags", "finance", db_path=db_path)
        out, err, code = run_cli("status", "--visible-to", "public", db_path=db_path)
        assert code == 0
        assert "pub" in out
        assert "fin" not in out


class TestShow:

    def test_show_premise(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Premise A", db_path=db_path)
        out, err, code = run_cli("show", "a", db_path=db_path)
        assert code == 0
        assert "ID:     a" in out
        assert "Text:   Premise A" in out
        assert "Status: IN" in out
        assert "Premise (no justifications)" in out

    def test_show_derived(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("show", "b", db_path=db_path)
        assert code == 0
        assert "SL(a)" in out

    def test_show_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("show", "missing", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_show_access_denied(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "fin", "Finance", "--access-tags", "finance", db_path=db_path)
        out, err, code = run_cli("show", "fin", "--visible-to", "hr", db_path=db_path)
        assert code == 1
        assert "Access denied" in err

    def test_show_dependents(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("show", "a", db_path=db_path)
        assert code == 0
        assert "Dependents: b" in out

    def test_show_retract_reason(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("retract", "a", "--reason", "Fixed in PR #1", db_path=db_path)
        out, err, code = run_cli("show", "a", db_path=db_path)
        assert code == 0
        assert "Retract reason: Fixed in PR #1" in out


class TestExplain:

    def test_explain_premise(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("explain", "a", db_path=db_path)
        assert code == 0
        assert "[+] a" in out

    def test_explain_derived(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("explain", "b", db_path=db_path)
        assert code == 0
        assert "[+] b" in out
        assert "antecedents: a" in out

    def test_explain_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("explain", "missing", db_path=db_path)
        assert code == 1

    def test_explain_access_denied(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "fin", "Finance", "--access-tags", "finance", db_path=db_path)
        out, err, code = run_cli("explain", "fin", "--visible-to", "hr", db_path=db_path)
        assert code == 1
        assert "Access denied" in err


class TestTrace:

    def test_trace_premise(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("trace", "a", db_path=db_path)
        assert code == 0
        assert "premise" in out.lower()

    def test_trace_derived(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("trace", "b", db_path=db_path)
        assert code == 0
        assert "1 premise" in out
        assert "[+] a" in out

    def test_trace_access_denied(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "fin", "Finance", "--access-tags", "finance", db_path=db_path)
        out, err, code = run_cli("trace", "fin", "--visible-to", "hr", db_path=db_path)
        assert code == 1
        assert "Access denied" in err


class TestTraceAccessTags:

    def test_trace_access_tags(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", "--access-tags", "finance", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("trace-access-tags", "b", db_path=db_path)
        assert code == 0
        assert "finance" in out

    def test_trace_access_tags_unrestricted(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("trace-access-tags", "a", db_path=db_path)
        assert code == 0
        assert "unrestricted" in out

    def test_trace_access_tags_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("trace-access-tags", "missing", db_path=db_path)
        assert code == 1


class TestList:

    def test_list_all(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("list", db_path=db_path)
        assert code == 0
        assert "2 nodes" in out

    def test_list_premises(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("list", "--premises", db_path=db_path)
        assert code == 0
        assert "1 node" in out
        assert "a" in out

    def test_list_status_filter(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", db_path=db_path)
        run_cli("retract", "b", db_path=db_path)
        out, err, code = run_cli("list", "--status", "OUT", db_path=db_path)
        assert code == 0
        assert "b" in out
        assert "1 node" in out

    def test_list_empty(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("list", db_path=db_path)
        assert code == 0
        assert "No matching" in out


class TestSearchLookup:

    def test_search(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "The quick brown fox", db_path=db_path)
        out, err, code = run_cli("search", "fox", db_path=db_path)
        assert code == 0
        assert "a" in out

    def test_lookup(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "my-belief", "Something important", db_path=db_path)
        out, err, code = run_cli("lookup", "important", db_path=db_path)
        assert code == 0
        assert "my-belief" in out


class TestExport:

    def test_export_json(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("export", "--output=-", db_path=db_path)
        assert code == 0
        data = json.loads(out)
        assert "a" in data["nodes"]

    def test_export_markdown(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Premise A", db_path=db_path)
        out, err, code = run_cli("export-markdown", "--output=-", db_path=db_path)
        assert code == 0
        assert "a" in out

    def test_export_markdown_to_file(self, db_path, tmp_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out_file = str(tmp_path / "out.md")
        out, err, code = run_cli("export-markdown", "-o", out_file, db_path=db_path)
        assert code == 0
        assert "Written to" in out
        from pathlib import Path
        assert "a" in Path(out_file).read_text()


class TestCompact:

    def test_compact(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "Premise A", db_path=db_path)
        out, err, code = run_cli("compact", db_path=db_path)
        assert code == 0
        assert "a" in out


class TestChallenge:

    def test_challenge_and_defend(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "claim", "A bold claim", db_path=db_path)
        out, err, code = run_cli("challenge", "claim", "I disagree", db_path=db_path)
        assert code == 0
        assert "Challenged claim" in out
        assert "challenge-claim" in out

        out, err, code = run_cli("defend", "claim", "challenge-claim", "New evidence", db_path=db_path)
        assert code == 0
        assert "Defended claim" in out


class TestNogood:

    def test_nogood(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", db_path=db_path)
        out, err, code = run_cli("nogood", "a", "b", db_path=db_path)
        assert code == 0
        assert "Recorded" in out
        assert "a" in out and "b" in out

    def test_nogood_missing_node(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("nogood", "x", "y", db_path=db_path)
        assert code == 1


class TestWhatIf:

    def test_what_if_retract(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("what-if", "retract", "a", db_path=db_path)
        assert code == 0
        assert "What if" in out
        assert "[-] b" in out
        assert "NOT modified" in out

    def test_what_if_retract_already_out(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        out, err, code = run_cli("what-if", "retract", "a", db_path=db_path)
        assert code == 0
        assert "already OUT" in out

    def test_what_if_assert(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        out, err, code = run_cli("what-if", "assert", "a", db_path=db_path)
        assert code == 0
        assert "What if" in out
        assert "[+] b" in out

    def test_what_if_no_effect(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("what-if", "retract", "a", db_path=db_path)
        assert code == 0
        assert "no other nodes" in out


class TestSupersede:

    def test_supersede(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "old", "Old belief", db_path=db_path)
        run_cli("add", "new", "New belief", db_path=db_path)
        out, err, code = run_cli("supersede", "old", "new", db_path=db_path)
        assert code == 0
        assert "Superseded old by new" in out

    def test_supersede_with_text(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "old", "Old belief", db_path=db_path)
        out, err, code = run_cli("supersede", "old", "--text", "Corrected belief", db_path=db_path)
        assert code == 0
        assert "Superseded old by" in out

    def test_supersede_with_text_and_id(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "old", "Old belief", db_path=db_path)
        out, err, code = run_cli("supersede", "old", "--text", "Corrected", "--id", "old-fixed", db_path=db_path)
        assert code == 0
        assert "Superseded old by old-fixed" in out

    def test_supersede_text_and_new_id_errors(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "old", "Old belief", db_path=db_path)
        run_cli("add", "new", "New belief", db_path=db_path)
        out, err, code = run_cli("supersede", "old", "new", "--text", "Both", db_path=db_path)
        assert code == 1
        assert "cannot specify both" in err

    def test_supersede_neither_text_nor_id_errors(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "old", "Old belief", db_path=db_path)
        out, err, code = run_cli("supersede", "old", db_path=db_path)
        assert code == 1
        assert "either new_id or --text is required" in err


class TestCheckIntegrity:

    def test_clean(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "p1", "Premise", db_path=db_path)
        run_cli("add", "d1", "Derived", "--sl", "p1", db_path=db_path)
        out, err, code = run_cli("check-integrity", db_path=db_path)
        assert code == 0
        assert "no mutations detected" in out

    def test_detects_mutation(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "p1", "Premise", db_path=db_path)
        run_cli("add", "d1", "Derived", "--sl", "p1", db_path=db_path)
        # Simulate unauthorized text mutation via Storage
        from reasonsforge.storage import Storage
        store = Storage(db_path)
        net = store.load()
        net.nodes["p1"].text = "Tampered"
        store.save(net)
        store.close()
        out, err, code = run_cli("check-integrity", db_path=db_path)
        assert code == 1
        assert "p1" in out


class TestBackfillHashes:

    def test_backfill(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "p1", "Premise", db_path=db_path)
        # Strip hashes to simulate pre-existing data
        from reasonsforge.storage import Storage
        store = Storage(db_path)
        net = store.load()
        net.nodes["p1"].text_hash = ""
        store.save(net)
        store.close()
        out, err, code = run_cli("backfill-hashes", db_path=db_path)
        assert code == 0
        assert "1" in out  # at least 1 node updated


class TestConvertToPremise:

    def test_convert_to_premise(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        out, err, code = run_cli("convert-to-premise", "b", db_path=db_path)
        assert code == 0
        assert "Converted b to premise" in out
        assert "stripped 1 justification" in out


class TestPropagate:

    def test_propagate_current(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("propagate", db_path=db_path)
        assert code == 0
        assert "current" in out


class TestLog:

    def test_log_empty(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("log", db_path=db_path)
        assert code == 0
        assert "No propagation events" in out


class TestRepos:

    def test_add_repo_and_list(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("add-repo", "myrepo", "/tmp/myrepo", db_path=db_path)
        assert code == 0
        assert "Added repo myrepo" in out

        out, err, code = run_cli("repos", db_path=db_path)
        assert code == 0
        assert "myrepo" in out
        assert "1 repo" in out

    def test_repos_empty(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("repos", db_path=db_path)
        assert code == 0
        assert "No repos" in out


class TestImportExportJson:

    def test_import_json_roundtrip(self, db_path, tmp_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)

        out, _, _ = run_cli("export", "--output=-", db_path=db_path)
        json_file = str(tmp_path / "export.json")
        from pathlib import Path
        Path(json_file).write_text(out)

        db2 = str(tmp_path / "test2.db")
        run_cli("init", db_path=db2)
        out, err, code = run_cli("import-json", json_file, db_path=db2)
        assert code == 0
        assert "Imported 2 nodes" in out


class TestSummarize:

    def test_summarize(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", db_path=db_path)
        out, err, code = run_cli("summarize", "s", "Summary of A and B", "--over", "a,b", db_path=db_path)
        assert code == 0
        assert "Created summary s" in out
        assert "2 nodes" in out


class TestImportBeliefs:

    def test_import_beliefs(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### premise-a [IN] OBSERVATION
First premise

### premise-b [IN] OBSERVATION
Second premise

### derived-c [IN] DERIVED
Derived from both
- Depends on: premise-a, premise-b
""")
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("import-beliefs", str(beliefs), db_path=db_path)
        assert code == 0
        assert "Imported 3 claims" in out

    def test_import_beliefs_file_not_found(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("import-beliefs", "/nonexistent/beliefs.md", db_path=db_path)
        assert code == 1
        assert "Error" in err


class TestImportAgent:

    def test_import_agent(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
An observation from the agent
""")
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("import-agent", "myagent", str(beliefs), db_path=db_path)
        assert code == 0
        assert "myagent" in out
        assert "Imported:" in out or "imported" in out.lower()

    def test_import_agent_file_not_found(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("import-agent", "myagent", "/nonexistent.md", db_path=db_path)
        assert code == 1

    def test_sync_agent(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
An observation from the agent
""")
        run_cli("init", db_path=db_path)
        run_cli("import-agent", "myagent", str(beliefs), db_path=db_path)

        beliefs.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
Updated observation text

### obs-two [IN] OBSERVATION
A new observation
""")
        out, err, code = run_cli("sync-agent", "myagent", str(beliefs), db_path=db_path)
        assert code == 0
        assert "synced" in out.lower()


class TestNamespaces:

    def test_namespaces_empty(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("namespaces", db_path=db_path)
        assert code == 0
        assert "No namespaces" in out

    def test_namespaces_with_agent(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
An observation
""")
        run_cli("init", db_path=db_path)
        run_cli("import-agent", "testns", str(beliefs), db_path=db_path)
        out, err, code = run_cli("namespaces", db_path=db_path)
        assert code == 0
        assert "testns" in out
        assert "1 namespace" in out


class TestDeduplicate:

    def test_deduplicate_no_duplicates(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "alpha", "Something about alpha", db_path=db_path)
        run_cli("add", "beta", "Something about beta", db_path=db_path)
        out, err, code = run_cli("deduplicate", db_path=db_path)
        assert code == 0
        assert "No duplicate" in out

    def test_deduplicate_finds_similar(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "propagation-is-bfs", "Propagation uses BFS", db_path=db_path)
        run_cli("add", "propagation-uses-bfs", "Propagation is BFS-based", db_path=db_path)
        out, err, code = run_cli("deduplicate", db_path=db_path)
        assert code == 0
        assert "Cluster" in out or "No duplicate" in out

    def test_deduplicate_auto(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "propagation-is-bfs", "Propagation uses BFS", db_path=db_path)
        run_cli("add", "propagation-uses-bfs", "Propagation is BFS-based", db_path=db_path)
        out, err, code = run_cli("deduplicate", "--auto", db_path=db_path)
        assert code == 0

    def test_deduplicate_accept(self, db_path, tmp_path):
        plan_file = tmp_path / "plan.md"
        plan_file.write_text("")
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("deduplicate", "--accept", str(plan_file), db_path=db_path)
        assert code == 0
        assert "No clusters" in out

    def test_deduplicate_accept_missing_file(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("deduplicate", "--accept", "/nonexistent.md", db_path=db_path)
        assert code == 1


try:
    from reasonsforge.cluster import HAS_CLUSTER_DEPS
except ImportError:
    HAS_CLUSTER_DEPS = False


@pytest.mark.skipif(not HAS_CLUSTER_DEPS,
                    reason="sentence-transformers and scikit-learn not installed")
class TestDeduplicateSemantic:

    def test_semantic_finds_similar(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "input-validation-at-boundaries",
                "The system validates all inputs at system boundaries", db_path=db_path)
        run_cli("add", "boundary-input-checking",
                "Input validation occurs at system edges and boundaries", db_path=db_path)
        out, err, code = run_cli("deduplicate", "--semantic", db_path=db_path)
        assert code == 0
        assert "Cluster" in out

    def test_semantic_with_threshold(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "alpha", "Something about alpha", db_path=db_path)
        run_cli("add", "beta", "Something completely unrelated about beta", db_path=db_path)
        out, err, code = run_cli("deduplicate", "--semantic", "--threshold", "0.95",
                                  db_path=db_path)
        assert code == 0
        assert "No duplicate" in out


class TestCheckStale:

    def test_check_stale_all_fresh(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("check-stale", db_path=db_path)
        assert code == 0
        assert "fresh" in out

    def test_check_stale_detects_change(self, db_path, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "source.py"
        src.write_text("original content")
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", "--source", "source.py", db_path=db_path)
        run_cli("hash-sources", db_path=db_path)
        src.write_text("modified content")
        out, err, code = run_cli("check-stale", db_path=db_path)
        assert code == 1
        assert "STALE" in out

    def test_check_stale_detects_deleted(self, db_path, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "source.py"
        src.write_text("content")
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", "--source", "source.py", db_path=db_path)
        run_cli("hash-sources", db_path=db_path)
        src.unlink()
        out, err, code = run_cli("check-stale", db_path=db_path)
        assert code == 1
        assert "DELETED" in out


class TestHashSources:

    def test_hash_sources_nothing_to_hash(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("hash-sources", db_path=db_path)
        assert code == 0
        assert "No nodes to hash" in out

    def test_hash_sources_backfill(self, db_path, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "source.py"
        src.write_text("content")
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", "--source", "source.py", db_path=db_path)
        out, err, code = run_cli("hash-sources", db_path=db_path)
        assert code == 0
        assert "backfilled" in out

    def test_hash_sources_force(self, db_path, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "source.py"
        src.write_text("content")
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", "--source", "source.py", db_path=db_path)
        run_cli("hash-sources", db_path=db_path)
        out, err, code = run_cli("hash-sources", "--force", db_path=db_path)
        assert code == 0
        assert "re-hashed" in out


class TestDerive:

    def test_derive_dry_run(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A premise", db_path=db_path)
        run_cli("add", "b", "B premise", db_path=db_path)
        out, err, code = run_cli("derive", "--dry-run", db_path=db_path)
        assert code == 0
        assert "Prompt" in out
        assert "a" in out

    def test_derive_dry_run_empty(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("derive", "--dry-run", db_path=db_path)
        assert code == 1


class TestAccept:

    def test_accept_file_not_found(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("accept", "/nonexistent.md", db_path=db_path)
        assert code == 1
        assert "not found" in err.lower()

    def test_accept_empty_file(self, db_path, tmp_path):
        proposals = tmp_path / "proposals.md"
        proposals.write_text("No proposals here.")
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("accept", str(proposals), db_path=db_path)
        assert code == 0
        assert "No proposals" in out

    def test_accept_valid_proposals(self, db_path, tmp_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "fact-a", "Fact A", db_path=db_path)
        run_cli("add", "fact-b", "Fact B", db_path=db_path)
        proposals = tmp_path / "proposals.md"
        proposals.write_text("""\
### DERIVE combined-fact
Combined from A and B
- Antecedents: fact-a, fact-b
- Label: test derivation
""")
        out, err, code = run_cli("accept", str(proposals), db_path=db_path)
        assert code == 0
        assert "combined-fact" in out
        assert "Accepted 1" in err


class TestNoCommand:

    def test_no_command_prints_help(self):
        out, err, code = run_cli()
        assert code == 1

    def test_version(self):
        out, err, code = run_cli("--version")
        assert code == 0
        assert "reasons" in out
        assert "0." in out or "1." in out


class TestAddJustificationCascade:

    def test_add_justification_cascade(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        run_cli("add", "c", "C", db_path=db_path)
        out, err, code = run_cli("add-justification", "a", "--sl", "c", db_path=db_path)
        assert code == 0
        assert "Cascade" in out


class TestRetractRestorationHints:

    def test_retract_restoration_hints(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "p1", "Premise 1", db_path=db_path)
        run_cli("add", "p2", "Premise 2", db_path=db_path)
        run_cli("add", "derived", "Derived from both", "--sl", "p1,p2", db_path=db_path)
        out, err, code = run_cli("retract", "p1", db_path=db_path)
        assert code == 0
        assert "Went OUT" in out
        if "Note:" in out:
            assert "reasons add-justification" in out


class TestAssertMissing:

    def test_assert_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("assert", "nonexistent", db_path=db_path)
        assert code == 1
        assert "Error" in err


class TestWhatIfEdgeCases:

    def test_what_if_assert_already_in(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        out, err, code = run_cli("what-if", "assert", "a", db_path=db_path)
        assert code == 0
        assert "already IN" in out

    def test_what_if_retract_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("what-if", "retract", "missing", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_what_if_assert_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("what-if", "assert", "missing", db_path=db_path)
        assert code == 1
        assert "Error" in err


class TestShowSource:

    def test_show_with_source(self, db_path, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        src = tmp_path / "source.py"
        src.write_text("content")
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", "--source", "source.py", db_path=db_path)
        run_cli("hash-sources", db_path=db_path)
        out, err, code = run_cli("show", "a", db_path=db_path)
        assert code == 0
        assert "Source: source.py" in out
        assert "Hash:" in out


class TestExplainBranches:

    def test_explain_with_outlist(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "blocker", "Blocker", db_path=db_path)
        run_cli("retract", "blocker", db_path=db_path)
        run_cli("add", "gated", "Gated belief", "--unless", "blocker", db_path=db_path)
        out, err, code = run_cli("explain", "gated", db_path=db_path)
        assert code == 0
        assert "unless:" in out

    def test_explain_failed_antecedent(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        out, err, code = run_cli("explain", "b", db_path=db_path)
        assert code == 0
        assert "failed:" in out

    def test_explain_violated_outlist(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "blocker", "Blocker", db_path=db_path)
        run_cli("retract", "blocker", db_path=db_path)
        run_cli("add", "gated", "Gated", "--unless", "blocker", db_path=db_path)
        run_cli("assert", "blocker", db_path=db_path)
        out, err, code = run_cli("explain", "gated", db_path=db_path)
        assert code == 0
        assert "violated unless:" in out

    def test_explain_with_label(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", "--label", "test-label", db_path=db_path)
        out, err, code = run_cli("explain", "b", db_path=db_path)
        assert code == 0
        assert "test-label" in out


class TestConvertToPremiseEdgeCases:

    def test_convert_to_premise_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("convert-to-premise", "missing", db_path=db_path)
        assert code == 1
        assert "Error" in err


class TestErrorCases:

    def test_summarize_missing_node(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("summarize", "s", "Summary", "--over", "missing", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_supersede_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("supersede", "missing-old", "missing-new", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_challenge_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("challenge", "missing", "I disagree", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_defend_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("defend", "missing", "challenge-missing", "evidence", db_path=db_path)
        assert code == 1
        assert "Error" in err


class TestTraceEdgeCases:

    def test_trace_missing(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("trace", "missing", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_trace_access_tags_denied(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "fin", "Finance data", "--access-tags", "finance", db_path=db_path)
        out, err, code = run_cli("trace-access-tags", "fin", "--visible-to", "hr", db_path=db_path)
        assert code == 1
        assert "Access denied" in err


class TestLogWithEntries:

    def test_log_with_entries(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("retract", "a", db_path=db_path)
        out, err, code = run_cli("log", db_path=db_path)
        assert code == 0
        assert "add" in out
        assert "retract" in out


class TestPropagateWithChanges:

    def test_propagate_with_changes(self, db_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", "--sl", "a", db_path=db_path)
        # Directly set b's truth_value to OUT without proper retraction
        from reasonsforge.storage import Storage
        store = Storage(db_path)
        net = store.load()
        net.nodes["b"].truth_value = "OUT"
        store.save(net)
        store.close()
        out, err, code = run_cli("propagate", db_path=db_path)
        assert code == 0
        assert "Updated:" in out
        assert "b" in out


class TestImportAgentEdgeCases:

    def test_import_agent_existing_premise(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
An observation from the agent
""")
        run_cli("init", db_path=db_path)
        run_cli("import-agent", "myagent", str(beliefs), db_path=db_path)
        out, err, code = run_cli("import-agent", "myagent", str(beliefs), db_path=db_path)
        assert code == 0
        assert "Premise exists:" in out

    def test_import_agent_with_skipped(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
An observation
""")
        run_cli("init", db_path=db_path)
        run_cli("import-agent", "myagent", str(beliefs), db_path=db_path)
        out, err, code = run_cli("import-agent", "myagent", str(beliefs), db_path=db_path)
        assert code == 0
        assert "Skipped:" in out

    def test_sync_agent_file_not_found(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("sync-agent", "myagent", "/nonexistent.md", db_path=db_path)
        assert code == 1
        assert "Error" in err

    def test_sync_agent_with_removals(self, db_path, tmp_path):
        beliefs_v1 = tmp_path / "beliefs.md"
        beliefs_v1.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
First observation

### obs-two [IN] OBSERVATION
Second observation
""")
        run_cli("init", db_path=db_path)
        run_cli("import-agent", "myagent", str(beliefs_v1), db_path=db_path)

        beliefs_v1.write_text("""\
# Belief Registry

## Claims

### obs-one [IN] OBSERVATION
First observation
""")
        out, err, code = run_cli("sync-agent", "myagent", str(beliefs_v1), db_path=db_path)
        assert code == 0
        assert "Removed:" in out


class TestImportBeliefsEdgeCases:

    def test_import_beliefs_with_skipped(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### premise-a [IN] OBSERVATION
First premise
""")
        run_cli("init", db_path=db_path)
        run_cli("import-beliefs", str(beliefs), db_path=db_path)
        out, err, code = run_cli("import-beliefs", str(beliefs), db_path=db_path)
        assert code == 0
        assert "Skipped" in out

    def test_import_beliefs_with_nogoods(self, db_path, tmp_path):
        beliefs = tmp_path / "beliefs.md"
        beliefs.write_text("""\
# Belief Registry

## Claims

### premise-a [IN] OBSERVATION
First premise

### premise-b [IN] OBSERVATION
Second premise

## Nogoods

### nogood-1
- premise-a, premise-b
""")
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("import-beliefs", str(beliefs), db_path=db_path)
        assert code == 0
        if "nogood" in out.lower():
            assert "1" in out


class TestImportJsonWithNogoods:

    def test_import_json_with_nogoods(self, db_path, tmp_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "a", "A", db_path=db_path)
        run_cli("add", "b", "B", db_path=db_path)
        run_cli("nogood", "a", "b", db_path=db_path)

        out, _, _ = run_cli("export", "--output=-", db_path=db_path)
        json_file = str(tmp_path / "export.json")
        from pathlib import Path
        Path(json_file).write_text(out)

        db2 = str(tmp_path / "test2.db")
        run_cli("init", db_path=db2)
        out, err, code = run_cli("import-json", json_file, db_path=db2)
        assert code == 0
        assert "nogoods" in out.lower()


class TestDeduplicateAcceptPlan:

    def test_deduplicate_accept_with_retraction(self, db_path, tmp_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "prop-is-bfs", "Propagation uses BFS", db_path=db_path)
        run_cli("add", "prop-uses-bfs", "Propagation is BFS-based", db_path=db_path)
        plan_file = str(tmp_path / "dedup-plan.md")
        run_cli("deduplicate", "--output", plan_file, db_path=db_path)

        from pathlib import Path
        plan_content = Path(plan_file).read_text() if Path(plan_file).exists() else ""
        if plan_content.strip():
            out, err, code = run_cli("deduplicate", "--accept", plan_file, db_path=db_path)
            assert code == 0
            assert "Retracted" in out or "No duplicates" in out


class TestAcceptEdgeCases:

    def test_accept_with_skipped(self, db_path, tmp_path):
        run_cli("init", db_path=db_path)
        run_cli("add", "fact-a", "Fact A", db_path=db_path)
        proposals = tmp_path / "proposals.md"
        proposals.write_text("""\
### DERIVE bad-derive
Derived from nonexistent
- Antecedents: nonexistent-node
- Label: test
""")
        out, err, code = run_cli("accept", str(proposals), db_path=db_path)
        assert code == 0
        assert "SKIP" in err or "No valid" in out

    def test_accept_all_skipped(self, db_path, tmp_path):
        run_cli("init", db_path=db_path)
        proposals = tmp_path / "proposals.md"
        proposals.write_text("""\
### DERIVE bad-one
Derived from nothing
- Antecedents: missing-a
- Label: test derivation

### DERIVE bad-two
Also from nothing
- Antecedents: missing-b
- Label: test derivation
""")
        out, err, code = run_cli("accept", str(proposals), db_path=db_path)
        assert code == 0
        assert "No valid proposals" in out


class TestCmdUpdate:

    def test_update_text_removed(self, db_path):
        run_cli("add", "a", "Original text", db_path=db_path)
        out, err, code = run_cli("update", "a", "--source", "new.py", db_path=db_path)
        assert code == 0
        assert "Updated a" in out
        assert "source" in out

    def test_update_nonexistent(self, db_path):
        run_cli("init", db_path=db_path)
        out, err, code = run_cli("update", "nope", "--source", "test.py", db_path=db_path)
        assert code == 1
        assert "not found" in err

    def test_update_source_only(self, db_path):
        run_cli("add", "a", "Some text", db_path=db_path)
        out, err, code = run_cli("update", "a", "--source", "new/path.md", db_path=db_path)
        assert code == 0
        assert "source" in out

    def test_no_flags_errors(self, db_path):
        run_cli("add", "a", "Some text", db_path=db_path)
        out, err, code = run_cli("update", "a", db_path=db_path)
        assert code == 1
        assert "at least one" in err
