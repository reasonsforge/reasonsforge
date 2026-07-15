"""Tests for the --example flag on add, update, and show commands."""

import sys
from io import StringIO
from unittest.mock import patch

import pytest

from reasonsforge import api


def run_cli(*args, db_path=None):
    from reasonsforge.cli import main
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
            code = 0
        except SystemExit as e:
            code = e.code if e.code is not None else 0
    return stdout.getvalue(), stderr.getvalue(), code


class TestAddNodeExample:
    def test_add_with_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        result = api.add_node("n1", "Node one", example="x = 1 + 2", db_path=db)
        assert result["node_id"] == "n1"
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["example"] == "x = 1 + 2"

    def test_add_without_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "Node one", db_path=db)
        node = api.show_node("n1", db_path=db)
        assert "example" not in node["metadata"]

    def test_add_example_with_access_tags(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "Node one", example="print('hi')",
                     access_tags=["tag1"], db_path=db)
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["example"] == "print('hi')"
        assert node["metadata"]["access_tags"] == ["tag1"]


class TestUpdateNodeExample:
    def test_update_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "Node one", db_path=db)
        result = api.update_node("n1", example="y = 2", db_path=db)
        assert "example" in result["updated_fields"]
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["example"] == "y = 2"

    def test_update_replaces_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "Node one", example="old code", db_path=db)
        api.update_node("n1", example="new code", db_path=db)
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["example"] == "new code"

    def test_update_example_only(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "Original text", db_path=db)
        api.update_node("n1", example="z = 3", db_path=db)
        node = api.show_node("n1", db_path=db)
        assert node["text"] == "Original text"
        assert node["metadata"]["example"] == "z = 3"

    def test_update_example_preserves_other_metadata(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "Node one", access_tags=["internal"], db_path=db)
        api.update_node("n1", example="code()", db_path=db)
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["access_tags"] == ["internal"]
        assert node["metadata"]["example"] == "code()"


class TestCmdAddExample:
    def test_cli_add_with_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        stdout, stderr, code = run_cli(
            "add", "n1", "A belief", "--example", "reasons add n1 'A belief'",
            db_path=db,
        )
        assert code == 0
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["example"] == "reasons add n1 'A belief'"

    def test_cli_add_without_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        stdout, stderr, code = run_cli("add", "n1", "A belief", db_path=db)
        assert code == 0
        node = api.show_node("n1", db_path=db)
        assert "example" not in node["metadata"]


class TestCmdUpdateExample:
    def test_cli_update_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        run_cli("add", "n1", "A belief", db_path=db)
        stdout, stderr, code = run_cli(
            "update", "n1", "--example", "reasons update n1 --text 'new'",
            db_path=db,
        )
        assert code == 0
        assert "example" in stdout
        node = api.show_node("n1", db_path=db)
        assert node["metadata"]["example"] == "reasons update n1 --text 'new'"

    def test_cli_update_no_flags_error(self, tmp_path):
        db = str(tmp_path / "test.db")
        run_cli("add", "n1", "A belief", db_path=db)
        stdout, stderr, code = run_cli("update", "n1", db_path=db)
        assert code == 1
        assert "--example" in stderr


class TestCmdShowExample:
    def test_show_displays_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "A belief", example="x = func()", db_path=db)
        stdout, stderr, code = run_cli("show", "n1", db_path=db)
        assert code == 0
        assert "Example:" in stdout
        assert "x = func()" in stdout

    def test_show_multiline_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        code_snippet = "x = 1\ny = 2\nprint(x + y)"
        api.add_node("n1", "A belief", example=code_snippet, db_path=db)
        stdout, stderr, code = run_cli("show", "n1", db_path=db)
        assert code == 0
        lines = stdout.split("\n")
        example_lines = []
        capture = False
        for line in lines:
            if line.strip().startswith("Example:"):
                capture = True
                continue
            if capture:
                if line.startswith("  "):
                    example_lines.append(line)
                elif line.strip() == "":
                    continue
                else:
                    break
        assert len(example_lines) == 3
        assert all(line.startswith("  ") for line in example_lines)

    def test_show_no_example(self, tmp_path):
        db = str(tmp_path / "test.db")
        api.add_node("n1", "A belief", db_path=db)
        stdout, stderr, code = run_cli("show", "n1", db_path=db)
        assert code == 0
        assert "Example:" not in stdout
