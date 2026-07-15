"""Shared fixtures for tests."""

import os
import uuid

import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "pg: requires PostgreSQL (set DATABASE_URL)")


pg_available = bool(os.environ.get("DATABASE_URL"))
skip_no_pg = pytest.mark.skipif(not pg_available, reason="DATABASE_URL not set")


@pytest.fixture
def pg_api():
    """Provide a PgApi instance with a unique project_id, cleaned up after test."""
    if not pg_available:
        pytest.skip("DATABASE_URL not set")

    from reasonsforge.pg import PgApi

    conninfo = os.environ["DATABASE_URL"]
    project_id = str(uuid.uuid4())
    api = PgApi(conninfo, project_id)
    api.init_db()

    yield api

    # Clean up test data (rollback any failed transaction first)
    api.conn.rollback()
    with api.conn.cursor() as cur:
        for table in ("rms_propagation_log", "rms_justifications",
                      "rms_nogoods", "rms_network_meta", "rms_nodes"):
            cur.execute(f"DELETE FROM {table} WHERE project_id = %s", (project_id,))
    api.conn.commit()
    api.close()
