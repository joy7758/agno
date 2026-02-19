"""Tests for MigrationManager.up() version-bump guard.

The `any_migration_executed` flag ensures that `upsert_schema_version`
is only called when at least one migration module actually ran.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from agno.db.base import AsyncBaseDb
from agno.db.migrations.manager import MigrationManager

_TABLE_ATTRS = {
    "memory_table_name": "memories",
    "session_table_name": "sessions",
    "metrics_table_name": "metrics",
    "eval_table_name": "evals",
    "knowledge_table_name": "knowledge",
    "culture_table_name": "culture",
}


def _make_sync_db(current_version: str = "2.0.0"):
    db = MagicMock()
    db.get_latest_schema_version.return_value = current_version
    for attr, val in _TABLE_ATTRS.items():
        setattr(db, attr, val)
    return db


def _make_async_db(current_version: str = "2.0.0"):
    db = MagicMock(spec=AsyncBaseDb)
    db.get_latest_schema_version = AsyncMock(return_value=current_version)
    db.upsert_schema_version = AsyncMock()
    for attr, val in _TABLE_ATTRS.items():
        setattr(db, attr, val)
    return db


class TestVersionBumpGuard:
    @pytest.mark.asyncio
    async def test_version_not_stored_when_no_migration_executed(self):
        db = _make_sync_db(current_version="2.0.0")
        manager = MigrationManager(db)

        with patch.object(manager, "_up_migration", new_callable=AsyncMock, return_value=False):
            await manager.up()

        db.upsert_schema_version.assert_not_called()

    @pytest.mark.asyncio
    async def test_version_stored_when_migration_executed(self):
        db = _make_sync_db(current_version="2.0.0")
        manager = MigrationManager(db)

        with patch.object(manager, "_up_migration", new_callable=AsyncMock, return_value=True):
            await manager.up()

        assert db.upsert_schema_version.call_count > 0

    @pytest.mark.asyncio
    async def test_async_version_not_stored_when_no_migration_executed(self):
        db = _make_async_db(current_version="2.0.0")
        manager = MigrationManager(db)

        with patch.object(manager, "_up_migration", new_callable=AsyncMock, return_value=False):
            await manager.up()

        db.upsert_schema_version.assert_not_called()

    @pytest.mark.asyncio
    async def test_async_version_stored_when_migration_executed(self):
        db = _make_async_db(current_version="2.0.0")
        manager = MigrationManager(db)

        with patch.object(manager, "_up_migration", new_callable=AsyncMock, return_value=True):
            await manager.up()

        assert db.upsert_schema_version.call_count > 0


class TestDownVersionBumpGuard:
    @pytest.mark.asyncio
    async def test_version_not_stored_on_noop_down(self):
        db = _make_sync_db(current_version="2.5.0")
        manager = MigrationManager(db)

        with patch.object(manager, "_down_migration", new_callable=AsyncMock, return_value=False):
            await manager.down(target_version="2.0.0")

        db.upsert_schema_version.assert_not_called()

    @pytest.mark.asyncio
    async def test_version_stored_on_successful_down(self):
        db = _make_sync_db(current_version="2.5.0")
        manager = MigrationManager(db)

        with patch.object(manager, "_down_migration", new_callable=AsyncMock, return_value=True):
            await manager.down(target_version="2.0.0")

        assert db.upsert_schema_version.call_count > 0
