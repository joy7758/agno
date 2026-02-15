"""
BUG #6327 regression guard: RunRequirement.to_dict() must serialize datetime
objects to ISO strings so PostgreSQL JSONB storage doesn't raise TypeError.
"""
import json
from datetime import datetime, timezone

from agno.models.response import ToolExecution
from agno.run.requirement import RunRequirement


def test_run_requirement_to_dict_serializes_datetime():
    """
    BUG #6327: RunRequirement.to_dict() must convert created_at datetime
    to an ISO string. Without this, json.dumps() raises
    TypeError: Object of type datetime is not JSON serializable.
    """
    te = ToolExecution(tool_name="test_tool", tool_args={"key": "val"})
    req = RunRequirement(
        tool_execution=te,
        created_at=datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
    )

    d = req.to_dict()
    assert isinstance(d["created_at"], str), (
        "BUG #6327 regression: created_at must be serialized to ISO string"
    )
    json.dumps(d)


def test_run_requirement_from_dict_parses_iso_datetime():
    """
    BUG #6327: RunRequirement.from_dict() must handle ISO datetime strings
    from previously serialized requirements.
    """
    te = ToolExecution(tool_name="test_tool", tool_args={"key": "val"})
    original = RunRequirement(
        tool_execution=te,
        created_at=datetime(2025, 6, 15, 10, 30, 0, tzinfo=timezone.utc),
    )

    d = original.to_dict()
    json_str = json.dumps(d)
    restored_dict = json.loads(json_str)

    restored = RunRequirement.from_dict(restored_dict)
    assert isinstance(restored.created_at, datetime), (
        "BUG #6327 regression: from_dict must parse ISO string back to datetime"
    )
    assert restored.created_at.year == 2025
    assert restored.created_at.month == 6
