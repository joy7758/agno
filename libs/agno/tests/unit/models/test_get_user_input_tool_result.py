"""
Tests for get_user_input tool result placeholder generation.

Reasoning models (o3, o4-mini, gpt-5.2) using the Responses API with
previous_response_id require ALL function calls to have tool results.
The get_user_input UCF tool must produce a placeholder result BEFORE
pausing so the result exists in function_call_results.
"""

from typing import List
from unittest.mock import MagicMock

import pytest

from agno.models.base import Model, ModelResponse, ModelResponseEvent
from agno.models.message import Message
from agno.tools.function import Function, FunctionCall
from agno.tools.user_control_flow import UserControlFlowTools


def _make_get_user_input_fc() -> FunctionCall:
    """Create a FunctionCall for get_user_input with typical UCF arguments."""
    ucf = UserControlFlowTools()
    func = None
    for f in ucf.functions.values():
        if f.name == "get_user_input":
            func = f
            break
    assert func is not None, "get_user_input not found in UserControlFlowTools"
    func.process_entrypoint()

    return FunctionCall(
        function=func,
        call_id="fc_test_123",
        arguments={
            "user_input_fields": [
                {
                    "field_name": "to_address",
                    "field_type": "str",
                    "field_description": "Recipient email",
                }
            ]
        },
    )


def test_sync_get_user_input_produces_tool_result_before_pause():
    """run_function_calls should execute get_user_input and produce a tool
    result in function_call_results BEFORE yielding the pause event."""
    fc = _make_get_user_input_fc()
    function_call_results: List[Message] = []

    # We need a concrete Model subclass to call run_function_calls.
    # Use a minimal mock that inherits from Model.
    model = MagicMock(spec=Model)
    model.tool_message_role = "tool"

    # Bind the real methods from Model
    model.run_function_calls = Model.run_function_calls.__get__(model, type(model))
    model.run_function_call = Model.run_function_call.__get__(model, type(model))
    model.create_function_call_result = Model.create_function_call_result.__get__(model, type(model))

    events = list(
        model.run_function_calls(
            function_calls=[fc],
            function_call_results=function_call_results,
        )
    )

    # Should have events: tool_call_started, tool_call_completed, tool_call_paused
    event_types = [e.event for e in events if isinstance(e, ModelResponse) and e.event]
    assert ModelResponseEvent.tool_call_paused.value in event_types, (
        f"Expected tool_call_paused in events, got: {event_types}"
    )

    # The critical assertion: function_call_results must contain the placeholder
    assert len(function_call_results) >= 1, (
        "function_call_results should have at least 1 entry (the placeholder)"
    )
    placeholder = function_call_results[0]
    assert placeholder.tool_call_id == "fc_test_123"
    assert placeholder.role == "tool"
    assert "User input received" in str(placeholder.content)


def test_sync_get_user_input_pause_event_has_user_input_schema():
    """The pause event should carry the user_input_schema for the UI."""
    fc = _make_get_user_input_fc()
    function_call_results: List[Message] = []

    model = MagicMock(spec=Model)
    model.tool_message_role = "tool"
    model.run_function_calls = Model.run_function_calls.__get__(model, type(model))
    model.run_function_call = Model.run_function_call.__get__(model, type(model))
    model.create_function_call_result = Model.create_function_call_result.__get__(model, type(model))

    events = list(
        model.run_function_calls(
            function_calls=[fc],
            function_call_results=function_call_results,
        )
    )

    pause_events = [
        e for e in events
        if isinstance(e, ModelResponse) and e.event == ModelResponseEvent.tool_call_paused.value
    ]
    assert len(pause_events) == 1
    pause = pause_events[0]
    assert pause.tool_executions is not None
    assert any(te.requires_user_input for te in pause.tool_executions)


def test_replace_in_place_on_resume():
    """handle_get_user_input_tool_update should replace existing placeholder,
    not append a duplicate."""
    import json

    from agno.agent._tools import handle_get_user_input_tool_update
    from agno.models.message import Message
    from agno.run.messages import RunMessages
    from agno.tools.function import UserInputField
    from agno.models.response import ToolExecution

    # Simulate: messages already contain a placeholder from pre-pause execution
    placeholder = Message(
        role="tool",
        content="User input received",
        tool_call_id="fc_test_456",
        tool_name="get_user_input",
    )
    run_messages = RunMessages(messages=[
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Send an email"),
        placeholder,
    ])

    # Simulate the ToolExecution from the paused state with user input filled in
    tool = ToolExecution(
        tool_call_id="fc_test_456",
        tool_name="get_user_input",
        tool_args={"user_input_fields": [{"field_name": "to_address"}]},
        requires_user_input=True,
        user_input_schema=[
            UserInputField(name="to_address", field_type=str, description="Email", value="test@example.com"),
        ],
    )

    # Mock agent with model
    agent = MagicMock()
    agent.model.tool_message_role = "tool"

    handle_get_user_input_tool_update(agent, run_messages, tool)

    # Should have replaced in-place, NOT appended
    tool_msgs = [m for m in run_messages.messages if m.tool_call_id == "fc_test_456"]
    assert len(tool_msgs) == 1, f"Expected 1 tool message, got {len(tool_msgs)}"
    assert "test@example.com" in str(tool_msgs[0].content)
    assert "User input received" not in str(tool_msgs[0].content)


def test_append_when_no_placeholder_exists():
    """handle_get_user_input_tool_update should append when no placeholder exists
    (backward compat for non-reasoning models)."""
    import json

    from agno.agent._tools import handle_get_user_input_tool_update
    from agno.models.message import Message
    from agno.run.messages import RunMessages
    from agno.tools.function import UserInputField
    from agno.models.response import ToolExecution

    run_messages = RunMessages(messages=[
        Message(role="system", content="You are helpful."),
        Message(role="user", content="Send an email"),
        # No placeholder — simulates non-reasoning model path
    ])

    tool = ToolExecution(
        tool_call_id="fc_test_789",
        tool_name="get_user_input",
        tool_args={"user_input_fields": [{"field_name": "to_address"}]},
        requires_user_input=True,
        user_input_schema=[
            UserInputField(name="to_address", field_type=str, description="Email", value="user@test.com"),
        ],
    )

    agent = MagicMock()
    agent.model.tool_message_role = "tool"

    handle_get_user_input_tool_update(agent, run_messages, tool)

    # Should have appended
    tool_msgs = [m for m in run_messages.messages if m.tool_call_id == "fc_test_789"]
    assert len(tool_msgs) == 1
    assert "user@test.com" in str(tool_msgs[0].content)
