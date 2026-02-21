from typing import Any, Callable, Coroutine, Literal, Optional

from agno.agent import RunEvent
from agno.os.interfaces.slack.helpers import member_name, task_id
from agno.os.interfaces.slack.state import StreamState
from agno.run.team import TeamRunEvent
from agno.run.workflow import WorkflowRunEvent

_BOT_SUBTYPES = frozenset({"bot_message", "bot_add", "bot_remove", "bot_enable", "bot_disable"})

_MAX_OUTPUT = 200


def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


HandlerFn = Callable[..., Coroutine[Any, Any, Optional[Literal["break"]]]]


async def _emit_task(
    stream: Any,
    task_id: str,
    title: str,
    status: str,
    *,
    output: str | None = None,
) -> None:
    chunk: dict = {"type": "task_update", "id": task_id, "title": title, "status": status}
    if output:
        chunk["output"] = _truncate(output)
    await stream.append(chunks=[chunk])


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


async def handle_reasoning_started(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    key = f"reasoning_{state.reasoning_round}"
    state.track_task(key, "Reasoning")
    await _emit_task(stream, key, "Reasoning", "in_progress")
    return None


async def handle_reasoning_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    key = f"reasoning_{state.reasoning_round}"
    state.complete_task(key)
    state.reasoning_round += 1
    await _emit_task(stream, key, "Reasoning", "complete")
    return None


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def handle_tool_started(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    tool_name = chunk.tool.tool_name if chunk.tool else "a tool"  # type: ignore[union-attr]
    call_id = chunk.tool.tool_call_id if chunk.tool else str(len(state.task_cards))  # type: ignore[union-attr]
    member = member_name(chunk, state.entity_name)
    label = tool_name
    if member:
        label = f"{member}: {label}"
    tid = task_id(member, call_id)  # type: ignore[arg-type]
    state.track_task(tid, label)
    await _emit_task(stream, tid, label, "in_progress")
    return None


async def handle_tool_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    call_id = chunk.tool.tool_call_id if chunk.tool else None  # type: ignore[union-attr]
    tool_name = chunk.tool.tool_name if chunk.tool else "tool"  # type: ignore[union-attr]
    member = member_name(chunk, state.entity_name)
    tid = task_id(member, call_id) if call_id else None  # type: ignore[arg-type]
    label = tool_name
    if member:
        label = f"{member}: {label}"

    errored = chunk.tool.tool_call_error if chunk.tool else False  # type: ignore[union-attr]
    status = "error" if errored else "complete"

    if tid:
        if tid not in state.task_cards:
            state.track_task(tid, label)
        if errored:
            state.error_task(tid)
        else:
            state.complete_task(tid)
        await _emit_task(stream, tid, label, status)

    state.collect_media(chunk)
    return None


async def handle_tool_error(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    tool_name = chunk.tool.tool_name if chunk.tool else "a tool"  # type: ignore[union-attr]
    call_id = chunk.tool.tool_call_id if chunk.tool else f"tool_error_{state.error_count}"  # type: ignore[union-attr]
    member = member_name(chunk, state.entity_name)
    tid = task_id(member, call_id)  # type: ignore[arg-type]
    label = tool_name
    if member:
        label = f"{member}: {label}"
    error_msg = getattr(chunk, "error", None) or "Tool call failed"
    state.error_count += 1

    if tid in state.task_cards:
        state.error_task(tid)
    else:
        state.track_task(tid, label)
        state.error_task(tid)

    await _emit_task(stream, tid, label, "error", output=str(error_msg))
    return None


# ---------------------------------------------------------------------------
# Content streaming
# ---------------------------------------------------------------------------


async def handle_content(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    if not chunk.content:
        return None
    state.text_buffer += str(chunk.content)
    return None


async def handle_intermediate_content(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    # Teams emit run_intermediate_content for each member's partial output.
    # Suppress it to avoid interleaved fragments — the final run_content
    # from the team leader contains the consolidated response.
    if state.entity_type == "team":
        return None
    return await handle_content(chunk, state, stream)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


async def handle_run_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    state.collect_media(chunk)
    if state.progress_started:
        chunks = state.resolve_all_pending()
        if chunks:
            await stream.append(chunks=chunks)
    return None


async def handle_run_failed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    state.error_count += 1
    error_msg = getattr(chunk, "content", None) or "An error occurred"
    state.text_buffer += f"\n_Error: {error_msg}_"
    if state.progress_started:
        chunks = state.resolve_all_pending("error")
        if chunks:
            await stream.append(chunks=chunks)
    return "break"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def handle_memory_started(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    state.track_task("memory_update", "Updating memory")
    await _emit_task(stream, "memory_update", "Updating memory", "in_progress")
    return None


async def handle_memory_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    state.complete_task("memory_update")
    await _emit_task(stream, "memory_update", "Updating memory", "complete")
    return None


# ---------------------------------------------------------------------------
# Workflow lifecycle
# ---------------------------------------------------------------------------


async def handle_workflow_started(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    wf_name = getattr(chunk, "workflow_name", None) or state.entity_name or "Workflow"
    run_id = getattr(chunk, "run_id", None) or "run"
    key = f"wf_run_{run_id}"
    state.track_task(key, f"Workflow: {wf_name}")
    await _emit_task(stream, key, f"Workflow: {wf_name}", "in_progress")
    return None


async def handle_workflow_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    run_id = getattr(chunk, "run_id", None) or "run"
    wf_name = getattr(chunk, "workflow_name", None) or state.entity_name or "Workflow"
    key = f"wf_run_{run_id}"
    state.complete_task(key)
    await _emit_task(stream, key, f"Workflow: {wf_name}", "complete")

    if state.progress_started:
        chunks = state.resolve_all_pending()
        if chunks:
            await stream.append(chunks=chunks)

    final = getattr(chunk, "content", None)
    if not final:
        final = state.workflow_final_content
    if final:
        state.text_buffer += str(final)
    return None


async def handle_workflow_failed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    state.error_count += 1
    error_msg = getattr(chunk, "error", None) or getattr(chunk, "content", None) or "Workflow failed"
    state.text_buffer += f"\n_Error: {error_msg}_"
    if state.progress_started:
        chunks = state.resolve_all_pending("error")
        if chunks:
            await stream.append(chunks=chunks)
    return "break"


# ---------------------------------------------------------------------------
# Workflow-mode overrides (used by WORKFLOW_DISPATCH)
# In workflow mode, nested agent events (reasoning, tools, content) are noisy —
# users care about step progress, not individual agent internals. These handlers
# suppress nested events while still collecting any media they carry.
# ---------------------------------------------------------------------------


async def handle_workflow_noop(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    state.collect_media(chunk)
    return None


async def handle_workflow_step_output(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    content = getattr(chunk, "content", None)
    if content is not None:
        state.workflow_final_content = str(content)
    state.collect_media(chunk)
    return None


# ---------------------------------------------------------------------------
# Workflow structural events (factory for started/completed pairs)
# ---------------------------------------------------------------------------


def _wf_handler(prefix: str, label: str, *, started: bool, name_attr: str = "step_name"):
    async def _handler(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
        name = getattr(chunk, name_attr, None) or prefix
        sid = getattr(chunk, "step_id", None) or name
        key = f"wf_{prefix}_{sid}"
        title = f"{label}: {name}" if label else name
        if started:
            state.track_task(key, title)
            await _emit_task(stream, key, title, "in_progress")
        else:
            state.complete_task(key)
            state.collect_media(chunk)
            await _emit_task(stream, key, title, "complete")
        return None

    return _handler


async def handle_step_error(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    step_name = getattr(chunk, "step_name", None) or "step"
    sid = getattr(chunk, "step_id", None) or step_name
    key = f"wf_step_{sid}"
    error_msg = getattr(chunk, "error", None) or "Step failed"
    state.error_task(key)
    await _emit_task(stream, key, step_name, "error", output=str(error_msg))
    return None


async def handle_loop_started(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    max_iter = getattr(chunk, "max_iterations", None)
    title = f"Loop: {step_name}" + (f" (max {max_iter})" if max_iter else "")
    key = f"wf_loop_{loop_key}"
    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return None


async def handle_loop_iter_started(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    iteration = getattr(chunk, "iteration", 0)
    max_iter = getattr(chunk, "max_iterations", None)
    title = f"Iteration {iteration}" + (f"/{max_iter}" if max_iter else "")
    key = f"wf_loop_{loop_key}_iter_{iteration}"
    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return None


async def handle_loop_iter_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    iteration = getattr(chunk, "iteration", 0)
    key = f"wf_loop_{loop_key}_iter_{iteration}"
    state.complete_task(key)
    await _emit_task(stream, key, f"Iteration {iteration}", "complete")
    return None


async def handle_loop_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[Literal["break"]]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    key = f"wf_loop_{loop_key}"
    state.complete_task(key)
    await _emit_task(stream, key, f"Loop: {step_name}", "complete")
    return None


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

DISPATCH: dict[str, HandlerFn] = {
    # Reasoning
    RunEvent.reasoning_started.value: handle_reasoning_started,
    TeamRunEvent.reasoning_started.value: handle_reasoning_started,
    RunEvent.reasoning_completed.value: handle_reasoning_completed,
    TeamRunEvent.reasoning_completed.value: handle_reasoning_completed,
    # Tools
    RunEvent.tool_call_started.value: handle_tool_started,
    TeamRunEvent.tool_call_started.value: handle_tool_started,
    RunEvent.tool_call_completed.value: handle_tool_completed,
    TeamRunEvent.tool_call_completed.value: handle_tool_completed,
    RunEvent.tool_call_error.value: handle_tool_error,
    TeamRunEvent.tool_call_error.value: handle_tool_error,
    # Content
    RunEvent.run_content.value: handle_content,
    TeamRunEvent.run_content.value: handle_content,
    RunEvent.run_intermediate_content.value: handle_intermediate_content,
    TeamRunEvent.run_intermediate_content.value: handle_intermediate_content,
    # Run lifecycle
    RunEvent.run_completed.value: handle_run_completed,
    TeamRunEvent.run_completed.value: handle_run_completed,
    RunEvent.run_error.value: handle_run_failed,
    TeamRunEvent.run_error.value: handle_run_failed,
    RunEvent.run_cancelled.value: handle_run_failed,
    TeamRunEvent.run_cancelled.value: handle_run_failed,
    # Memory
    RunEvent.memory_update_started.value: handle_memory_started,
    TeamRunEvent.memory_update_started.value: handle_memory_started,
    RunEvent.memory_update_completed.value: handle_memory_completed,
    TeamRunEvent.memory_update_completed.value: handle_memory_completed,
    # Workflow -- step output treated as content
    WorkflowRunEvent.step_output.value: handle_content,
    # Workflow lifecycle
    WorkflowRunEvent.workflow_started.value: handle_workflow_started,
    WorkflowRunEvent.workflow_completed.value: handle_workflow_completed,
    WorkflowRunEvent.workflow_error.value: handle_workflow_failed,
    WorkflowRunEvent.workflow_cancelled.value: handle_workflow_failed,
    # Workflow steps
    WorkflowRunEvent.step_started.value: _wf_handler("step", "", started=True),
    WorkflowRunEvent.step_completed.value: _wf_handler("step", "", started=False),
    WorkflowRunEvent.step_error.value: handle_step_error,
    # Workflow loops
    WorkflowRunEvent.loop_execution_started.value: handle_loop_started,
    WorkflowRunEvent.loop_iteration_started.value: handle_loop_iter_started,
    WorkflowRunEvent.loop_iteration_completed.value: handle_loop_iter_completed,
    WorkflowRunEvent.loop_execution_completed.value: handle_loop_completed,
    # Workflow parallel / conditions / routing / agent / steps-execution
    WorkflowRunEvent.parallel_execution_started.value: _wf_handler("parallel", "Parallel", started=True),
    WorkflowRunEvent.parallel_execution_completed.value: _wf_handler("parallel", "Parallel", started=False),
    WorkflowRunEvent.condition_execution_started.value: _wf_handler("cond", "Condition", started=True),
    WorkflowRunEvent.condition_execution_completed.value: _wf_handler("cond", "Condition", started=False),
    WorkflowRunEvent.router_execution_started.value: _wf_handler("router", "Router", started=True),
    WorkflowRunEvent.router_execution_completed.value: _wf_handler("router", "Router", started=False),
    WorkflowRunEvent.workflow_agent_started.value: _wf_handler(
        "agent", "Running", started=True, name_attr="agent_name"
    ),
    WorkflowRunEvent.workflow_agent_completed.value: _wf_handler(
        "agent", "Running", started=False, name_attr="agent_name"
    ),
    WorkflowRunEvent.steps_execution_started.value: _wf_handler("steps", "Steps", started=True),
    WorkflowRunEvent.steps_execution_completed.value: _wf_handler("steps", "Steps", started=False),
}

# Workflow-mode dispatch: shows only workflow structure + final output
WORKFLOW_DISPATCH: dict[str, HandlerFn] = {
    **DISPATCH,
    # Suppress nested agent reasoning cards
    RunEvent.reasoning_started.value: handle_workflow_noop,
    TeamRunEvent.reasoning_started.value: handle_workflow_noop,
    RunEvent.reasoning_completed.value: handle_workflow_noop,
    TeamRunEvent.reasoning_completed.value: handle_workflow_noop,
    # Suppress nested agent tool cards
    RunEvent.tool_call_started.value: handle_workflow_noop,
    TeamRunEvent.tool_call_started.value: handle_workflow_noop,
    RunEvent.tool_call_completed.value: handle_workflow_noop,
    TeamRunEvent.tool_call_completed.value: handle_workflow_noop,
    RunEvent.tool_call_error.value: handle_workflow_noop,
    TeamRunEvent.tool_call_error.value: handle_workflow_noop,
    # Suppress nested agent memory cards
    RunEvent.memory_update_started.value: handle_workflow_noop,
    TeamRunEvent.memory_update_started.value: handle_workflow_noop,
    RunEvent.memory_update_completed.value: handle_workflow_noop,
    TeamRunEvent.memory_update_completed.value: handle_workflow_noop,
    # Suppress intermediate content from nested agents
    RunEvent.run_content.value: handle_workflow_noop,
    TeamRunEvent.run_content.value: handle_workflow_noop,
    RunEvent.run_intermediate_content.value: handle_workflow_noop,
    TeamRunEvent.run_intermediate_content.value: handle_workflow_noop,
    # Capture step output for final rendering
    WorkflowRunEvent.step_output.value: handle_workflow_step_output,
    # Nested agent runs complete inside workflow steps — if we honored these,
    # task cards would resolve before the parent step finishes.
    RunEvent.run_completed.value: handle_workflow_noop,
    TeamRunEvent.run_completed.value: handle_workflow_noop,
    RunEvent.run_error.value: handle_workflow_noop,
    TeamRunEvent.run_error.value: handle_workflow_noop,
    RunEvent.run_cancelled.value: handle_workflow_noop,
    TeamRunEvent.run_cancelled.value: handle_workflow_noop,
}
