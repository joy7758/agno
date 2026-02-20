from typing import Any, Callable, Coroutine, Literal, Optional

from agno.agent import RunEvent
from agno.os.interfaces.slack.helpers import extract_sources, member_name, task_id
from agno.os.interfaces.slack.state import StreamState
from agno.run.team import TeamRunEvent
from agno.run.workflow import WorkflowRunEvent

_BOT_SUBTYPES = frozenset({"bot_message", "bot_add", "bot_remove", "bot_enable", "bot_disable"})

_MAX_OUTPUT = 200

_TOOL_LABELS: dict[str, str] = {
    "web_search": "Searching the web",
    "search_news": "Searching news",
    "search_exa": "Searching Exa",
    "get_top_hackernews_stories": "Checking HackerNews",
    "get_hackernews_story": "Reading HackerNews story",
    "get_hackernews_user": "Looking up HackerNews user",
    "delegate_task_to_member": "Delegating to {member}",
    "search_messages": "Searching Slack messages",
    "get_thread": "Reading Slack thread",
    "get_user_info": "Looking up user info",
    "list_users": "Listing users",
    "send_message": "Sending message",
    "send_message_thread": "Sending thread reply",
    "upload_file": "Uploading file",
    "download_file_bytes": "Downloading file",
    "read_url": "Reading URL",
    "get_result_from_webpage": "Reading webpage",
}


def _humanize_tool(tool_name: str, tool_args: dict | None = None) -> str:
    template = _TOOL_LABELS.get(tool_name)
    if template:
        if "{member}" in template and tool_args:
            member = tool_args.get("member_id") or tool_args.get("member_name") or ""
            return template.replace("{member}", member.replace("_", " ").title() if member else "member")
        return template
    return tool_name.replace("_", " ").title()


def _truncate(text: str, limit: int = _MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


HandlerAction = Literal["continue", "break"]
HandlerFn = Callable[..., Coroutine[Any, Any, Optional[HandlerAction]]]


async def _emit_task(
    stream: Any,
    task_id: str,
    title: str,
    status: str,
    *,
    output: str | None = None,
    sources: list | None = None,
) -> None:
    chunk: dict = {"type": "task_update", "id": task_id, "title": title, "status": status}
    if output:
        chunk["output"] = _truncate(output)
    if sources:
        chunk["sources"] = sources[:5]
    await stream.append(chunks=[chunk])


# ---------------------------------------------------------------------------
# Reasoning
# ---------------------------------------------------------------------------


async def handle_reasoning_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    key = f"reasoning_{state.reasoning_round}"
    state.track_task(key, "Reasoning")
    await _emit_task(stream, key, "Reasoning", "in_progress")
    return "continue"


async def handle_reasoning_step(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    return "continue"


async def handle_reasoning_delta(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    return "continue"


async def handle_reasoning_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    key = f"reasoning_{state.reasoning_round}"
    state.complete_task(key)
    state.reasoning_round += 1
    await _emit_task(stream, key, "Reasoning", "complete")
    return "continue"


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def handle_tool_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    tool_name = chunk.tool.tool_name if chunk.tool else "a tool"  # type: ignore[union-attr]
    call_id = chunk.tool.tool_call_id if chunk.tool else str(len(state.tool_line_map))  # type: ignore[union-attr]
    tool_args = chunk.tool.tool_args if chunk.tool else {}  # type: ignore[union-attr]
    member = member_name(chunk, state.entity_name)
    label = _humanize_tool(tool_name, tool_args)
    if member:
        label = f"{member}: {label}"
    tid = task_id(member, call_id)  # type: ignore[arg-type]
    state.track_task(tid, label)
    state.tool_line_map[tid] = tid
    await _emit_task(stream, tid, label, "in_progress")
    return None


async def handle_tool_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    call_id = chunk.tool.tool_call_id if chunk.tool else None  # type: ignore[union-attr]
    tool_name = chunk.tool.tool_name if chunk.tool else "tool"  # type: ignore[union-attr]
    tool_args = chunk.tool.tool_args if chunk.tool else {}  # type: ignore[union-attr]
    member = member_name(chunk, state.entity_name)
    tid = task_id(member, call_id) if call_id else None  # type: ignore[arg-type]
    label = _humanize_tool(tool_name, tool_args)
    if member:
        label = f"{member}: {label}"

    tool_result = getattr(chunk.tool, "result", None) if chunk.tool else None  # type: ignore[union-attr]
    errored = chunk.tool.tool_call_error if chunk.tool else False  # type: ignore[union-attr]
    status = "error" if errored else "complete"
    sources: list = []
    output_text = None

    if tool_result:
        result_text = str(tool_result).strip()
        if result_text:
            sources = extract_sources(result_text)
            if sources:
                output_text = f"Found {len(sources)} sources"
            else:
                output_text = _truncate(result_text, 120)

    if tid:
        if tid in state.tool_line_map:
            if errored:
                state.error_task(tid)
            else:
                state.complete_task(tid)
        else:
            state.track_task(tid, label)
            if errored:
                state.error_task(tid)
            else:
                state.complete_task(tid)
        await _emit_task(stream, tid, label, status, output=output_text, sources=sources[:5] if sources else None)

    state.collect_media(chunk)
    return None


async def handle_tool_error(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    tool_name = chunk.tool.tool_name if chunk.tool else "a tool"  # type: ignore[union-attr]
    call_id = chunk.tool.tool_call_id if chunk.tool else f"tool_error_{state.error_count}"  # type: ignore[union-attr]
    tool_args = chunk.tool.tool_args if chunk.tool else {}  # type: ignore[union-attr]
    member = member_name(chunk, state.entity_name)
    tid = task_id(member, call_id)  # type: ignore[arg-type]
    label = _humanize_tool(tool_name, tool_args)
    if member:
        label = f"{member}: {label}"
    error_msg = getattr(chunk, "error", None) or "Tool call failed"
    state.error_count += 1

    if tid in state.tool_line_map:
        state.error_task(tid)
    else:
        state.track_task(tid, label)
        state.error_task(tid)

    await _emit_task(stream, tid, label, "error", output=str(error_msg)[:120])
    return None


# ---------------------------------------------------------------------------
# Content streaming
# ---------------------------------------------------------------------------


async def handle_content(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    if not chunk.content:
        return None
    state.text_buffer += str(chunk.content)
    return None


async def handle_intermediate_content(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    if state.entity_type == "team":
        return None
    return await handle_content(chunk, state, stream)


# ---------------------------------------------------------------------------
# Run lifecycle
# ---------------------------------------------------------------------------


async def handle_run_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.collect_media(chunk)
    if state.progress_started:
        chunks = state.complete_all_pending()
        if chunks:
            await stream.append(chunks=chunks)
    return None


async def handle_run_error(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.error_count += 1
    if state.progress_started:
        chunks = state.error_all_pending()
        if chunks:
            await stream.append(chunks=chunks)
    return "break"


async def handle_run_cancelled(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.error_count += 1
    if state.progress_started:
        chunks = state.error_all_pending()
        if chunks:
            await stream.append(chunks=chunks)
    return "break"


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


async def handle_memory_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.track_task("memory_update", "Updating memory")
    await _emit_task(stream, "memory_update", "Updating memory", "in_progress")
    return "continue"


async def handle_memory_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    memories = getattr(chunk, "memories", None)
    output = f"{len(memories)} memories saved" if memories else None
    state.complete_task("memory_update")
    await _emit_task(stream, "memory_update", "Updating memory", "complete", output=output)
    return "continue"


# ---------------------------------------------------------------------------
# Workflow lifecycle
# ---------------------------------------------------------------------------


async def handle_workflow_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    wf_name = getattr(chunk, "workflow_name", None) or state.entity_name or "Workflow"
    run_id = getattr(chunk, "run_id", None) or "run"
    key = f"wf_run_{run_id}"
    state.track_task(key, f"Workflow: {wf_name}")
    await _emit_task(stream, key, f"Workflow: {wf_name}", "in_progress")
    return "continue"


async def handle_workflow_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    run_id = getattr(chunk, "run_id", None) or "run"
    wf_name = getattr(chunk, "workflow_name", None) or state.entity_name or "Workflow"
    key = f"wf_run_{run_id}"
    state.complete_task(key)
    await _emit_task(stream, key, f"Workflow: {wf_name}", "complete")

    if state.progress_started:
        chunks = state.complete_all_pending()
        if chunks:
            await stream.append(chunks=chunks)

    final = getattr(chunk, "content", None)
    if not final:
        final = state.workflow_final_content
    if final:
        state.text_buffer += str(final)
    return "continue"


async def handle_workflow_error(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.error_count += 1
    if state.progress_started:
        chunks = state.error_all_pending()
        if chunks:
            await stream.append(chunks=chunks)
    return "break"


async def handle_workflow_cancelled(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.error_count += 1
    if state.progress_started:
        chunks = state.error_all_pending()
        if chunks:
            await stream.append(chunks=chunks)
    return "break"


# ---------------------------------------------------------------------------
# Workflow-mode overrides (used by WORKFLOW_DISPATCH)
# ---------------------------------------------------------------------------


async def handle_workflow_content(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.collect_media(chunk)
    return "continue"


async def handle_workflow_step_output(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    content = getattr(chunk, "content", None)
    if content is not None:
        state.workflow_final_content = str(content)
    state.collect_media(chunk)
    return "continue"


async def handle_workflow_run_noop(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    state.collect_media(chunk)
    return "continue"


# ---------------------------------------------------------------------------
# Workflow steps
# ---------------------------------------------------------------------------


async def handle_step_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "step"
    wf_step_id = getattr(chunk, "step_id", None) or ""
    key = f"wf_step_{wf_step_id or step_name}"

    state.track_task(key, step_name)
    await _emit_task(stream, key, step_name, "in_progress")
    return "continue"


async def handle_step_completed_wf(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "step"
    wf_step_id = getattr(chunk, "step_id", None) or ""
    key = f"wf_step_{wf_step_id or step_name}"

    state.complete_task(key)
    state.collect_media(chunk)
    await _emit_task(stream, key, step_name, "complete")
    return "continue"


async def handle_step_error(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "step"
    wf_step_id = getattr(chunk, "step_id", None) or ""
    key = f"wf_step_{wf_step_id or step_name}"
    error_msg = getattr(chunk, "error", None) or "Step failed"

    state.error_task(key)
    await _emit_task(stream, key, step_name, "error", output=str(error_msg)[:120])
    return "continue"


# ---------------------------------------------------------------------------
# Workflow loops
# ---------------------------------------------------------------------------


async def handle_loop_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    max_iter = getattr(chunk, "max_iterations", None)
    title = f"Loop: {step_name}" + (f" (max {max_iter})" if max_iter else "")
    key = f"wf_loop_{loop_key}"

    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return "continue"


async def handle_loop_iter_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    iteration = getattr(chunk, "iteration", 0)
    max_iter = getattr(chunk, "max_iterations", None)
    title = f"Iteration {iteration}" + (f"/{max_iter}" if max_iter else "")
    key = f"wf_loop_{loop_key}_iter_{iteration}"

    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return "continue"


async def handle_loop_iter_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    iteration = getattr(chunk, "iteration", 0)
    key = f"wf_loop_{loop_key}_iter_{iteration}"

    state.complete_task(key)
    await _emit_task(stream, key, f"Iteration {iteration}", "complete")
    return "continue"


async def handle_loop_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "loop"
    loop_key = getattr(chunk, "step_id", None) or step_name
    total = getattr(chunk, "total_iterations", None)
    key = f"wf_loop_{loop_key}"
    title = f"Loop: {step_name}"

    output = f"Completed {total} iterations" if total else None
    state.complete_task(key)
    await _emit_task(stream, key, title, "complete", output=output)
    return "continue"


# ---------------------------------------------------------------------------
# Workflow parallel
# ---------------------------------------------------------------------------


async def handle_parallel_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "parallel"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_parallel_{step_id}"
    title = f"Parallel: {step_name}"
    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return "continue"


async def handle_parallel_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "parallel"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_parallel_{step_id}"
    title = f"Parallel: {step_name}"
    branch_count = getattr(chunk, "branch_count", None)
    output = f"{branch_count} branches completed" if branch_count else None
    state.complete_task(key)
    await _emit_task(stream, key, title, "complete", output=output)
    return "continue"


# ---------------------------------------------------------------------------
# Workflow conditions
# ---------------------------------------------------------------------------


async def handle_condition_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "condition"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_cond_{step_id}"
    title = f"Condition: {step_name}"
    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return "continue"


async def handle_condition_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "condition"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_cond_{step_id}"
    title = f"Condition: {step_name}"
    selected = getattr(chunk, "selected_step", None)
    output = f"Selected: {selected}" if selected else None
    state.complete_task(key)
    await _emit_task(stream, key, title, "complete", output=output)
    return "continue"


# ---------------------------------------------------------------------------
# Workflow routing
# ---------------------------------------------------------------------------


async def handle_router_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "router"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_router_{step_id}"
    title = f"Router: {step_name}"
    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return "continue"


async def handle_router_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "router"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_router_{step_id}"
    title = f"Router: {step_name}"
    state.complete_task(key)
    await _emit_task(stream, key, title, "complete")
    return "continue"


# ---------------------------------------------------------------------------
# Workflow agent delegation
# ---------------------------------------------------------------------------


async def handle_workflow_agent_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    agent_name = getattr(chunk, "agent_name", None) or "agent"
    step_id = getattr(chunk, "step_id", None) or agent_name
    key = f"wf_agent_{step_id}"

    state.track_task(key, f"Running {agent_name}")
    await _emit_task(stream, key, f"Running {agent_name}", "in_progress")
    return "continue"


async def handle_workflow_agent_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    agent_name = getattr(chunk, "agent_name", None) or "agent"
    step_id = getattr(chunk, "step_id", None) or agent_name
    key = f"wf_agent_{step_id}"

    state.complete_task(key)
    state.collect_media(chunk)
    await _emit_task(stream, key, f"Running {agent_name}", "complete")
    return "continue"


# ---------------------------------------------------------------------------
# Workflow steps execution (container for sequential steps)
# ---------------------------------------------------------------------------


async def handle_steps_execution_started(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "steps"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_steps_{step_id}"
    title = f"Steps: {step_name}"
    state.track_task(key, title)
    await _emit_task(stream, key, title, "in_progress")
    return "continue"


async def handle_steps_execution_completed(chunk: Any, state: StreamState, stream: Any) -> Optional[HandlerAction]:
    step_name = getattr(chunk, "step_name", None) or "steps"
    step_id = getattr(chunk, "step_id", None) or step_name
    key = f"wf_steps_{step_id}"
    title = f"Steps: {step_name}"
    state.complete_task(key)
    await _emit_task(stream, key, title, "complete")
    return "continue"


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

DISPATCH: dict[str, HandlerFn] = {
    # Reasoning
    RunEvent.reasoning_started.value: handle_reasoning_started,
    TeamRunEvent.reasoning_started.value: handle_reasoning_started,
    RunEvent.reasoning_step.value: handle_reasoning_step,
    TeamRunEvent.reasoning_step.value: handle_reasoning_step,
    RunEvent.reasoning_content_delta.value: handle_reasoning_delta,
    TeamRunEvent.reasoning_content_delta.value: handle_reasoning_delta,
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
    RunEvent.run_error.value: handle_run_error,
    TeamRunEvent.run_error.value: handle_run_error,
    RunEvent.run_cancelled.value: handle_run_cancelled,
    TeamRunEvent.run_cancelled.value: handle_run_cancelled,
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
    WorkflowRunEvent.workflow_error.value: handle_workflow_error,
    WorkflowRunEvent.workflow_cancelled.value: handle_workflow_cancelled,
    # Workflow steps
    WorkflowRunEvent.step_started.value: handle_step_started,
    WorkflowRunEvent.step_completed.value: handle_step_completed_wf,
    WorkflowRunEvent.step_error.value: handle_step_error,
    # Workflow loops
    WorkflowRunEvent.loop_execution_started.value: handle_loop_started,
    WorkflowRunEvent.loop_iteration_started.value: handle_loop_iter_started,
    WorkflowRunEvent.loop_iteration_completed.value: handle_loop_iter_completed,
    WorkflowRunEvent.loop_execution_completed.value: handle_loop_completed,
    # Workflow parallel
    WorkflowRunEvent.parallel_execution_started.value: handle_parallel_started,
    WorkflowRunEvent.parallel_execution_completed.value: handle_parallel_completed,
    # Workflow conditions
    WorkflowRunEvent.condition_execution_started.value: handle_condition_started,
    WorkflowRunEvent.condition_execution_completed.value: handle_condition_completed,
    # Workflow routing
    WorkflowRunEvent.router_execution_started.value: handle_router_started,
    WorkflowRunEvent.router_execution_completed.value: handle_router_completed,
    # Workflow agent delegation
    WorkflowRunEvent.workflow_agent_started.value: handle_workflow_agent_started,
    WorkflowRunEvent.workflow_agent_completed.value: handle_workflow_agent_completed,
    # Workflow steps execution (sequential container)
    WorkflowRunEvent.steps_execution_started.value: handle_steps_execution_started,
    WorkflowRunEvent.steps_execution_completed.value: handle_steps_execution_completed,
}

# Workflow-mode dispatch: shows only workflow structure + final output
WORKFLOW_DISPATCH: dict[str, HandlerFn] = {
    **DISPATCH,
    # Suppress nested agent reasoning cards
    RunEvent.reasoning_started.value: handle_workflow_run_noop,
    TeamRunEvent.reasoning_started.value: handle_workflow_run_noop,
    RunEvent.reasoning_step.value: handle_workflow_run_noop,
    TeamRunEvent.reasoning_step.value: handle_workflow_run_noop,
    RunEvent.reasoning_content_delta.value: handle_workflow_run_noop,
    TeamRunEvent.reasoning_content_delta.value: handle_workflow_run_noop,
    RunEvent.reasoning_completed.value: handle_workflow_run_noop,
    TeamRunEvent.reasoning_completed.value: handle_workflow_run_noop,
    # Suppress nested agent tool cards
    RunEvent.tool_call_started.value: handle_workflow_run_noop,
    TeamRunEvent.tool_call_started.value: handle_workflow_run_noop,
    RunEvent.tool_call_completed.value: handle_workflow_run_noop,
    TeamRunEvent.tool_call_completed.value: handle_workflow_run_noop,
    RunEvent.tool_call_error.value: handle_workflow_run_noop,
    TeamRunEvent.tool_call_error.value: handle_workflow_run_noop,
    # Suppress nested agent memory cards
    RunEvent.memory_update_started.value: handle_workflow_run_noop,
    TeamRunEvent.memory_update_started.value: handle_workflow_run_noop,
    RunEvent.memory_update_completed.value: handle_workflow_run_noop,
    TeamRunEvent.memory_update_completed.value: handle_workflow_run_noop,
    # Suppress intermediate content from nested agents
    RunEvent.run_content.value: handle_workflow_content,
    TeamRunEvent.run_content.value: handle_workflow_content,
    RunEvent.run_intermediate_content.value: handle_workflow_content,
    TeamRunEvent.run_intermediate_content.value: handle_workflow_content,
    # Capture step output for final rendering
    WorkflowRunEvent.step_output.value: handle_workflow_step_output,
    # No-op nested run lifecycle to prevent premature card completion
    RunEvent.run_completed.value: handle_workflow_run_noop,
    TeamRunEvent.run_completed.value: handle_workflow_run_noop,
    RunEvent.run_error.value: handle_workflow_run_noop,
    TeamRunEvent.run_error.value: handle_workflow_run_noop,
    RunEvent.run_cancelled.value: handle_workflow_run_noop,
    TeamRunEvent.run_cancelled.value: handle_workflow_run_noop,
}
