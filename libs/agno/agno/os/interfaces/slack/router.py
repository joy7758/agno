import asyncio
from typing import Any, Dict, List, Optional, Tuple, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from agno.agent import Agent, RemoteAgent, RunEvent
from agno.media import File, Image
from agno.os.interfaces.slack.security import verify_slack_signature
from agno.run.team import TeamRunEvent
from agno.run.workflow import WorkflowRunEvent
from agno.team import RemoteTeam, Team
from agno.tools.slack import SlackTools
from agno.utils.log import log_error
from agno.workflow import RemoteWorkflow, Workflow

_BOT_SUBTYPES = frozenset({"bot_message", "bot_add", "bot_remove", "bot_enable", "bot_disable"})

# --- Event classification sets (agent + team variants) ---

# Reasoning
_REASONING_STARTED = frozenset({RunEvent.reasoning_started.value, TeamRunEvent.reasoning_started.value})
_REASONING_STEP = frozenset({RunEvent.reasoning_step.value, TeamRunEvent.reasoning_step.value})
_REASONING_CONTENT_DELTA = frozenset(
    {RunEvent.reasoning_content_delta.value, TeamRunEvent.reasoning_content_delta.value}
)
_REASONING_COMPLETED = frozenset({RunEvent.reasoning_completed.value, TeamRunEvent.reasoning_completed.value})

# Tools
_TOOL_STARTED = frozenset({RunEvent.tool_call_started.value, TeamRunEvent.tool_call_started.value})
_TOOL_COMPLETED = frozenset({RunEvent.tool_call_completed.value, TeamRunEvent.tool_call_completed.value})
_TOOL_ERROR = frozenset({RunEvent.tool_call_error.value, TeamRunEvent.tool_call_error.value})

# Content
_CONTENT_EVENTS = frozenset({RunEvent.run_content.value, TeamRunEvent.run_content.value})
_INTERMEDIATE_CONTENT = frozenset(
    {
        RunEvent.run_intermediate_content.value,
        TeamRunEvent.run_intermediate_content.value,
    }
)

# Run lifecycle
_RUN_COMPLETED = frozenset(
    {
        RunEvent.run_completed.value,
        TeamRunEvent.run_completed.value,
        WorkflowRunEvent.step_completed.value,
    }
)
_RUN_ERROR = frozenset({RunEvent.run_error.value, TeamRunEvent.run_error.value})
_RUN_CANCELLED = frozenset({RunEvent.run_cancelled.value, TeamRunEvent.run_cancelled.value})

# Memory
_MEMORY_STARTED = frozenset({RunEvent.memory_update_started.value, TeamRunEvent.memory_update_started.value})
_MEMORY_COMPLETED = frozenset({RunEvent.memory_update_completed.value, TeamRunEvent.memory_update_completed.value})

# Workflow (no team variants — workflow events are their own enum)
_STEP_OUTPUT = frozenset({WorkflowRunEvent.step_output.value})
_WORKFLOW_STARTED = frozenset({WorkflowRunEvent.workflow_started.value})
_WORKFLOW_COMPLETED = frozenset({WorkflowRunEvent.workflow_completed.value})
_WORKFLOW_ERROR = frozenset({WorkflowRunEvent.workflow_error.value})
_WORKFLOW_CANCELLED = frozenset({WorkflowRunEvent.workflow_cancelled.value})
_STEP_STARTED = frozenset({WorkflowRunEvent.step_started.value})
_STEP_COMPLETED_WF = frozenset({WorkflowRunEvent.step_completed.value})
_STEP_ERROR = frozenset({WorkflowRunEvent.step_error.value})
_LOOP_STARTED = frozenset({WorkflowRunEvent.loop_execution_started.value})
_LOOP_ITER_STARTED = frozenset({WorkflowRunEvent.loop_iteration_started.value})
_LOOP_ITER_COMPLETED = frozenset({WorkflowRunEvent.loop_iteration_completed.value})
_LOOP_COMPLETED = frozenset({WorkflowRunEvent.loop_execution_completed.value})
_PARALLEL_STARTED = frozenset({WorkflowRunEvent.parallel_execution_started.value})
_PARALLEL_COMPLETED = frozenset({WorkflowRunEvent.parallel_execution_completed.value})
_CONDITION_STARTED = frozenset({WorkflowRunEvent.condition_execution_started.value})
_CONDITION_COMPLETED = frozenset({WorkflowRunEvent.condition_execution_completed.value})
_ROUTER_STARTED = frozenset({WorkflowRunEvent.router_execution_started.value})
_ROUTER_COMPLETED = frozenset({WorkflowRunEvent.router_execution_completed.value})


def _extract_sources(tool_result: str) -> list:
    """Extract URL sources from a tool result string for Slack task card citations."""
    import json
    import re
    from urllib.parse import urlparse

    sources: list[dict] = []
    seen_urls: set[str] = set()

    def _add(url: str, text: Optional[str] = None) -> None:
        if url in seen_urls:
            return
        seen_urls.add(url)
        if not text:
            text = urlparse(url).netloc or url
        sources.append({"type": "url", "url": url, "text": text})

    # Try JSON parsing first (search tools return structured JSON)
    try:
        data = json.loads(tool_result)
        items: list = []
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict):
            for key in ("results", "citations", "data", "organic"):
                if key in data and isinstance(data[key], list):
                    items = data[key]
                    break
        for item in items:
            if isinstance(item, dict):
                url = item.get("url") or item.get("link") or item.get("href")
                title = item.get("title") or item.get("text") or item.get("name")
                if url and isinstance(url, str) and url.startswith("http"):
                    _add(url, title if isinstance(title, str) else None)
        if sources:
            return sources[:5]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: extract URLs via regex
    url_pattern = r"https?://[^\s<>\"')\],}]+"
    for match in re.findall(url_pattern, tool_result):
        _add(match.rstrip("."))
        if len(sources) >= 5:
            break

    return sources


def _task_id(agent_name: Optional[str], base_id: str) -> str:
    """Generate a unique task card ID, optionally prefixed by agent name for team members."""
    if agent_name:
        safe = agent_name.lower().replace(" ", "_")[:20]
        return f"{safe}_{base_id}"
    return base_id


def _member_name(chunk: Any, entity_name: str) -> Optional[str]:
    """Extract team member name from a chunk if it differs from the top-level entity."""
    name = getattr(chunk, "agent_name", None)
    if name and isinstance(name, str) and name != entity_name:
        return name
    return None


class SlackEventResponse(BaseModel):
    """Response model for Slack event processing"""

    status: str = Field(default="ok", description="Processing status")


class SlackChallengeResponse(BaseModel):
    """Response model for Slack URL verification challenge"""

    challenge: str = Field(description="Challenge string to echo back to Slack")


def attach_routes(
    router: APIRouter,
    agent: Optional[Union[Agent, RemoteAgent]] = None,
    team: Optional[Union[Team, RemoteTeam]] = None,
    workflow: Optional[Union[Workflow, RemoteWorkflow]] = None,
    reply_to_mentions_only: bool = True,
    token: Optional[str] = None,
    signing_secret: Optional[str] = None,
    streaming: bool = False,
    loading_messages: Optional[List[str]] = None,
    task_display_mode: str = "plan",
    buffer_size: int = 256,
    initial_buffer_size: int = 1,
    suggested_prompts: Optional[List[Dict[str, str]]] = None,
) -> APIRouter:
    entity = agent or team or workflow
    entity_type = "agent" if agent else "team" if team else "workflow" if workflow else "unknown"
    raw_name = getattr(entity, "name", None)
    entity_name = raw_name if isinstance(raw_name, str) else entity_type
    op_suffix = entity_name.lower().replace(" ", "_")

    slack_tools = SlackTools(token=token)

    @router.post(
        "/events",
        operation_id=f"slack_events_{op_suffix}",
        name="slack_events",
        description="Process incoming Slack events",
        response_model=Union[SlackChallengeResponse, SlackEventResponse],
        response_model_exclude_none=True,
        responses={
            200: {"description": "Event processed successfully"},
            400: {"description": "Missing Slack headers"},
            403: {"description": "Invalid Slack signature"},
        },
    )
    async def slack_events(request: Request, background_tasks: BackgroundTasks):
        body = await request.body()
        timestamp = request.headers.get("X-Slack-Request-Timestamp")
        slack_signature = request.headers.get("X-Slack-Signature", "")

        if not timestamp or not slack_signature:
            raise HTTPException(status_code=400, detail="Missing Slack headers")

        if not verify_slack_signature(body, timestamp, slack_signature, signing_secret=signing_secret):
            raise HTTPException(status_code=403, detail="Invalid signature")

        if request.headers.get("X-Slack-Retry-Num"):
            return SlackEventResponse(status="ok")

        data = await request.json()

        if data.get("type") == "url_verification":
            return SlackChallengeResponse(challenge=data.get("challenge"))

        if "event" in data:
            event = data["event"]
            event_type = event.get("type")

            if event_type == "assistant_thread_started" and streaming:
                background_tasks.add_task(_handle_thread_started, event, slack_tools.token or "")
            elif (
                event.get("bot_id")
                or (event.get("message") or {}).get("bot_id")
                or event.get("subtype") in _BOT_SUBTYPES
            ):
                pass
            elif streaming:
                background_tasks.add_task(_stream_slack_response, data)
            else:
                background_tasks.add_task(_process_slack_event, event)

        return SlackEventResponse(status="ok")

    def _should_respond(event: dict) -> bool:
        event_type = event.get("type")
        if event_type not in ("app_mention", "message"):
            return False

        channel_type = event.get("channel_type", "")
        is_dm = channel_type == "im"

        if reply_to_mentions_only and event_type == "message" and not is_dm:
            return False
        return True

    def _extract_event_context(event: dict) -> Dict[str, Any]:
        return {
            "message_text": event.get("text", ""),
            "channel_id": event.get("channel", ""),
            "user": event.get("user", ""),
            "ts": event.get("thread_ts") or event.get("ts", ""),
        }

    def _fetch_mention_files(event: dict, channel_id: str, ts: str) -> dict:
        """app_mention events don't include file attachments — fetch the full message."""
        if event.get("type") != "app_mention" or event.get("files"):
            return event
        try:
            result = slack_tools.client.conversations_history(channel=channel_id, latest=ts, inclusive=True, limit=1)
            messages: list = result.get("messages", [])
            if messages and messages[0].get("files"):
                return {**event, "files": messages[0]["files"]}
        except Exception as e:
            log_error(f"Failed to fetch files for app_mention: {e}")
        return event

    async def _process_slack_event(event: dict):
        if not _should_respond(event):
            return

        ctx = _extract_event_context(event)
        event = _fetch_mention_files(event, ctx["channel_id"], ctx["ts"])
        files, images = _download_event_files(slack_tools, event)

        response = None
        if agent:
            response = await agent.arun(  # type: ignore[misc]
                ctx["message_text"],
                user_id=ctx["user"],
                session_id=ctx["ts"],
                files=files if files else None,
                images=images if images else None,
            )
        elif team:
            response = await team.arun(
                ctx["message_text"],
                user_id=ctx["user"],
                session_id=ctx["ts"],
                files=files if files else None,
                images=images if images else None,
            )  # type: ignore
        elif workflow:
            response = await workflow.arun(
                ctx["message_text"],
                user_id=ctx["user"],
                session_id=ctx["ts"],
                files=files if files else None,
                images=images if images else None,
            )  # type: ignore

        if response:
            if response.status == "ERROR":
                log_error(f"Error processing message: {response.content}")
                _send_slack_message(
                    slack_tools,
                    channel=ctx["channel_id"],
                    message="Sorry, there was an error processing your message. Please try again later.",
                    thread_ts=ctx["ts"],
                )
                return

            if hasattr(response, "reasoning_content") and response.reasoning_content:
                rc = response.reasoning_content
                formatted = f"*Reasoning:*\n> {rc.replace(chr(10), chr(10) + '> ')}"
                _send_slack_message(
                    slack_tools,
                    channel=ctx["channel_id"],
                    message=formatted,
                    thread_ts=ctx["ts"],
                )

            _send_slack_message(
                slack_tools, channel=ctx["channel_id"], message=response.content or "", thread_ts=ctx["ts"]
            )
            _upload_response_media(slack_tools, response, ctx["channel_id"], ctx["ts"])

    async def _stream_slack_response(data: dict):
        from slack_sdk.web.async_client import AsyncWebClient

        event = data["event"]
        if not _should_respond(event):
            return

        ctx = _extract_event_context(event)

        # Streaming requires thread_ts for startStream/appendStream/stopStream.
        # For top-level messages (app_mention without thread), fall back to
        # non-streaming. For DMs without thread_ts in assistant mode, these
        # are duplicate deliveries — ignore them.
        if not event.get("thread_ts"):
            channel_type = event.get("channel_type", "")
            if channel_type == "im":
                return
            await _process_slack_event(event)
            return

        is_assistant_thread = True

        team_id = data.get("team_id") or event.get("team") or None
        # recipient_user_id must be the HUMAN user, not the bot.
        # Slack streams content in real-time only to the recipient.
        user_id = ctx.get("user") or event.get("user")

        async_client = AsyncWebClient(token=slack_tools.token)
        stream_ts: Optional[str] = None
        stream_started = False

        try:
            # Set status before any file I/O (only in assistant threads)
            if is_assistant_thread:
                try:
                    status_kwargs: Dict[str, Any] = {
                        "channel_id": ctx["channel_id"],
                        "thread_ts": ctx["ts"],
                        "status": "Thinking...",
                    }
                    if loading_messages:
                        status_kwargs["loading_messages"] = loading_messages
                    await async_client.assistant_threads_setStatus(**status_kwargs)
                except Exception:
                    pass

            # File downloads happen while user sees "Thinking..."
            event = _fetch_mention_files(event, ctx["channel_id"], ctx["ts"])
            files, images = _download_event_files(slack_tools, event)

            # Buffer tool chunks until real content triggers stream start.
            # This prevents an empty bubble from appearing while tools run.
            pending_tool_chunks: list = []

            async def _ensure_stream_started(initial_text: str = "") -> str:
                nonlocal stream_ts, stream_started
                if stream_started:
                    return stream_ts  # type: ignore[return-value]
                start_kwargs: Dict[str, Any] = {
                    "channel": ctx["channel_id"],
                    "thread_ts": ctx["ts"],
                    "recipient_team_id": team_id,
                    "recipient_user_id": user_id,
                }
                buffered_chunks: list = list(pending_tool_chunks)
                pending_tool_chunks.clear()
                # Slack rejects startStream with both markdown_text and chunks.
                # If we have buffered chunks, start with those and append text after.
                if buffered_chunks:
                    start_kwargs["chunks"] = buffered_chunks
                elif initial_text:
                    start_kwargs["chunks"] = [{"type": "markdown_text", "text": initial_text}]
                if task_display_mode:
                    start_kwargs["task_display_mode"] = task_display_mode
                start_resp = await async_client.chat_startStream(**start_kwargs)
                stream_ts = start_resp["ts"]
                stream_started = True
                # If we had chunks and also text, send the text now via append.
                if buffered_chunks and initial_text:
                    await async_client.chat_appendStream(
                        channel=ctx["channel_id"],
                        ts=stream_ts,
                        chunks=[{"type": "markdown_text", "text": initial_text}],
                    )
                return stream_ts

            title_set = False
            text_buffer = ""
            first_flush_done = False
            response_stream = None

            if agent:
                response_stream = agent.arun(
                    ctx["message_text"],
                    stream=True,
                    stream_events=True,
                    user_id=ctx["user"],
                    session_id=ctx["ts"],
                    files=files if files else None,
                    images=images if images else None,
                )
            elif team:
                response_stream = team.arun(  # type: ignore[assignment]
                    ctx["message_text"],
                    stream=True,
                    stream_events=True,
                    user_id=ctx["user"],
                    session_id=ctx["ts"],
                    files=files if files else None,
                    images=images if images else None,
                )
            elif workflow:
                response_stream = workflow.arun(  # type: ignore[assignment]
                    ctx["message_text"],
                    stream=True,
                    stream_events=True,
                    user_id=ctx["user"],
                    session_id=ctx["ts"],
                    files=files if files else None,
                    images=images if images else None,
                )

            if response_stream is None:
                return

            tool_tasks: dict[str, str] = {}
            reasoning_step_count = 0
            reasoning_delta_buffer = ""
            reasoning_flushed_len = 0
            error_count = 0
            collected_images: list = []
            collected_videos: list = []
            collected_audio: list = []

            async def _send_chunks(chunks: list) -> None:
                nonlocal stream_started
                if stream_started:
                    await async_client.chat_appendStream(channel=ctx["channel_id"], ts=stream_ts, chunks=chunks)
                else:
                    pending_tool_chunks.extend(chunks)

            async for chunk in response_stream:
                # --- Reasoning events ---
                # Reasoning steps are shown as task cards inside a collapsible
                # plan section. The plan title updates to "Done thinking" on
                # completion; individual steps show their title and reasoning.
                if chunk.event in _REASONING_STARTED:  # type: ignore[union-attr]
                    reasoning_step_count = 0
                    reasoning_flushed_len = 0
                    first_step_id = "reasoning_0"
                    reasoning_chunks = [
                        {"type": "plan_update", "title": "Thinking..."},
                        {"type": "task_update", "id": first_step_id, "title": "Thinking...", "status": "in_progress"},
                    ]
                    if stream_started:
                        await async_client.chat_appendStream(
                            channel=ctx["channel_id"], ts=stream_ts, chunks=reasoning_chunks
                        )
                    else:
                        pending_tool_chunks.extend(reasoning_chunks)
                        await _ensure_stream_started()
                    continue

                if chunk.event in _REASONING_STEP:  # type: ignore[union-attr]
                    reasoning_step_count += 1
                    step = getattr(chunk, "content", None)
                    prev_id = f"reasoning_{reasoning_step_count - 1}"
                    next_id = f"reasoning_{reasoning_step_count}"

                    # Complete the previous step with its title
                    step_title = getattr(step, "title", None) or "Thinking..."
                    prev_chunk: dict = {
                        "type": "task_update",
                        "id": prev_id,
                        "title": step_title,
                        "status": "complete",
                    }
                    reasoning_text = getattr(step, "reasoning", None)
                    if reasoning_text:
                        prev_chunk["details"] = reasoning_text

                    # Start the next step card
                    next_action = getattr(step, "next_action", None)
                    next_title = "Finalizing..." if str(next_action) == "NextAction.FINAL_ANSWER" else "Thinking..."
                    next_chunk: dict = {
                        "type": "task_update",
                        "id": next_id,
                        "title": next_title,
                        "status": "in_progress",
                    }
                    action = getattr(step, "action", None)
                    if action:
                        next_chunk["details"] = action

                    await _send_chunks([prev_chunk, next_chunk])
                    continue

                if chunk.event in _REASONING_COMPLETED:  # type: ignore[union-attr]
                    last_id = f"reasoning_{reasoning_step_count}"
                    last_title = "Thinking..."
                    steps_obj = getattr(chunk, "content", None)
                    if steps_obj is not None:
                        steps_list = getattr(steps_obj, "reasoning_steps", None) or []
                        if steps_list:
                            last_step = steps_list[-1]
                            title = getattr(last_step, "title", None)
                            if title:
                                last_title = title
                    completed_chunk: dict = {
                        "type": "task_update",
                        "id": last_id,
                        "title": last_title,
                        "status": "complete",
                    }
                    unsent = reasoning_delta_buffer[reasoning_flushed_len:]
                    if unsent:
                        completed_chunk["details"] = unsent
                    reasoning_delta_buffer = ""
                    reasoning_flushed_len = 0
                    await _send_chunks(
                        [
                            completed_chunk,
                            {"type": "plan_update", "title": "Thinking..."},
                        ]
                    )
                    continue

                # --- Tool events ---
                # Each tool call gets its own task card. The plan title updates
                # to show the current tool name.
                if chunk.event in _TOOL_STARTED:  # type: ignore[union-attr]
                    tool_name = chunk.tool.tool_name if chunk.tool else "a tool"  # type: ignore[union-attr]
                    task_id = chunk.tool.tool_call_id if chunk.tool else str(len(tool_tasks))  # type: ignore[union-attr]
                    tool_args = chunk.tool.tool_args if chunk.tool else {}  # type: ignore[union-attr]
                    member = _member_name(chunk, entity_name)
                    display_name = f"{member}: {tool_name}" if member else tool_name
                    task_id = _task_id(member, task_id)  # type: ignore[arg-type]
                    tool_tasks[task_id] = display_name  # type: ignore[index, assignment]

                    details = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else None
                    task_chunk: dict = {
                        "type": "task_update",
                        "id": task_id,
                        "title": display_name,
                        "status": "in_progress",
                    }
                    if details:
                        task_chunk["details"] = details

                    await _send_chunks(
                        [
                            {"type": "plan_update", "title": display_name},
                            task_chunk,
                        ]
                    )

                elif chunk.event in _TOOL_COMPLETED:  # type: ignore[union-attr]
                    task_id = chunk.tool.tool_call_id if chunk.tool else None  # type: ignore[union-attr]
                    member = _member_name(chunk, entity_name)
                    task_id = _task_id(member, task_id) if task_id else None  # type: ignore[arg-type]
                    if task_id and task_id in tool_tasks:
                        errored = chunk.tool.tool_call_error if chunk.tool else False  # type: ignore[union-attr]
                        task_chunk = {
                            "type": "task_update",
                            "id": task_id,
                            "title": tool_tasks[task_id],
                            "status": "error" if errored else "complete",
                        }
                        tool_result = getattr(chunk.tool, "result", None) if chunk.tool else None  # type: ignore[union-attr]
                        if tool_result:
                            result_text = str(tool_result).strip()
                            if result_text:
                                sources = _extract_sources(result_text)
                                if sources:
                                    task_chunk["sources"] = sources
                                else:
                                    task_chunk["output"] = result_text
                        await _send_chunks([task_chunk])

                    # Collect media from tool completed events for upload after stream ends.
                    for img in getattr(chunk, "images", None) or []:
                        if img not in collected_images:
                            collected_images.append(img)
                    for vid in getattr(chunk, "videos", None) or []:
                        if vid not in collected_videos:
                            collected_videos.append(vid)
                    for aud in getattr(chunk, "audio", None) or []:
                        if aud not in collected_audio:
                            collected_audio.append(aud)

                elif chunk.event in _TOOL_ERROR:  # type: ignore[union-attr]
                    tool_name = chunk.tool.tool_name if chunk.tool else "a tool"  # type: ignore[union-attr]
                    task_id = chunk.tool.tool_call_id if chunk.tool else f"tool_error_{error_count}"  # type: ignore[union-attr]
                    error_msg = getattr(chunk, "error", None) or "Tool call failed"
                    error_count += 1
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": task_id,
                                "title": tool_name,
                                "status": "error",
                                "output": str(error_msg),
                            }
                        ]
                    )

                # --- Reasoning content delta ---
                if chunk.event in _REASONING_CONTENT_DELTA:  # type: ignore[union-attr]
                    delta = getattr(chunk, "reasoning_content", None) or ""
                    if delta:
                        reasoning_delta_buffer += delta
                        unsent = reasoning_delta_buffer[reasoning_flushed_len:]
                        if len(unsent) >= 200:
                            await _send_chunks(
                                [
                                    {
                                        "type": "task_update",
                                        "id": f"reasoning_{reasoning_step_count}",
                                        "title": "Thinking...",
                                        "status": "in_progress",
                                        "details": unsent,
                                    }
                                ]
                            )
                            reasoning_flushed_len = len(reasoning_delta_buffer)
                    continue

                # --- Run-level error / cancel ---
                if chunk.event in _RUN_ERROR:  # type: ignore[union-attr]
                    error_msg = getattr(chunk, "content", None) or "An error occurred"
                    error_id = f"error_{error_count}"
                    error_count += 1
                    if not stream_started:
                        pending_tool_chunks.append({"type": "plan_update", "title": "Error"})
                        pending_tool_chunks.append(
                            {
                                "type": "task_update",
                                "id": error_id,
                                "title": "Error",
                                "status": "error",
                                "output": str(error_msg),
                            }
                        )
                        await _ensure_stream_started()
                    else:
                        await _send_chunks(
                            [
                                {"type": "plan_update", "title": "Error"},
                                {
                                    "type": "task_update",
                                    "id": error_id,
                                    "title": "Error",
                                    "status": "error",
                                    "output": str(error_msg),
                                },
                            ]
                        )
                    break

                if chunk.event in _RUN_CANCELLED:  # type: ignore[union-attr]
                    reason = getattr(chunk, "reason", None) or "Run was cancelled"
                    cancel_id = f"error_{error_count}"
                    error_count += 1
                    if not stream_started:
                        pending_tool_chunks.append({"type": "plan_update", "title": "Cancelled"})
                        pending_tool_chunks.append(
                            {
                                "type": "task_update",
                                "id": cancel_id,
                                "title": "Cancelled",
                                "status": "error",
                                "output": str(reason),
                            }
                        )
                        await _ensure_stream_started()
                    else:
                        await _send_chunks(
                            [
                                {"type": "plan_update", "title": "Cancelled"},
                                {
                                    "type": "task_update",
                                    "id": cancel_id,
                                    "title": "Cancelled",
                                    "status": "error",
                                    "output": str(reason),
                                },
                            ]
                        )
                    break

                # --- Memory events ---
                if chunk.event in _MEMORY_STARTED:  # type: ignore[union-attr]
                    await _send_chunks(
                        [
                            {"type": "plan_update", "title": "Updating memory..."},
                            {
                                "type": "task_update",
                                "id": "memory_update",
                                "title": "Updating memory...",
                                "status": "in_progress",
                            },
                        ]
                    )
                    continue

                if chunk.event in _MEMORY_COMPLETED:  # type: ignore[union-attr]
                    memories = getattr(chunk, "memories", None)
                    mem_output = f"{len(memories)} memories saved" if memories else None
                    mem_chunk: dict = {
                        "type": "task_update",
                        "id": "memory_update",
                        "title": "Memory updated",
                        "status": "complete",
                    }
                    if mem_output:
                        mem_chunk["output"] = mem_output
                    await _send_chunks([mem_chunk])
                    continue

                # --- Workflow events ---
                if chunk.event in _WORKFLOW_STARTED:  # type: ignore[union-attr]
                    wf_name = getattr(chunk, "workflow_name", None) or "workflow"
                    await _send_chunks([{"type": "plan_update", "title": f"Running {wf_name}..."}])
                    continue

                if chunk.event in _WORKFLOW_COMPLETED:  # type: ignore[union-attr]
                    continue

                if chunk.event in _WORKFLOW_ERROR:  # type: ignore[union-attr]
                    error_msg = getattr(chunk, "error", None) or "Workflow failed"
                    error_id = f"wf_error_{error_count}"
                    error_count += 1
                    await _send_chunks(
                        [
                            {"type": "plan_update", "title": "Error"},
                            {
                                "type": "task_update",
                                "id": error_id,
                                "title": "Workflow error",
                                "status": "error",
                                "output": str(error_msg),
                            },
                        ]
                    )
                    break

                if chunk.event in _WORKFLOW_CANCELLED:  # type: ignore[union-attr]
                    reason = getattr(chunk, "reason", None) or "Workflow cancelled"
                    cancel_id = f"wf_cancel_{error_count}"
                    error_count += 1
                    await _send_chunks(
                        [
                            {"type": "plan_update", "title": "Cancelled"},
                            {
                                "type": "task_update",
                                "id": cancel_id,
                                "title": "Workflow cancelled",
                                "status": "error",
                                "output": str(reason),
                            },
                        ]
                    )
                    break

                if chunk.event in _STEP_STARTED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "step"
                    step_id = f"wf_step_{step_name}"
                    await _send_chunks(
                        [
                            {"type": "plan_update", "title": step_name},
                            {"type": "task_update", "id": step_id, "title": step_name, "status": "in_progress"},
                        ]
                    )
                    continue

                if chunk.event in _STEP_COMPLETED_WF:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "step"
                    step_id = f"wf_step_{step_name}"
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": step_id,
                                "title": step_name,
                                "status": "complete",
                            }
                        ]
                    )
                    for img in getattr(chunk, "images", None) or []:
                        if img not in collected_images:
                            collected_images.append(img)
                    for vid in getattr(chunk, "videos", None) or []:
                        if vid not in collected_videos:
                            collected_videos.append(vid)
                    for aud in getattr(chunk, "audio", None) or []:
                        if aud not in collected_audio:
                            collected_audio.append(aud)
                    continue

                if chunk.event in _STEP_ERROR:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "step"
                    step_id = f"wf_step_{step_name}"
                    error_msg = getattr(chunk, "error", None) or "Step failed"
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": step_id,
                                "title": step_name,
                                "status": "error",
                                "output": str(error_msg),
                            }
                        ]
                    )
                    continue

                if chunk.event in _LOOP_STARTED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "loop"
                    max_iter = getattr(chunk, "max_iterations", None)
                    loop_title = f"Loop: {step_name}" + (f" (max {max_iter})" if max_iter else "")
                    await _send_chunks(
                        [
                            {"type": "plan_update", "title": loop_title},
                            {
                                "type": "task_update",
                                "id": f"wf_loop_{step_name}",
                                "title": loop_title,
                                "status": "in_progress",
                            },
                        ]
                    )
                    continue

                if chunk.event in _LOOP_ITER_STARTED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "loop"
                    iteration = getattr(chunk, "iteration", 0)
                    max_iter = getattr(chunk, "max_iterations", None)
                    iter_title = f"Iteration {iteration}" + (f"/{max_iter}" if max_iter else "")
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": f"wf_loop_{step_name}_iter_{iteration}",
                                "title": iter_title,
                                "status": "in_progress",
                            }
                        ]
                    )
                    continue

                if chunk.event in _LOOP_ITER_COMPLETED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "loop"
                    iteration = getattr(chunk, "iteration", 0)
                    should_continue = getattr(chunk, "should_continue", True)
                    status = "complete" if should_continue else "complete"
                    iter_title = f"Iteration {iteration}"
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": f"wf_loop_{step_name}_iter_{iteration}",
                                "title": iter_title,
                                "status": status,
                            }
                        ]
                    )
                    continue

                if chunk.event in _LOOP_COMPLETED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "loop"
                    total = getattr(chunk, "total_iterations", None)
                    loop_output = f"Completed {total} iterations" if total else None
                    loop_chunk: dict = {
                        "type": "task_update",
                        "id": f"wf_loop_{step_name}",
                        "title": f"Loop: {step_name}",
                        "status": "complete",
                    }
                    if loop_output:
                        loop_chunk["output"] = loop_output
                    await _send_chunks([loop_chunk])
                    continue

                if chunk.event in _PARALLEL_STARTED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "parallel"
                    count = getattr(chunk, "parallel_step_count", None) or 0
                    par_title = f"Running {count} steps in parallel"
                    await _send_chunks(
                        [
                            {"type": "plan_update", "title": par_title},
                            {
                                "type": "task_update",
                                "id": f"wf_parallel_{step_name}",
                                "title": par_title,
                                "status": "in_progress",
                            },
                        ]
                    )
                    continue

                if chunk.event in _PARALLEL_COMPLETED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "parallel"
                    count = getattr(chunk, "parallel_step_count", None) or 0
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": f"wf_parallel_{step_name}",
                                "title": f"{count} parallel steps completed",
                                "status": "complete",
                            }
                        ]
                    )
                    continue

                if chunk.event in _CONDITION_STARTED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "condition"
                    cond_result = getattr(chunk, "condition_result", None)
                    cond_title = f"Evaluating: {step_name}"
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": f"wf_cond_{step_name}",
                                "title": cond_title,
                                "status": "in_progress",
                                **({"details": f"Result: {cond_result}"} if cond_result is not None else {}),
                            }
                        ]
                    )
                    continue

                if chunk.event in _CONDITION_COMPLETED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "condition"
                    branch = getattr(chunk, "branch", None) or ""
                    await _send_chunks(
                        [
                            {
                                "type": "task_update",
                                "id": f"wf_cond_{step_name}",
                                "title": f"Condition: {step_name} ({branch})",
                                "status": "complete",
                            }
                        ]
                    )
                    continue

                if chunk.event in _ROUTER_STARTED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "router"
                    selected = getattr(chunk, "selected_steps", None) or []
                    details = ", ".join(str(s) for s in selected) if selected else None
                    router_chunk: dict = {
                        "type": "task_update",
                        "id": f"wf_router_{step_name}",
                        "title": f"Routing: {step_name}",
                        "status": "in_progress",
                    }
                    if details:
                        router_chunk["details"] = f"Selected: {details}"
                    await _send_chunks([router_chunk])
                    continue

                if chunk.event in _ROUTER_COMPLETED:  # type: ignore[union-attr]
                    step_name = getattr(chunk, "step_name", None) or "router"
                    executed = getattr(chunk, "executed_steps", None)
                    router_chunk = {
                        "type": "task_update",
                        "id": f"wf_router_{step_name}",
                        "title": f"Routed: {step_name}",
                        "status": "complete",
                    }
                    if executed:
                        router_chunk["output"] = f"Executed {executed} steps"
                    await _send_chunks([router_chunk])
                    continue

                # --- Collect media from run/team/workflow completed events ---
                if getattr(chunk, "event", None) in _RUN_COMPLETED:
                    for img in getattr(chunk, "images", None) or []:
                        if img not in collected_images:
                            collected_images.append(img)
                    for vid in getattr(chunk, "videos", None) or []:
                        if vid not in collected_videos:
                            collected_videos.append(vid)
                    for aud in getattr(chunk, "audio", None) or []:
                        if aud not in collected_audio:
                            collected_audio.append(aud)

                # --- Content streaming ---
                content_text: Optional[str] = None
                if chunk.event in _CONTENT_EVENTS and chunk.content:  # type: ignore[union-attr]
                    content_text = str(chunk.content)
                elif chunk.event in _INTERMEDIATE_CONTENT and chunk.content:  # type: ignore[union-attr]
                    content_text = str(chunk.content)
                elif chunk.event in _STEP_OUTPUT and chunk.content:  # type: ignore[union-attr]
                    content_text = str(chunk.content)

                if content_text:
                    if is_assistant_thread and not title_set:
                        title = ctx["message_text"][:50].strip() or "New conversation"

                        async def _set_title():
                            try:
                                await async_client.assistant_threads_setTitle(
                                    channel_id=ctx["channel_id"],
                                    thread_ts=ctx["ts"],
                                    title=title,
                                )
                            except Exception:
                                pass

                        asyncio.create_task(_set_title())
                        title_set = True

                    text_buffer += content_text
                    threshold = buffer_size if first_flush_done else initial_buffer_size
                    # Guard: Slack limits markdown_text to 12K per append. Flush at 10K.
                    if len(text_buffer) >= 10000:
                        threshold = 0
                    if len(text_buffer) >= threshold:
                        if not stream_started:
                            await _ensure_stream_started(initial_text=text_buffer)
                        else:
                            await async_client.chat_appendStream(
                                channel=ctx["channel_id"],
                                ts=stream_ts,
                                chunks=[{"type": "markdown_text", "text": text_buffer}],
                            )
                        text_buffer = ""
                        first_flush_done = True

            # Flush any buffered chunks/text that weren't sent yet.
            if not stream_started and (pending_tool_chunks or text_buffer):
                await _ensure_stream_started(initial_text=text_buffer or None)
                text_buffer = ""

            if stream_started:
                assert stream_ts is not None
                stop_chunks: list = []
                if text_buffer:
                    stop_chunks.append({"type": "markdown_text", "text": text_buffer})
                stop_kwargs: Dict[str, Any] = {
                    "channel": ctx["channel_id"],
                    "ts": stream_ts,
                }
                if stop_chunks:
                    stop_kwargs["chunks"] = stop_chunks
                await async_client.chat_stopStream(**stop_kwargs)

            # Upload any media (images/videos/audio) collected during streaming.
            media_items: list[tuple[list, str]] = [
                (collected_images, "image.png"),
                (collected_videos, "video.mp4"),
                (collected_audio, "audio.mp3"),
            ]
            for items, default_name in media_items:
                for item in items:
                    try:
                        content_bytes = item.get_content_bytes()
                        if content_bytes:
                            slack_tools.upload_file(
                                channel=ctx["channel_id"],
                                content=content_bytes,
                                filename=getattr(item, "filename", None) or default_name,
                                thread_ts=ctx["ts"],
                            )
                    except Exception as e:
                        log_error(f"Failed to upload media: {e}")

        except Exception as e:
            log_error(f"Error streaming slack response: {e}")
            if is_assistant_thread:
                try:
                    await async_client.assistant_threads_setStatus(
                        channel_id=ctx["channel_id"],
                        thread_ts=ctx["ts"],
                        status="",
                    )
                except Exception:
                    pass
            if stream_started and stream_ts:
                try:
                    await async_client.chat_stopStream(channel=ctx["channel_id"], ts=stream_ts)
                except Exception:
                    pass
            _send_slack_message(
                slack_tools,
                channel=ctx["channel_id"],
                message="Sorry, there was an error processing your message.",
                thread_ts=ctx["ts"],
            )

    async def _handle_thread_started(event: dict, token: str):
        from slack_sdk.web.async_client import AsyncWebClient

        async_client = AsyncWebClient(token=token)
        thread_info = event.get("assistant_thread", {})
        channel_id = thread_info.get("channel_id", "")
        thread_ts = thread_info.get("thread_ts", "")
        if not channel_id or not thread_ts:
            return

        prompts = suggested_prompts or [
            {"title": "Help", "message": "What can you help me with?"},
            {"title": "Search", "message": "Search the web for..."},
        ]
        try:
            await async_client.assistant_threads_setSuggestedPrompts(
                channel_id=channel_id,
                thread_ts=thread_ts,
                prompts=prompts,
            )
        except Exception as e:
            log_error(f"Failed to set suggested prompts: {e}")

    def _download_event_files(slack_tools: SlackTools, event: dict) -> Tuple[List[File], List[Image]]:
        files: List[File] = []
        images: List[Image] = []

        if not event.get("files"):
            return files, images

        for file_info in event["files"]:
            file_id = file_info.get("id")
            filename = file_info.get("name", "file")
            mimetype = file_info.get("mimetype", "application/octet-stream")

            try:
                file_content = slack_tools.download_file_bytes(file_id)
                if file_content is not None:
                    if mimetype.startswith("image/"):
                        images.append(Image(content=file_content, id=file_id))
                    else:
                        safe_mime = mimetype if mimetype in File.valid_mime_types() else None
                        files.append(File(content=file_content, filename=filename, mime_type=safe_mime))
            except Exception as e:
                log_error(f"Failed to download file {file_id}: {e}")

        return files, images

    def _upload_response_media(slack_tools: SlackTools, response, channel_id: str, thread_ts: str):  # type: ignore[type-arg]
        media_attrs = [
            ("images", "image.png"),
            ("files", "file"),
            ("videos", "video.mp4"),
            ("audio", "audio.mp3"),
        ]
        for attr, default_name in media_attrs:
            items = getattr(response, attr, None)
            if not items:
                continue
            for item in items:
                content_bytes = item.get_content_bytes()
                if content_bytes:
                    try:
                        slack_tools.upload_file(
                            channel=channel_id,
                            content=content_bytes,
                            filename=getattr(item, "filename", None) or default_name,
                            thread_ts=thread_ts,
                        )
                    except Exception as e:
                        log_error(f"Failed to upload {attr.rstrip('s')}: {e}")

    def _send_slack_message(slack_tools: SlackTools, channel: str, thread_ts: str, message: str, italics: bool = False):
        if not message or not message.strip():
            return

        def _format(text: str) -> str:
            if italics:
                return "\n".join([f"_{line}_" for line in text.split("\n")])
            return text

        if len(message) <= 40000:
            slack_tools.send_message_thread(channel=channel, text=_format(message) or "", thread_ts=thread_ts)
            return

        message_batches = [message[i : i + 40000] for i in range(0, len(message), 40000)]
        for i, batch in enumerate(message_batches, 1):
            batch_message = f"[{i}/{len(message_batches)}] {batch}"
            slack_tools.send_message_thread(channel=channel, text=_format(batch_message) or "", thread_ts=thread_ts)

    return router
