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
_TOOL_STARTED = {RunEvent.tool_call_started.value, TeamRunEvent.tool_call_started.value}
_TOOL_COMPLETED = {RunEvent.tool_call_completed.value, TeamRunEvent.tool_call_completed.value}
_CONTENT_EVENTS = {RunEvent.run_content.value, TeamRunEvent.run_content.value}
_STEP_OUTPUT = {WorkflowRunEvent.step_output.value}


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
                _send_slack_message(
                    slack_tools,
                    channel=ctx["channel_id"],
                    message=f"Reasoning: \n{response.reasoning_content}",
                    thread_ts=ctx["ts"],
                    italics=True,
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
        event = _fetch_mention_files(event, ctx["channel_id"], ctx["ts"])
        files, images = _download_event_files(slack_tools, event)

        team_id = data.get("team_id") or event.get("team") or None
        authorizations = data.get("authorizations", [])
        bot_user_id = authorizations[0]["user_id"] if authorizations else None

        async_client = AsyncWebClient(token=slack_tools.token)
        stream_ts: Optional[str] = None
        try:
            try:
                await async_client.assistant_threads_setStatus(
                    channel_id=ctx["channel_id"],
                    thread_ts=ctx["ts"],
                    status="Thinking...",
                )
            except Exception:
                pass

            start_resp = await async_client.chat_startStream(
                channel=ctx["channel_id"],
                thread_ts=ctx["ts"],
                markdown_text="",
                recipient_team_id=team_id,
                recipient_user_id=bot_user_id,
            )
            stream_ts = start_resp["ts"]

            status_cleared = False
            title_set = False
            text_buffer = ""
            BUFFER_SIZE = 256
            response_stream = None

            if agent:
                response_stream = agent.arun(
                    ctx["message_text"],
                    stream=True,
                    user_id=ctx["user"],
                    session_id=ctx["ts"],
                    files=files if files else None,
                    images=images if images else None,
                )
            elif team:
                response_stream = team.arun(  # type: ignore[assignment]
                    ctx["message_text"],
                    stream=True,
                    user_id=ctx["user"],
                    session_id=ctx["ts"],
                    files=files if files else None,
                    images=images if images else None,
                )
            elif workflow:
                response_stream = workflow.arun(  # type: ignore[assignment]
                    ctx["message_text"],
                    stream=True,
                    user_id=ctx["user"],
                    session_id=ctx["ts"],
                    files=files if files else None,
                    images=images if images else None,
                )

            if response_stream is None:
                await async_client.chat_stopStream(channel=ctx["channel_id"], ts=stream_ts, markdown_text="")
                return

            tool_tasks: dict[str, str] = {}

            async for chunk in response_stream:
                if chunk.event in _TOOL_STARTED:  # type: ignore[union-attr]
                    tool_name = chunk.tool.tool_name if chunk.tool else "a tool"  # type: ignore[union-attr]
                    task_id = chunk.tool.tool_call_id if chunk.tool else str(len(tool_tasks))  # type: ignore[union-attr]
                    tool_args = chunk.tool.tool_args if chunk.tool else {}  # type: ignore[union-attr]
                    tool_tasks[task_id] = tool_name  # type: ignore[index, assignment]

                    details = ", ".join(f"{k}={v}" for k, v in tool_args.items()) if tool_args else None
                    task_chunk: dict = {
                        "type": "task_update",
                        "id": task_id,
                        "title": tool_name,
                        "status": "in_progress",
                    }
                    if details:
                        task_chunk["details"] = details

                    await async_client.chat_appendStream(
                        channel=ctx["channel_id"],
                        ts=stream_ts,
                        markdown_text="",
                        chunks=[
                            {"type": "plan_update", "title": "Working on it..."},
                            task_chunk,
                        ],
                    )

                elif chunk.event in _TOOL_COMPLETED:  # type: ignore[union-attr]
                    task_id = chunk.tool.tool_call_id if chunk.tool else None  # type: ignore[union-attr]
                    if task_id and task_id in tool_tasks:
                        errored = chunk.tool.tool_call_error if chunk.tool else False  # type: ignore[union-attr]
                        task_chunk = {
                            "type": "task_update",
                            "id": task_id,
                            "title": tool_tasks[task_id],
                            "status": "error" if errored else "complete",
                        }
                        await async_client.chat_appendStream(
                            channel=ctx["channel_id"],
                            ts=stream_ts,
                            markdown_text="",
                            chunks=[task_chunk],
                        )

                content_text: Optional[str] = None
                if chunk.event in _CONTENT_EVENTS and chunk.content:  # type: ignore[union-attr]
                    content_text = str(chunk.content)
                elif chunk.event in _STEP_OUTPUT and chunk.content:  # type: ignore[union-attr]
                    content_text = str(chunk.content)

                if content_text:
                    if not status_cleared:
                        try:
                            await async_client.assistant_threads_setStatus(
                                channel_id=ctx["channel_id"],
                                thread_ts=ctx["ts"],
                                status="",
                            )
                        except Exception:
                            pass
                        status_cleared = True
                    if not title_set:
                        title = ctx["message_text"][:50].strip() or "New conversation"
                        try:
                            await async_client.assistant_threads_setTitle(
                                channel_id=ctx["channel_id"],
                                thread_ts=ctx["ts"],
                                title=title,
                            )
                        except Exception:
                            pass
                        title_set = True

                    text_buffer += content_text
                    if len(text_buffer) >= BUFFER_SIZE:
                        await async_client.chat_appendStream(
                            channel=ctx["channel_id"],
                            ts=stream_ts,
                            markdown_text=text_buffer,
                        )
                        text_buffer = ""

            if not status_cleared:
                try:
                    await async_client.assistant_threads_setStatus(
                        channel_id=ctx["channel_id"],
                        thread_ts=ctx["ts"],
                        status="",
                    )
                except Exception:
                    pass
            await async_client.chat_stopStream(
                channel=ctx["channel_id"],
                ts=stream_ts,
                markdown_text=text_buffer,
            )
        except Exception as e:
            log_error(f"Error streaming slack response: {e}")
            try:
                await async_client.assistant_threads_setStatus(
                    channel_id=ctx["channel_id"],
                    thread_ts=ctx["ts"],
                    status="",
                )
            except Exception:
                pass
            if stream_ts:
                try:
                    await async_client.chat_stopStream(channel=ctx["channel_id"], ts=stream_ts, markdown_text="")
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
        try:
            await async_client.assistant_threads_setSuggestedPrompts(
                channel_id=channel_id,
                thread_ts=thread_ts,
                prompts=[
                    {"title": "Help", "message": "What can you help me with?"},
                    {"title": "Search", "message": "Search the web for..."},
                ],
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
