from ssl import SSLContext
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from agno.agent import Agent, RemoteAgent
from agno.os.interfaces.slack.handlers import (
    _BOT_SUBTYPES,
    DISPATCH,
    WORKFLOW_DISPATCH,
)
from agno.os.interfaces.slack.helpers import (
    download_event_files,
    extract_event_context,
    fetch_mention_files,
    send_slack_message,
    should_respond,
    upload_response_media,
)
from agno.os.interfaces.slack.security import verify_slack_signature
from agno.os.interfaces.slack.state import StreamState
from agno.team import RemoteTeam, Team
from agno.tools.slack import SlackTools
from agno.utils.log import log_error
from agno.workflow import RemoteWorkflow, Workflow


class SlackEventResponse(BaseModel):
    status: str = Field(default="ok")


class SlackChallengeResponse(BaseModel):
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
    ssl: Optional[SSLContext] = None,
) -> APIRouter:
    entity = agent or team or workflow
    entity_type = "agent" if agent else "team" if team else "workflow" if workflow else "unknown"
    raw_name = getattr(entity, "name", None)
    entity_name = raw_name if isinstance(raw_name, str) else entity_type
    op_suffix = entity_name.lower().replace(" ", "_")

    slack_tools = SlackTools(token=token, ssl=ssl)

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

        # Slack retries events after ~3s if the previous delivery timed out.
        # Since we ACK immediately and process in the background, retries are
        # always duplicates. Trade-off: if the server crashes mid-processing,
        # the retry that carries the same event won't be reprocessed.
        if request.headers.get("X-Slack-Retry-Num"):
            return SlackEventResponse(status="ok")

        data = await request.json()

        if data.get("type") == "url_verification":
            return SlackChallengeResponse(challenge=data.get("challenge"))

        if "event" in data:
            event = data["event"]
            event_type = event.get("type")

            if event_type == "assistant_thread_started" and streaming:
                background_tasks.add_task(_handle_thread_started, event)
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

    async def _process_slack_event(event: dict):
        if not should_respond(event, reply_to_mentions_only):
            return

        ctx = extract_event_context(event)
        event = fetch_mention_files(slack_tools, event, ctx["channel_id"], ctx["ts"])
        files, images, videos, audio = download_event_files(slack_tools, event)

        run_kwargs: Dict[str, Any] = {
            "user_id": ctx["user"],
            "session_id": ctx["ts"],
            "files": files if files else None,
            "images": images if images else None,
            "videos": videos if videos else None,
            "audio": audio if audio else None,
        }

        try:
            response = None
            if agent:
                response = await agent.arun(ctx["message_text"], **run_kwargs)  # type: ignore[misc]
            elif team:
                response = await team.arun(ctx["message_text"], **run_kwargs)  # type: ignore
            elif workflow:
                response = await workflow.arun(ctx["message_text"], **run_kwargs)  # type: ignore

            if response:
                if response.status == "ERROR":
                    log_error(f"Error processing message: {response.content}")
                    send_slack_message(
                        slack_tools,
                        channel=ctx["channel_id"],
                        message="Sorry, there was an error processing your message. Please try again later.",
                        thread_ts=ctx["ts"],
                    )
                    return

                if hasattr(response, "reasoning_content") and response.reasoning_content:
                    rc = str(response.reasoning_content)
                    formatted = f"*Reasoning:*\n> {rc.replace(chr(10), chr(10) + '> ')}"
                    send_slack_message(slack_tools, channel=ctx["channel_id"], message=formatted, thread_ts=ctx["ts"])

                content = str(response.content) if response.content else ""
                send_slack_message(slack_tools, channel=ctx["channel_id"], message=content, thread_ts=ctx["ts"])
                upload_response_media(slack_tools, response, ctx["channel_id"], ctx["ts"])
        except Exception as e:
            log_error(f"Error processing slack event: {e}")
            send_slack_message(
                slack_tools,
                channel=ctx["channel_id"],
                message="Sorry, there was an error processing your message.",
                thread_ts=ctx["ts"],
            )

    async def _stream_slack_response(data: dict):
        from slack_sdk.web.async_client import AsyncWebClient

        event = data["event"]
        if not should_respond(event, reply_to_mentions_only):
            return

        ctx = extract_event_context(event)

        # Slack streaming API (chat_stream) requires a thread_ts.
        # For non-threaded messages, fall back to the non-streaming path.
        if not event.get("thread_ts"):
            await _process_slack_event(event)
            return

        team_id = data.get("team_id") or event.get("team") or None
        user_id = ctx.get("user") or event.get("user")

        async_client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
        state = StreamState(entity_type=entity_type, entity_name=entity_name)
        stream = None

        try:
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

            event = fetch_mention_files(slack_tools, event, ctx["channel_id"], ctx["ts"])
            files, images, videos, audio = download_event_files(slack_tools, event)

            response_stream = None
            run_kwargs: Dict[str, Any] = {
                "stream": True,
                "stream_events": True,
                "user_id": ctx["user"],
                "session_id": ctx["ts"],
                "files": files if files else None,
                "images": images if images else None,
                "videos": videos if videos else None,
                "audio": audio if audio else None,
            }

            if agent:
                response_stream = agent.arun(ctx["message_text"], **run_kwargs)
            elif team:
                response_stream = team.arun(ctx["message_text"], **run_kwargs)  # type: ignore[assignment]
            elif workflow:
                response_stream = workflow.arun(ctx["message_text"], **run_kwargs)  # type: ignore[assignment]

            if response_stream is None:
                try:
                    await async_client.assistant_threads_setStatus(
                        channel_id=ctx["channel_id"], thread_ts=ctx["ts"], status=""
                    )
                except Exception:
                    pass
                return

            # buffer_size=0: we handle text buffering ourselves via state.text_buffer
            # thresholds (initial_buffer_size for first flush, buffer_size after).
            stream = await async_client.chat_stream(
                channel=ctx["channel_id"],
                thread_ts=ctx["ts"],
                recipient_team_id=team_id,
                recipient_user_id=user_id,
                task_display_mode=task_display_mode,
                buffer_size=0,
            )

            dispatch = WORKFLOW_DISPATCH if state.entity_type == "workflow" else DISPATCH
            stream_initialized = False
            async for chunk in response_stream:
                event_name = getattr(chunk, "event", None)
                handler = dispatch.get(event_name) if event_name else None
                if handler:
                    # Lazy-start: send plan_update via startStream so all
                    # subsequent task_update chunks go via appendStream
                    # (Slack silently discards task_update in startStream).
                    # Delaying until the first handler fires preserves the
                    # setStatus typing indicator while the model thinks.
                    if not stream_initialized:
                        await stream.append(chunks=[{"type": "plan_update", "title": "Working..."}])
                        stream_initialized = True
                    action = await handler(chunk, state, stream)
                    if action == "break":
                        break

                # Flush text buffer when threshold reached
                if state.text_buffer:
                    if not stream_initialized:
                        await stream.append(chunks=[{"type": "plan_update", "title": "Working..."}])
                        stream_initialized = True
                    if not state.title_set:
                        state.title_set = True
                        title = ctx["message_text"][:50].strip() or "New conversation"
                        try:
                            await async_client.assistant_threads_setTitle(
                                channel_id=ctx["channel_id"], thread_ts=ctx["ts"], title=title
                            )
                        except Exception:
                            pass

                    threshold = buffer_size if state.first_flush_done else initial_buffer_size
                    if len(state.text_buffer) >= threshold:
                        await stream.append(markdown_text=state.text_buffer)
                        state.text_buffer = ""
                        state.first_flush_done = True

            completion_chunks = state.resolve_all_pending() if state.progress_started else []
            stop_kwargs: Dict[str, Any] = {}
            if state.text_buffer:
                stop_kwargs["markdown_text"] = state.text_buffer
            if completion_chunks:
                stop_kwargs["chunks"] = completion_chunks
            await stream.stop(**stop_kwargs)

            # Upload collected media after stream ends
            upload_response_media(slack_tools, state, ctx["channel_id"], ctx["ts"])

        except Exception as e:
            log_error(f"Error streaming slack response: {e}")
            try:
                await async_client.assistant_threads_setStatus(
                    channel_id=ctx["channel_id"], thread_ts=ctx["ts"], status=""
                )
            except Exception:
                pass
            if stream is not None:
                try:
                    await stream.stop()
                except Exception:
                    pass
            send_slack_message(
                slack_tools,
                channel=ctx["channel_id"],
                message="Sorry, there was an error processing your message.",
                thread_ts=ctx["ts"],
            )

    async def _handle_thread_started(event: dict):
        from slack_sdk.web.async_client import AsyncWebClient

        async_client = AsyncWebClient(token=slack_tools.token, ssl=ssl)
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
                channel_id=channel_id, thread_ts=thread_ts, prompts=prompts
            )
        except Exception as e:
            log_error(f"Failed to set suggested prompts: {e}")

    return router
