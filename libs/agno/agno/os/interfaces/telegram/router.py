import os
import re
import time
from typing import Any, AsyncIterator, List, NamedTuple, Optional, Union

from fastapi import APIRouter, BackgroundTasks, HTTPException, Request
from pydantic import BaseModel, Field

from agno.agent import Agent, RemoteAgent
from agno.media import Audio, File, Image, Video
from agno.os.interfaces.telegram.security import validate_webhook_secret_token
from agno.run.agent import RunCompletedEvent, RunContentEvent, RunOutput
from agno.team import RemoteTeam, Team
from agno.utils.log import log_error, log_info, log_warning
from agno.workflow import RemoteWorkflow, Workflow

try:
    from telebot.async_telebot import AsyncTeleBot
except ImportError as e:
    raise ImportError("`pyTelegramBotAPI` not installed. Please install using `pip install 'agno[telegram]'`") from e

TG_MAX_MESSAGE_LENGTH = 4096
TG_CHUNK_SIZE = 4000
TG_MAX_CAPTION_LENGTH = 1024
TG_GROUP_CHAT_TYPES = {"group", "supergroup"}
TG_STREAM_EDIT_INTERVAL = 1.0  # Minimum seconds between message edits to avoid rate limits

# --- Session ID scheme ---
# Session IDs use a "tg:" prefix to namespace Telegram sessions.
# Format variants:
#   tg:{chat_id}                        — DMs / private chats (one session per chat)
#   tg:{chat_id}:thread:{root_msg_id}   — Group chats (scoped by reply thread)


class TelegramStatusResponse(BaseModel):
    status: str = Field(default="available")


class TelegramWebhookResponse(BaseModel):
    status: str = Field(description="Processing status")


class ParsedMessage(NamedTuple):
    text: Optional[str]
    image_file_id: Optional[str]
    audio_file_id: Optional[str]
    video_file_id: Optional[str]
    document_meta: Optional[dict]


DEFAULT_START_MESSAGE = "Hello! I'm ready to help. Send me a message to get started."
DEFAULT_HELP_MESSAGE = "Send me text, photos, voice notes, videos, or documents and I'll help you with them."
DEFAULT_ERROR_MESSAGE = "Sorry, there was an error processing your message. Please try again later."


def attach_routes(
    router: APIRouter,
    agent: Optional[Union[Agent, RemoteAgent]] = None,
    team: Optional[Union[Team, RemoteTeam]] = None,
    workflow: Optional[Union[Workflow, RemoteWorkflow]] = None,
    reply_to_mentions_only: bool = True,
    reply_to_bot_messages: bool = True,
    start_message: str = DEFAULT_START_MESSAGE,
    help_message: str = DEFAULT_HELP_MESSAGE,
    error_message: str = DEFAULT_ERROR_MESSAGE,
    stream: bool = False,
) -> APIRouter:
    if agent is None and team is None and workflow is None:
        raise ValueError("Either agent, team, or workflow must be provided.")

    entity_type = "agent" if agent else "team" if team else "workflow"

    token = os.getenv("TELEGRAM_TOKEN")
    if not token:
        raise ValueError("TELEGRAM_TOKEN environment variable is not set")

    bot = AsyncTeleBot(token)

    _bot_username: Optional[str] = None
    _bot_id: Optional[int] = None

    async def _get_bot_info() -> tuple:
        nonlocal _bot_username, _bot_id
        if _bot_username is None or _bot_id is None:
            me = await bot.get_me()
            _bot_username = me.username
            _bot_id = me.id
        return _bot_username, _bot_id

    async def _get_bot_username() -> str:
        username, _ = await _get_bot_info()
        return username

    async def _get_bot_id() -> int:
        _, bot_id = await _get_bot_info()
        return bot_id

    def _message_mentions_bot(message: dict, bot_username: str) -> bool:
        text = message.get("text", "") or message.get("caption", "")
        entities = message.get("entities", []) or message.get("caption_entities", [])
        # NOTE: Telegram entity offsets are UTF-16 code units; Python slices by
        # code points. Mentions after non-BMP characters (e.g. emoji) may be
        # misparsed. Acceptable trade-off for now.
        for entity in entities:
            if entity.get("type") == "mention":
                offset = entity["offset"]
                length = entity["length"]
                mention = text[offset : offset + length].lstrip("@").lower()
                if mention == bot_username.lower():
                    return True
        return False

    def _is_reply_to_bot(message: dict, bot_id: int) -> bool:
        reply_msg = message.get("reply_to_message")
        if not reply_msg:
            return False
        return reply_msg.get("from", {}).get("id") == bot_id

    def _strip_bot_mention(text: str, bot_username: str) -> str:
        return re.sub(rf"@{re.escape(bot_username)}\b", "", text, flags=re.IGNORECASE).strip()

    def _parse_inbound_message(message: dict) -> ParsedMessage:
        message_text: Optional[str] = None
        image_file_id: Optional[str] = None
        audio_file_id: Optional[str] = None
        video_file_id: Optional[str] = None
        document_meta: Optional[dict] = None

        if message.get("text"):
            message_text = message["text"]
        elif message.get("photo"):
            image_file_id = message["photo"][-1]["file_id"]
            message_text = message.get("caption", "Describe the image")
        elif message.get("sticker"):
            image_file_id = message["sticker"]["file_id"]
            message_text = "Describe this sticker"
        elif message.get("voice"):
            audio_file_id = message["voice"]["file_id"]
            message_text = message.get("caption", "Transcribe or describe this audio")
        elif message.get("audio"):
            audio_file_id = message["audio"]["file_id"]
            message_text = message.get("caption", "Describe this audio")
        elif message.get("video") or message.get("video_note") or message.get("animation"):
            vid: dict = message.get("video") or message.get("video_note") or message.get("animation")  # type: ignore[assignment]
            video_file_id = vid["file_id"]
            message_text = message.get("caption", "Describe this video")
        elif message.get("document"):
            document_meta = message["document"]
            message_text = message.get("caption", "Process this file")

        return ParsedMessage(message_text, image_file_id, audio_file_id, video_file_id, document_meta)

    async def _download_inbound_media(
        image_file_id: Optional[str],
        audio_file_id: Optional[str],
        video_file_id: Optional[str],
        document_meta: Optional[dict],
    ) -> tuple[Optional[List[Image]], Optional[List[Audio]], Optional[List[Video]], Optional[List[File]]]:
        images: Optional[List[Image]] = None
        audio: Optional[List[Audio]] = None
        videos: Optional[List[Video]] = None
        files: Optional[List[File]] = None

        if image_file_id:
            image_bytes = await _get_file_bytes(image_file_id)
            if image_bytes:
                images = [Image(content=image_bytes)]
        if audio_file_id:
            audio_bytes = await _get_file_bytes(audio_file_id)
            if audio_bytes:
                audio = [Audio(content=audio_bytes)]
        if video_file_id:
            video_bytes = await _get_file_bytes(video_file_id)
            if video_bytes:
                videos = [Video(content=video_bytes)]
        if document_meta:
            doc_bytes = await _get_file_bytes(document_meta["file_id"])
            if doc_bytes:
                doc_mime = document_meta.get("mime_type")
                files = [
                    File(
                        content=doc_bytes,
                        mime_type=doc_mime if doc_mime in File.valid_mime_types() else None,
                        filename=document_meta.get("file_name"),
                    )
                ]

        return images, audio, videos, files

    async def _get_file_bytes(file_id: str) -> Optional[bytes]:
        try:
            file_info = await bot.get_file(file_id)
            return await bot.download_file(file_info.file_path)
        except Exception as e:
            log_error(f"Error downloading file: {e}")
            return None

    async def _send_text_chunked(chat_id: int, text: str, reply_to_message_id: Optional[int] = None) -> None:
        if len(text) <= TG_MAX_MESSAGE_LENGTH:
            await bot.send_message(chat_id, text, reply_to_message_id=reply_to_message_id)
            return
        chunks: List[str] = [text[i : i + TG_CHUNK_SIZE] for i in range(0, len(text), TG_CHUNK_SIZE)]
        for i, chunk in enumerate(chunks, 1):
            reply_id = reply_to_message_id if i == 1 else None
            await bot.send_message(chat_id, f"[{i}/{len(chunks)}] {chunk}", reply_to_message_id=reply_id)

    def _resolve_media_data(item: Any) -> Optional[Any]:
        url = getattr(item, "url", None)
        if url:
            return url
        get_bytes = getattr(item, "get_content_bytes", None)
        return get_bytes() if callable(get_bytes) else None

    async def _send_response_media(response: RunOutput, chat_id: int, reply_to: Optional[int]) -> bool:
        """Send all media items from the response. Caption goes on the first item only."""
        any_media_sent = False
        caption = response.content[:TG_MAX_CAPTION_LENGTH] if response.content else None

        # Data-driven dispatch: maps response attributes to Telegram sender methods
        media_senders = [
            ("images", bot.send_photo),
            ("audio", bot.send_audio),
            ("videos", bot.send_video),
            ("files", bot.send_document),
        ]
        for attr, sender in media_senders:
            items = getattr(response, attr, None)
            if not items:
                continue
            for item in items:
                try:
                    data = _resolve_media_data(item)
                    if data:
                        await sender(chat_id, data, caption=caption, reply_to_message_id=reply_to)
                        any_media_sent = True
                        # Clear caption and reply_to after first successful send
                        caption = None
                        reply_to = None
                except Exception as e:
                    log_error(f"Failed to send {attr.rstrip('s')} to chat {chat_id}: {e}")

        return any_media_sent

    async def _stream_to_telegram(
        event_stream: AsyncIterator[Any],
        chat_id: int,
        reply_to: Optional[int],
    ) -> Optional[RunOutput]:
        """Consume a streaming agent response and progressively edit a Telegram message.

        Sends an initial placeholder message, then edits it as content arrives.
        Edits are throttled to TG_STREAM_EDIT_INTERVAL seconds to respect Telegram rate limits.
        Returns the final RunOutput (yielded at end of stream) for media handling.
        """
        sent_message_id: Optional[int] = None
        accumulated_content = ""
        last_edit_time = 0.0
        final_run_output: Optional[RunOutput] = None

        async for event in event_stream:
            if isinstance(event, RunOutput):
                final_run_output = event
                continue

            if isinstance(event, RunContentEvent) and event.content:
                accumulated_content += str(event.content)

                now = time.monotonic()
                if now - last_edit_time < TG_STREAM_EDIT_INTERVAL:
                    continue

                # Truncate to Telegram's max message length for in-progress edits
                display_text = accumulated_content[:TG_MAX_MESSAGE_LENGTH]

                try:
                    if sent_message_id is None:
                        msg = await bot.send_message(chat_id, display_text, reply_to_message_id=reply_to)
                        sent_message_id = msg.message_id
                    else:
                        await bot.edit_message_text(display_text, chat_id, sent_message_id)
                    last_edit_time = now
                except Exception as e:
                    log_warning(f"Stream edit failed (will retry on next chunk): {e}")

            elif isinstance(event, RunCompletedEvent):
                # RunCompletedEvent carries the final content and media
                if event.content:
                    accumulated_content = str(event.content)

        # Final edit with complete content
        if accumulated_content and sent_message_id:
            try:
                if len(accumulated_content) <= TG_MAX_MESSAGE_LENGTH:
                    await bot.edit_message_text(accumulated_content, chat_id, sent_message_id)
                else:
                    # Content exceeds max length: delete the streamed message and send chunked
                    try:
                        await bot.delete_message(chat_id, sent_message_id)
                    except Exception:
                        pass
                    await _send_text_chunked(chat_id, accumulated_content, reply_to_message_id=reply_to)
                    sent_message_id = None  # Already handled
            except Exception as e:
                log_warning(f"Final stream edit failed: {e}")
        elif accumulated_content and not sent_message_id:
            # Never sent an initial message (very fast response), send normally
            await _send_text_chunked(chat_id, accumulated_content, reply_to_message_id=reply_to)

        return final_run_output

    @router.get(
        "/status",
        operation_id=f"telegram_status_{entity_type}",
        name="telegram_status",
        description="Check Telegram interface status",
        response_model=TelegramStatusResponse,
    )
    async def status():
        return TelegramStatusResponse()

    @router.post(
        "/webhook",
        operation_id=f"telegram_webhook_{entity_type}",
        name="telegram_webhook",
        description="Process incoming Telegram webhook events",
        response_model=TelegramWebhookResponse,
        responses={
            200: {"description": "Event processed successfully"},
            403: {"description": "Invalid webhook secret token"},
        },
    )
    async def webhook(request: Request, background_tasks: BackgroundTasks):
        try:
            secret_token = request.headers.get("X-Telegram-Bot-Api-Secret-Token")
            if not validate_webhook_secret_token(secret_token):
                log_warning("Invalid webhook secret token")
                raise HTTPException(status_code=403, detail="Invalid secret token")

            body = await request.json()

            # Only process new messages. edited_message, channel_post, and
            # callback_query are intentionally ignored for now.
            # TODO: Track processed update_ids to prevent duplicate processing
            # on webhook retries. Duplicates are rare and handled gracefully.
            message = body.get("message")
            if not message:
                return TelegramWebhookResponse(status="ignored")

            background_tasks.add_task(_process_message, message, agent, team, workflow)
            return TelegramWebhookResponse(status="processing")

        except HTTPException:
            raise
        except Exception as e:
            log_error(f"Error processing webhook: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def _process_message(
        message: dict,
        agent: Optional[Union[Agent, RemoteAgent]],
        team: Optional[Union[Team, RemoteTeam]],
        workflow: Optional[Union[Workflow, RemoteWorkflow]] = None,
    ):
        chat_id = message.get("chat", {}).get("id")
        if not chat_id:
            log_warning("Received message without chat_id")
            return

        try:
            if message.get("from", {}).get("is_bot"):
                return

            chat_type = message.get("chat", {}).get("type", "private")
            is_group = chat_type in TG_GROUP_CHAT_TYPES
            incoming_message_id = message.get("message_id")

            text = message.get("text", "")
            cmd = text.split()[0].split("@")[0] if text else ""
            if cmd == "/start":
                await bot.send_message(chat_id, start_message)
                return
            if cmd == "/help":
                await bot.send_message(chat_id, help_message)
                return

            if is_group:
                bot_username = await _get_bot_username()
                if reply_to_mentions_only:
                    is_mentioned = _message_mentions_bot(message, bot_username)
                    is_reply = reply_to_bot_messages and _is_reply_to_bot(message, await _get_bot_id())
                    if not is_mentioned and not is_reply:
                        return

            await bot.send_chat_action(chat_id, "typing")

            parsed = _parse_inbound_message(message)
            if parsed.text is None:
                return
            message_text = parsed.text

            if is_group and message_text:
                message_text = _strip_bot_mention(message_text, bot_username)

            user_id = str(message.get("from", {}).get("id", chat_id))
            # DMs: one session per chat. Groups: thread by the replied-to message ID.
            # Note: Telegram has no stable thread_ts like Slack, so session may drift
            # when users reply to different bot messages in the same conversation.
            if is_group:
                reply_msg = message.get("reply_to_message")
                root_msg_id = reply_msg.get("message_id", incoming_message_id) if reply_msg else incoming_message_id
                session_id = f"tg:{chat_id}:thread:{root_msg_id}"
            else:
                session_id = f"tg:{chat_id}"

            log_info(f"Processing message from {user_id}: {message_text}")

            reply_to = incoming_message_id if is_group else None

            images, audio, videos, files = await _download_inbound_media(
                parsed.image_file_id, parsed.audio_file_id, parsed.video_file_id, parsed.document_meta
            )

            run_kwargs: dict = dict(
                user_id=user_id,
                session_id=session_id,
                images=images,
                audio=audio,
                videos=videos,
                files=files,
            )

            # Streaming mode: progressively edit a single Telegram message as content arrives.
            # Only supported for Agent (not Team/Workflow) since they have different event types.
            use_stream = stream and agent and not isinstance(agent, RemoteAgent)

            if use_stream:
                event_stream = agent.arun(message_text, stream=True, yield_run_output=True, **run_kwargs)  # type: ignore[union-attr]
                response = await _stream_to_telegram(event_stream, chat_id, reply_to)

                # Handle media from the final RunOutput if present
                if response:
                    if response.status == "ERROR":
                        log_error(response.content)
                        return
                    await _send_response_media(response, chat_id, reply_to=None)
            else:
                response = None
                if agent:
                    response = await agent.arun(message_text, **run_kwargs)
                elif team:
                    response = await team.arun(message_text, **run_kwargs)  # type: ignore
                elif workflow:
                    response = await workflow.arun(message_text, **run_kwargs)  # type: ignore

                if not response:
                    return

                if response.status == "ERROR":
                    await _send_text_chunked(
                        chat_id,
                        error_message,
                        reply_to_message_id=reply_to,
                    )
                    log_error(response.content)
                    return

                reasoning = getattr(response, "reasoning_content", None)
                if reasoning:
                    await _send_text_chunked(chat_id, f"Reasoning: \n{reasoning}", reply_to_message_id=reply_to)

                any_media_sent = await _send_response_media(response, chat_id, reply_to)

                # Media captions are capped at 1024 chars. If text overflows the caption,
                # send the full text as a follow-up message so nothing is lost.
                if response.content:
                    if any_media_sent and len(response.content) > TG_MAX_CAPTION_LENGTH:
                        await _send_text_chunked(chat_id, response.content)
                    elif not any_media_sent:
                        await _send_text_chunked(chat_id, response.content, reply_to_message_id=reply_to)

        except Exception as e:
            log_error(f"Error processing message: {e}")
            try:
                await _send_text_chunked(chat_id, error_message)
            except Exception as send_error:
                log_error(f"Error sending error message: {send_error}")

    return router
