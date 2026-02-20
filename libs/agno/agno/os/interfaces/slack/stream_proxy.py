import json
from typing import Any, Dict, List

from agno.os.interfaces.slack.state import StreamState
from agno.utils.log import log_error, logger


def _is_msg_too_long(exc: Exception) -> bool:
    return "msg_too_long" in str(exc)


def _split_text(text: str, max_chars: int = 10000) -> List[str]:
    """Split text on paragraph boundaries to stay under per-call limit."""
    if len(text) <= max_chars:
        return [text]

    parts: List[str] = []
    while text:
        if len(text) <= max_chars:
            parts.append(text)
            break
        cut = text.rfind("\n\n", 0, max_chars)
        if cut <= 0:
            cut = text.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = text.rfind(" ", 0, max_chars)
        if cut <= 0:
            cut = max_chars
        parts.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return parts


class SplitStreamProxy:
    """Wraps Slack chat_stream with automatic stream splitting on size limit.

    Slack's chat_stream has an undocumented total message size limit (~40KB).
    Once exceeded, ALL operations fail with ``msg_too_long``. This proxy
    tracks accumulated bytes and proactively rotates to a new stream before
    hitting that wall.

    On rotation, all in-progress task cards are marked complete via
    state.complete_all_pending() so the first message's plan block ends
    cleanly. The new stream continues with content only — no second plan block.
    """

    def __init__(
        self,
        async_client: Any,
        state: StreamState,
        stream_kwargs: Dict[str, Any],
        *,
        soft_limit_bytes: int = 28000,
        headroom_bytes: int = 8000,
    ):
        self.client = async_client
        self.state = state
        self.stream_kwargs = stream_kwargs
        self.soft_limit = soft_limit_bytes
        self.headroom = headroom_bytes
        self._stream: Any = None
        self._segment_bytes: int = 0
        self._segment_count: int = 0

    async def start(self) -> None:
        self._stream = await self.client.chat_stream(**self.stream_kwargs)
        self._segment_bytes = 0
        self._segment_count += 1
        logger.debug(f"[stream_proxy] started segment {self._segment_count}")

    async def append(self, **kwargs: Any) -> None:
        kwargs = self._filter_frozen_chunks(kwargs)
        if not kwargs:
            return
        md = kwargs.get("markdown_text")
        if md and len(md) > 10000:
            chunks_arg = kwargs.get("chunks")
            for part in _split_text(md, 10000):
                await self._append_one(markdown_text=part, chunks=chunks_arg)
                chunks_arg = None
            return
        await self._append_one(**kwargs)

    async def _append_one(self, **kwargs: Any) -> None:
        est = self._estimate_bytes(kwargs)

        if self._segment_bytes + est > (self.soft_limit - self.headroom):
            logger.debug(
                f"[stream_proxy] approaching limit ({self._segment_bytes}+{est} > "
                f"{self.soft_limit - self.headroom}), rotating"
            )
            await self._rotate()

        filtered = {k: v for k, v in kwargs.items() if v is not None}
        try:
            await self._stream.append(**filtered)
            self._segment_bytes += est
        except Exception as exc:
            if _is_msg_too_long(exc):
                logger.debug("[stream_proxy] msg_too_long on append, rotating and retrying")
                await self._rotate()
                await self._stream.append(**filtered)
                self._segment_bytes += est
            else:
                raise

    async def stop(self, **kwargs: Any) -> None:
        kwargs = self._filter_frozen_chunks(kwargs)
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        try:
            await self._stream.stop(**filtered)
        except Exception as exc:
            if _is_msg_too_long(exc):
                log_error("[stream_proxy] msg_too_long on stop, rotating for final stop")
                await self._rotate()
                await self._stream.stop(**filtered)
            else:
                raise

    async def _rotate(self) -> None:
        # Close all in-progress task cards so the plan block ends cleanly.
        # Freeze cards so no new plan block appears in subsequent streams.
        close_chunks = self.state.complete_all_pending() if self.state.progress_started else []
        self.state.cards_frozen = True
        try:
            if close_chunks:
                await self._stream.stop(chunks=close_chunks)
            else:
                await self._stream.stop()
        except Exception as stop_exc:
            log_error(f"[stream_proxy] stop failed during rotation: {stop_exc}")
            try:
                await self._stream.stop()
            except Exception:
                pass

        # Fresh stream in the same thread — content only, no task cards
        await self.start()

    @staticmethod
    def _strip_task_chunks(chunks: list) -> list:
        return [c for c in chunks if not (isinstance(c, dict) and c.get("type") == "task_update")]

    def _filter_frozen_chunks(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:
        if not self.state.cards_frozen:
            return kwargs
        chunks = kwargs.get("chunks")
        if chunks is None:
            return kwargs
        filtered = self._strip_task_chunks(chunks)
        if filtered:
            return {**kwargs, "chunks": filtered}
        out = {k: v for k, v in kwargs.items() if k != "chunks"}
        return out if out else {}

    @staticmethod
    def _estimate_bytes(kwargs: Dict[str, Any]) -> int:
        total = 0
        md = kwargs.get("markdown_text")
        if md:
            total += len(md.encode("utf-8"))
        chunks = kwargs.get("chunks")
        if chunks:
            total += len(json.dumps(chunks, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))
        return total + 200
