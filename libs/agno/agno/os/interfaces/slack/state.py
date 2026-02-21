from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class StreamState:
    first_flush_done: bool = False
    title_set: bool = False
    error_count: int = 0

    text_buffer: str = ""

    reasoning_round: int = 0

    progress_started: bool = False
    # {task_key: [title, status]}
    task_cards: Dict[str, List[str]] = field(default_factory=dict)

    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    audio: list = field(default_factory=list)
    files: list = field(default_factory=list)

    entity_type: str = "agent"
    entity_name: str = ""

    workflow_final_content: str = ""

    def track_task(self, key: str, title: str) -> None:
        self.task_cards[key] = [title, "in_progress"]
        self.progress_started = True

    def complete_task(self, key: str) -> None:
        card = self.task_cards.get(key)
        if card:
            card[1] = "complete"

    def error_task(self, key: str) -> None:
        card = self.task_cards.get(key)
        if card:
            card[1] = "error"

    def resolve_all_pending(self, status: str = "complete") -> List[dict]:
        chunks: List[dict] = []
        for key, card in self.task_cards.items():
            if card[1] == "in_progress":
                card[1] = status
                chunks.append({"type": "task_update", "id": key, "title": card[0], "status": status})
        return chunks

    def collect_media(self, chunk: Any) -> None:
        for img in getattr(chunk, "images", None) or []:
            if img not in self.images:
                self.images.append(img)
        for vid in getattr(chunk, "videos", None) or []:
            if vid not in self.videos:
                self.videos.append(vid)
        for aud in getattr(chunk, "audio", None) or []:
            if aud not in self.audio:
                self.audio.append(aud)
        for f in getattr(chunk, "files", None) or []:
            if f not in self.files:
                self.files.append(f)
