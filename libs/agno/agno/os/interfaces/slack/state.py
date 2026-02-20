from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TaskCard:
    key: str
    title: str
    status: str = "in_progress"


@dataclass
class StreamState:
    first_flush_done: bool = False
    title_set: bool = False
    error_count: int = 0

    text_buffer: str = ""

    reasoning_round: int = 0

    # Active task cards tracked for stream rotation and completion
    progress_started: bool = False
    cards_frozen: bool = False
    task_cards: Dict[str, TaskCard] = field(default_factory=dict)

    # Lookup maps
    tool_line_map: Dict[str, str] = field(default_factory=dict)

    # Media collected from responses
    images: list = field(default_factory=list)
    videos: list = field(default_factory=list)
    audio: list = field(default_factory=list)
    files: list = field(default_factory=list)

    entity_type: str = "agent"
    entity_name: str = ""

    # Workflow mode: buffer only the last step's output separately
    workflow_final_content: str = ""

    def track_task(self, key: str, title: str) -> None:
        if self.cards_frozen:
            return
        self.task_cards[key] = TaskCard(key=key, title=title)
        self.progress_started = True

    def complete_task(self, key: str) -> None:
        card = self.task_cards.get(key)
        if card:
            card.status = "complete"

    def error_task(self, key: str) -> None:
        card = self.task_cards.get(key)
        if card:
            card.status = "error"

    def complete_all_pending(self) -> List[dict]:
        chunks: List[dict] = []
        for card in self.task_cards.values():
            if card.status == "in_progress":
                card.status = "complete"
                chunks.append({"type": "task_update", "id": card.key, "title": card.title, "status": "complete"})
        return chunks

    def error_all_pending(self) -> List[dict]:
        chunks: List[dict] = []
        for card in self.task_cards.values():
            if card.status == "in_progress":
                card.status = "error"
                chunks.append({"type": "task_update", "id": card.key, "title": card.title, "status": "error"})
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
