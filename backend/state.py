from typing import Any, Dict, List, Optional

from ctm_ai.chunks import Chunk


class AppState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.query: Optional[str] = None
        self.winning_chunk: Optional[Chunk] = None
        self.chunks: List[Chunk] = []
        self.node_details: Dict[str, Any] = {}
        self.node_parents: Dict[str, List[str]] = {}
        self.node_gists: Dict[str, Any] = {}
        self.saved_files: Dict[str, List[str]] = {
            'images': [],
            'audios': [],
            'videos': [],
            'video_frames': [],
        }
