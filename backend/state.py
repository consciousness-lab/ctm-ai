from typing import Any, Dict, List, Optional

from ctm_ai.chunks import Chunk


class AppState:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.query: Optional[str] = None
        self.text: Optional[str] = None
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
        # 存储输入参数，用于后续的fuse_processor和link_form调用
        self.input_params: Dict[str, Any] = {}

    def get_input_params(self) -> Dict[str, Any]:
        """获取当前的输入参数字典"""
        return self.input_params.copy()

    def update_input_params(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        audio_path: Optional[str] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> None:
        """更新输入参数"""
        self.input_params = {
            'text': text,
            'image_path': image_path,
            'audio_path': audio_path,
            'video_frames_path': video_frames_path,
            'video_path': video_path,
        }
