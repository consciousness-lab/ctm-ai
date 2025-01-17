from typing import Dict, List, Optional, cast

import numpy as np
from numpy.typing import NDArray

from ctm_ai.chunks import Chunk, ChunkManager
from ctm_ai.ctms.ctm import ConsciousnessTuringMachine


class ChunkProcessor:
    @staticmethod
    def process_chunks(
        ctm_instance: ConsciousnessTuringMachine,
        query: Optional[str],
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Dict[str, Chunk]:
        if query is None:
            return {}
        new_chunks = ctm_instance.ask_processors(
            query=query,
            text=text,
            image=image,
            image_path=image_path,
            audio=audio,
            audio_path=audio_path,
            video_frames=video_frames,
            video_frames_path=video_frames_path,
            video_path=video_path,
        )
        return {chunk.processor_name: chunk for chunk in new_chunks}

    @staticmethod
    def fuse_chunks(
        ctm_instance: ConsciousnessTuringMachine, chunks: List[Chunk]
    ) -> List[Chunk]:
        return cast(List[Chunk], ctm_instance.fuse_processor(chunks))

    @staticmethod
    def compete_chunks(
        chunk_manager: ChunkManager, chunk1: Chunk, chunk2: Chunk
    ) -> Chunk:
        if isinstance(chunk1, Chunk) and isinstance(chunk2, Chunk):
            return cast(Chunk, chunk_manager.compete(chunk1, chunk2))
        return chunk1 if isinstance(chunk1, Chunk) else chunk2
