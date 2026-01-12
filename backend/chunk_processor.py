from typing import Any, Dict, List, Optional, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from ctm_ai.chunks import Chunk, ChunkManager
from ctm_ai.ctms.ctm import ConsciousTuringMachine


class ChunkProcessor:
    @staticmethod
    def process_chunks(
        ctm_instance: ConsciousTuringMachine,
        query: Optional[str],
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Tuple[Dict[str, Chunk], List[Chunk]]:
        """
        调用 ask_processors 获取所有处理器的输出。

        返回:
            Tuple[Dict[str, Chunk], List[Chunk]]: (processor_name -> chunk 映射, chunks 列表)
        """
        if query is None:
            return {}, []
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
        chunk_map = {chunk.processor_name: chunk for chunk in new_chunks}
        return chunk_map, new_chunks

    @staticmethod
    def fuse_chunks(
        ctm_instance: ConsciousTuringMachine,
        chunks: List[Chunk],
        query: str,
        **input_kwargs: Any,
    ) -> List[Chunk]:
        """
        调用 fuse_processor 融合处理器输出。

        参数:
            ctm_instance: CTM 实例
            chunks: 要融合的 chunks 列表
            query: 原始查询
            **input_kwargs: 其他输入参数 (text, image_path, audio_path 等)

        返回:
            List[Chunk]: 融合后的 chunks 列表
        """
        return cast(
            List[Chunk], ctm_instance.fuse_processor(chunks, query, **input_kwargs)
        )

    @staticmethod
    def uptree_competition(
        ctm_instance: ConsciousTuringMachine, chunks: List[Chunk]
    ) -> Chunk:
        """
        调用 uptree_competition 进行上树竞争。

        参数:
            ctm_instance: CTM 实例
            chunks: chunks 列表

        返回:
            Chunk: 获胜的 chunk
        """
        return ctm_instance.uptree_competition(chunks)

    @staticmethod
    def ask_supervisor(
        ctm_instance: ConsciousTuringMachine,
        query: str,
        winning_chunk: Chunk,
    ) -> Tuple[str, float]:
        """
        调用 ask_supervisor 获取最终答案。

        参数:
            ctm_instance: CTM 实例
            query: 原始查询
            winning_chunk: 获胜的 chunk

        返回:
            Tuple[str, float]: (答案, 置信度分数)
        """
        return ctm_instance.ask_supervisor(query, winning_chunk)

    @staticmethod
    def downtree_broadcast(
        ctm_instance: ConsciousTuringMachine,
        winning_chunk: Chunk,
    ) -> None:
        """
        调用 downtree_broadcast 进行下树广播。

        参数:
            ctm_instance: CTM 实例
            winning_chunk: 获胜的 chunk
        """
        ctm_instance.downtree_broadcast(winning_chunk)

    @staticmethod
    def link_form(
        ctm_instance: ConsciousTuringMachine,
        chunks: List[Chunk],
        winning_chunk: Chunk,
        **input_kwargs: Any,
    ) -> None:
        """
        调用 link_form 形成处理器之间的链接。

        参数:
            ctm_instance: CTM 实例
            chunks: chunks 列表
            winning_chunk: 获胜的 chunk
            **input_kwargs: 其他输入参数
        """
        ctm_instance.link_form(chunks, winning_chunk, **input_kwargs)

    @staticmethod
    def compete_chunks(
        chunk_manager: ChunkManager, chunk1: Chunk, chunk2: Chunk
    ) -> Chunk:
        """两个 chunk 之间的竞争"""
        if isinstance(chunk1, Chunk) and isinstance(chunk2, Chunk):
            return cast(Chunk, chunk_manager.compete(chunk1, chunk2))
        return chunk1 if isinstance(chunk1, Chunk) else chunk2
