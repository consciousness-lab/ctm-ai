import concurrent.futures
import random
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from ..chunks import Chunk, ChunkManager
from ..configs import ConsciousnessTuringMachineConfig
from ..graphs import ProcessorGraph
from ..processors import BaseProcessor
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import logging_func


class BaseConsciousnessTuringMachine(ABC):
    def __init__(self, ctm_name: Optional[str] = None) -> None:
        super().__init__()
        self.config = (
            ConsciousnessTuringMachineConfig.from_ctm(ctm_name)
            if ctm_name
            else ConsciousnessTuringMachineConfig()
        )
        self.load_ctm()

    def __call__(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Tuple[str, float]:
        return self.forward(
            query,
            text,
            image,
            image_path,
            audio,
            audio_path,
            video_frames,
            video_frames_path,
            video_path,
        )

    def reset(self) -> None:
        self.load_ctm()

    def load_ctm(self) -> None:
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []

        for group_name, processors in self.config.groups_of_processors.items():
            for processor_name in processors:
                self.processor_graph.add_node(
                    processor_name=processor_name, processor_group_name=group_name
                )

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)

    def add_processor(
        self, processor_name: str, group_name: Optional[str] = None
    ) -> None:
        self.processor_graph.add_node(processor_name, group_name)

    def remove_processor(self, processor_name: str) -> None:
        self.processor_graph.remove_node(processor_name)

    def add_supervisor(self, name: str) -> None:
        self.supervisors.append(BaseSupervisor(name))

    def remove_supervisor(self, name: str) -> None:
        self.supervisors = [
            supervisor for supervisor in self.supervisors if supervisor.name != name
        ]

    def add_scorer(self, name: str) -> None:
        self.scorers.append(BaseScorer(name))

    def remove_scorer(self, name: str) -> None:
        self.scorers = [scorer for scorer in self.scorers if scorer.name != name]

    @staticmethod
    def ask_processor(
        processor: BaseProcessor,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Chunk:
        return processor.ask(
            query,
            text,
            image,
            image_path,
            audio,
            audio_path,
            video_frames,
            video_frames_path,
            video_path,
        )

    @logging_func
    def ask_processors(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> List[Chunk]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.ask_processor,
                    processor,
                    query,
                    text,
                    image,
                    image_path,
                    audio,
                    audio_path,
                    video_frames,
                    video_frames_path,
                    video_path,
                )
                for processor in self.processor_graph.nodes
            ]
            chunks = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        assert len(chunks) == len(self.processor_graph.nodes)
        return chunks

    @logging_func
    def ask_supervisor(
        self, query: str, chunk: Chunk
    ) -> Tuple[Union[str, None], float]:
        final_answer, score = self.supervisors[0].ask(query, chunk.gist)
        return final_answer, score

    @logging_func
    def uptree_competition(self, chunks: List[Chunk]) -> Chunk:
        chunk_manager = ChunkManager(chunks)
        return chunk_manager.uptree_competition()

    @logging_func
    def downtree_broadcast(self, chunk: Chunk) -> None:
        """Enhanced downtree broadcast with processor information exchange"""
        # First, update all processors with the winning chunk
        for processor in self.processor_graph.nodes:
            processor.update(chunk)

        # Then, enable processors to ask each other for information to enhance their gists
        self._enable_processor_information_exchange()

    def _enable_processor_information_exchange(self) -> None:
        """Enable processors to exchange information to enhance their gists"""
        processors = list(self.processor_graph.nodes)

        for i, processor in enumerate(processors):
            # Get current gist from this processor
            current_gist = self._get_processor_current_gist(processor)
            if not current_gist:
                continue

            # Ask other processors if they can provide useful information
            for j, other_processor in enumerate(processors):
                if i == j:  # Skip self
                    continue

                # Check if these processors are connected in the graph
                if not self.processor_graph.has_link(
                    processor.name, other_processor.name
                ):
                    continue

                # Ask if the other processor can provide useful information
                should_include_info = self._ask_for_information_inclusion(
                    processor, other_processor, current_gist
                )

                if should_include_info:
                    # Get information from the other processor
                    other_info = self._get_processor_information(other_processor)
                    if other_info:
                        # Enhance current processor's gist with the information
                        self._enhance_processor_gist(processor, other_info)

    def _get_processor_current_gist(self, processor: BaseProcessor) -> Optional[str]:
        """Get the current gist from a processor's messenger"""
        try:
            # Get the most recent executor message which contains the gist
            executor_messages = processor.messenger.get_executor_messages()
            if executor_messages:
                latest_message = executor_messages[-1]
                return getattr(latest_message, 'gist', None)
        except Exception as e:
            print(f'Error getting gist from processor {processor.name}: {e}')
        return None

    def _get_processor_information(self, processor: BaseProcessor) -> Optional[str]:
        """Get useful information from a processor"""
        try:
            # Get the most recent executor message
            executor_messages = processor.messenger.get_executor_messages()
            if executor_messages:
                latest_message = executor_messages[-1]
                # Return gist if available, otherwise content
                return getattr(latest_message, 'gist', None) or getattr(
                    latest_message, 'content', None
                )
        except Exception as e:
            print(f'Error getting information from processor {processor.name}: {e}')
        return None

    def _ask_for_information_inclusion(
        self,
        requesting_processor: BaseProcessor,
        providing_processor: BaseProcessor,
        current_gist: str,
    ) -> bool:
        """Ask if the requesting processor should include information from the providing processor"""
        try:
            # Create a query to determine if information should be included
            query = f"""Based on my current understanding: "{current_gist}"

I am a {requesting_processor.name} processor. The {providing_processor.name} processor may have additional information that could enhance my understanding.

Should I include information from the {providing_processor.name} processor to make my understanding more complete and accurate?

Please respond with only 'YES' if the information would be valuable, or 'NO' if it would not add meaningful value."""

            # Use the requesting processor's executor to make this decision
            messages = [{'role': 'user', 'content': query}]
            response = requesting_processor.executor.ask(messages=messages)

            # Parse the response
            response_text = getattr(response, 'content', '') or getattr(
                response, 'gist', ''
            )
            return 'YES' in response_text.upper()

        except Exception as e:
            print(f'Error asking for information inclusion: {e}')
            return False

    def _enhance_processor_gist(
        self, processor: BaseProcessor, additional_info: str
    ) -> None:
        """Enhance a processor's gist with additional information"""
        try:
            # Create a query to enhance the gist
            current_gist = self._get_processor_current_gist(processor)
            if not current_gist:
                return

            query = f"""My current understanding is: "{current_gist}"

I have received additional information: "{additional_info}"

Please enhance my understanding by incorporating this additional information in a way that makes my response more complete and accurate. 
Provide an improved version that integrates both pieces of information seamlessly."""

            # Use the processor's executor to enhance the gist
            messages = [{'role': 'user', 'content': query}]
            enhanced_response = processor.executor.ask(messages=messages)

            # Update the processor's messenger with the enhanced gist
            enhanced_gist = getattr(enhanced_response, 'gist', None) or getattr(
                enhanced_response, 'content', None
            )
            if enhanced_gist:
                # Create a new message with the enhanced gist
                from ..messengers import Message

                enhanced_message = Message(
                    role='assistant', gist=enhanced_gist, content=enhanced_gist
                )
                processor.messenger.executor_messages.append(enhanced_message)

        except Exception as e:
            print(f'Error enhancing processor gist: {e}')

    @logging_func
    def link_form(self, chunks: List[Chunk]) -> None:
        chunk_manager = ChunkManager(chunks, self.config)
        interaction_matrix = chunk_manager.get_interaction_type_matrix()

        for i in range(len(interaction_matrix)):
            for j in range(i + 1, len(interaction_matrix)):
                interaction_type = interaction_matrix[i][j]

                if not self.processor_graph.has_node(
                    chunks[i].processor_name
                ) or not self.processor_graph.has_node(chunks[j].processor_name):
                    continue

                if interaction_type != 0:
                    self.processor_graph.add_link(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                    )
                else:
                    self.processor_graph.remove_link(
                        processor1_name=chunks[i].processor_name,
                        processor2_name=chunks[j].processor_name,
                    )

    @abstractmethod
    def forward(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[np.uint8] = None,
        image_path: Optional[str] = None,
        audio: Optional[NDArray[np.float32]] = None,
        audio_path: Optional[str] = None,
        video_frames: Optional[List[NDArray[np.uint8]]] = None,
        video_frames_path: Optional[List[str]] = None,
        video_path: Optional[str] = None,
    ) -> Tuple[str, float]:
        pass
