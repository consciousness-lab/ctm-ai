from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from ctm_ai.ctms import BaseCTM
from ctm_ai.graphs import ProcessorGraph
from ctm_ai.supervisors import BaseSupervisor
from ctm_ai.utils import logging_func


class ConsciousTuringMachineBaseline(BaseCTM):
    def load_ctm(self) -> None:
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers = []

        for processor_name, processor_config in self.config.processors_config.items():
            self.processor_graph.add_node(
                processor_name=processor_name,
                processor_group_name=None,
                system_prompt=processor_config.get("system_prompt"),
                model=processor_config.get("model"),
            )

        self.add_supervisor(self.config.supervisor)

    @logging_func
    def ask_supervisor(
        self,
        query: str,
        gist: str,
    ) -> Tuple[str, float]:
        final_answer, score = self.supervisors[0].ask(query, gist)
        return final_answer, score

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
        chunks = self.ask_processors(
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

        prompt_header = (
            f"Based on the following processor outputs across different modalities, please provide the most accurate and comprehensive answer to the query: {query}.\n\n"
            "### Processor Outputs:\n"
        )

        processor_outputs = "\n".join(
            [
                f"**Processor [{chunk.processor_name}] Output:**\n{chunk.gist}\n"
                for chunk in chunks
            ]
        )

        combined_gist = prompt_header + processor_outputs

        final_answer, confidence_score = self.ask_supervisor(query, combined_gist)

        return final_answer, confidence_score
