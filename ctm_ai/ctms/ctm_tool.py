from typing import Callable, Dict, List, Optional, Tuple
from .ctm_base import BaseConsciousnessTuringMachine
from ..chunks import Chunk
from ..processors import BaseProcessor
from ..configs import ConsciousnessTuringMachineConfig
from ..utils import logging_func_with_count
from ..fusers import BaseFuser
from ..graphs import ProcessorGraph
from ..processors import BaseProcessor
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor


class ToolConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(
        self,
        tool_map: Dict[str, Callable[[dict], dict]],
        config: Optional[ConsciousnessTuringMachineConfig] = None
    ) -> None:
        self._tool_map = tool_map
        self.config = config or ConsciousnessTuringMachineConfig()
        super().__init__()

    def load_ctm(self) -> None:
        self.processor_graph = ProcessorGraph()
        self.supervisors: List[BaseSupervisor] = []
        self.scorers: List[BaseScorer] = []
        self.fusers: List[BaseFuser] = []

        for tool_name, tool_fn in self._tool_map.items():
            processor = ToolProcessor(tool_name, tool_fn)
            self.processor_graph.add_node(processor, group_name="tools")

        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)
        self.add_fuser(self.config.fuser)

    @logging_func_with_count
    def go_up(self, query: str) -> Tuple[Chunk, List[Chunk]]:
        chunks = self.ask_processors(query)
        chunks = self.fuse_processor(chunks)
        winning_chunk = self.uptree_competition(chunks)
        return winning_chunk, chunks

    @logging_func_with_count
    def go_down(self, winning_chunk: Chunk, chunks: List[Chunk]) -> None:
        self.downtree_broadcast(winning_chunk)
        self.link_form(chunks)

    def forward(self, query: str) -> Tuple[str, float]:
        for _ in range(self.config.max_iter_num):
            winning_chunk, chunks = self.go_up(query)
            answer, confidence_score = self.ask_supervisor(query, winning_chunk)
            confidence_score = 0 
            if confidence_score > self.config.output_threshold:
                return answer, confidence_score
            self.go_down(winning_chunk, chunks)
        return answer, confidence_score