from typing import Dict, List, Optional, Set

from ..processors import BaseProcessor
from ..utils import logger


class ProcessorGraph(object):
    def __init__(self) -> None:
        self.graph: Dict[BaseProcessor, Set[BaseProcessor]] = {}

    def add_node(
        self, processor_name: str, processor_group_name: Optional[str] = None
    ) -> None:
        processor = BaseProcessor(name=processor_name, group_name=processor_group_name)
        self.graph[processor] = set()
        logger.info(f'Added processor {processor_name} to graph')

    def remove_node(self, processor_name: str) -> None:
        processor = self.get_node(processor_name)
        for conn in list(self.graph[processor]):
            self.graph[conn].discard(processor)
        logger.info(f'Removed processor {processor_name} from graph')
        del self.graph[processor]

    def add_link(self, processor1_name: str, processor2_name: str) -> None:
        processor1 = self.get_node(processor1_name)
        processor2 = self.get_node(processor2_name)
        self.graph[processor1].add(processor2)
        self.graph[processor2].add(processor1)
        logger.info(f'Added link between {processor1_name} and {processor2_name}')

    def remove_link(self, processor1_name: str, processor2_name: str) -> None:
        processor1 = self.get_node(processor1_name)
        processor2 = self.get_node(processor2_name)
        if processor2 in self.graph[processor1]:
            self.graph[processor1].remove(processor2)
        if processor1 in self.graph[processor2]:
            self.graph[processor2].remove(processor1)
        logger.info(f'Removed link between {processor1_name} and {processor2_name}')

    def get_node(self, processor_name: str) -> BaseProcessor:
        for processor in self.graph.keys():
            if processor.name == processor_name:
                return processor
        raise ValueError(f'Processor with name {processor_name} not found in graph')

    def get_neighbor(self, processor_name: str) -> List[BaseProcessor]:
        processor = self.get_node(processor_name)
        return [node for node in self.graph[processor]]

    def get_neighbor_names(self, processor_name: str) -> List[str]:
        processor = self.get_node(processor_name)
        return [node.name for node in self.graph[processor]]

    @property
    def nodes(self) -> List[BaseProcessor]:
        return list(self.graph.keys())

    def __len__(self) -> int:
        return len(self.graph)
