from typing import Dict, List, Set

from ..processors import BaseProcessor


class ProcessorGraph(object):
    def __init__(self):
        self.graph: Dict[BaseProcessor, Set[BaseProcessor]] = {}

    def add_node(self, processor_name: str, processor_group_name: str) -> None:
        processor = BaseProcessor(processor_name, processor_group_name)
        self.graph[processor] = set()

    def remove_node(self, processor_name: str) -> None:
        processor = self.get_node(processor_name)
        for conn in list(self.graph[processor]):
            self.graph[conn].discard(processor)
        del self.graph[processor]

    def add_link(self, processor1_name: str, processor2_name: str) -> None:
        processor1 = self.get_node(processor1_name)
        processor2 = self.get_node(processor2_name)
        self.graph[processor1].add(processor2)
        self.graph[processor2].add(processor1)

    def remove_link(self, processor1_name: str, processor2_name: str) -> None:
        processor1 = self.get_node(processor1_name)
        processor2 = self.get_node(processor2_name)
        if processor2 in self.graph[processor1]:
            self.graph[processor1].remove(processor2)
        if processor1 in self.graph[processor2]:
            self.graph[processor2].remove(processor1)

    def get_node(self, processor_name: str) -> BaseProcessor:
        for processor in self.graph.keys():
            if processor.name == processor_name:
                return processor
        raise ValueError(
            f"Processor with name {processor_name} not found in graph"
        )

    def get_linked_node_names(self, processor_name: str) -> List[str]:
        processor = self.get_node(processor_name)
        return [node.name for node in self.graph[processor]]

    @property
    def nodes(self) -> List[BaseProcessor]:
        return list(self.graph.keys())

    def __len__(self) -> int:
        return len(self.graph)
