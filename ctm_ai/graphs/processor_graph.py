from typing import Dict, List, Optional

from ..processors import BaseProcessor


class ProcessorGraph:
    def __init__(self) -> None:
        self.nodes: List[BaseProcessor] = []
        self.adjacency_list: Dict[str, List[str]] = {}

    def add_node(
        self,
        processor_name: str,
        processor_group_name: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if processor_name not in [p.name for p in self.nodes]:
            processor = BaseProcessor(
                name=processor_name,
                group_name=processor_group_name,
                system_prompt=system_prompt,
            )
            self.nodes.append(processor)
            self.adjacency_list[processor_name] = []

    def remove_node(self, processor_name: str) -> None:
        node_to_remove = self.get_node(processor_name)
        if node_to_remove:
            self.nodes.remove(node_to_remove)
            del self.adjacency_list[processor_name]
            for neighbors in self.adjacency_list.values():
                if processor_name in neighbors:
                    neighbors.remove(processor_name)

    def add_link(self, processor1_name: str, processor2_name: str) -> None:
        if (
            processor1_name in self.adjacency_list
            and processor2_name in self.adjacency_list
        ):
            self.adjacency_list[processor1_name].append(processor2_name)
            self.adjacency_list[processor2_name].append(processor1_name)

    def remove_link(self, processor1_name: str, processor2_name: str) -> None:
        if (
            processor1_name in self.adjacency_list
            and processor2_name in self.adjacency_list[processor1_name]
        ):
            self.adjacency_list[processor1_name].remove(processor2_name)
        if (
            processor2_name in self.adjacency_list
            and processor1_name in self.adjacency_list[processor2_name]
        ):
            self.adjacency_list[processor2_name].remove(processor1_name)

    def get_node(self, processor_name: str) -> Optional[BaseProcessor]:
        for node in self.nodes:
            if node.name == processor_name:
                return node
        return None

    def get_neighbor_names(self, processor_name: str) -> List[str]:
        return self.adjacency_list.get(processor_name, [])

    def __len__(self) -> int:
        return len(self.nodes)

    def has_node(self, processor_name: str) -> bool:
        return any(node.name == processor_name for node in self.nodes)
