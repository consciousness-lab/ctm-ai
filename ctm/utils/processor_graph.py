from typing import Dict, Set

from ..processors import BaseProcessor


def add_node_on_processor_graph(
    processor_name: str,
    processor_group_name: str,
    processor_graph: Dict[BaseProcessor, Set[BaseProcessor]],
) -> Dict[BaseProcessor, Set[BaseProcessor]]:
    processor = BaseProcessor(
        processor_name,
        processor_group_name,
    )
    processor_graph[processor] = set()
    return processor_graph


def remove_node_on_processor_graph(
    processor_name: str,
    processor_graph: Dict[BaseProcessor, Set[BaseProcessor]],
) -> Dict[BaseProcessor, Set[BaseProcessor]]:
    processor = get_node_from_processor_graph(
        processor_name=processor_name,
        processor_graph=processor_graph,
    )
    for conn in list(processor_graph[processor]):
        processor_graph[conn].discard(processor)
    del processor_graph[processor]


def add_link_on_processor_graph(
    processor1_name: str,
    processor2_name: str,
    processor_graph: Dict[BaseProcessor, Set[BaseProcessor]],
) -> Dict[BaseProcessor, Set[BaseProcessor]]:
    processor1 = get_node_from_processor_graph(
        processor_name=processor1_name,
        processor_graph=processor_graph,
    )
    processor2 = get_node_from_processor_graph(
        processor_name=processor2_name,
        processor_graph=processor_graph,
    )
    processor_graph[processor1].add(processor2)
    processor_graph[processor2].add(processor1)
    return processor_graph


def remove_link_on_processor_graph(
    processor1_name: str,
    processor2_name: str,
    processor_graph: Dict[BaseProcessor, Set[BaseProcessor]],
) -> Dict[BaseProcessor, Set[BaseProcessor]]:
    processor1 = get_node_from_processor_graph(
        processor_name=processor1_name,
        processor_graph=processor_graph,
    )
    processor2 = get_node_from_processor_graph(
        processor_name=processor2_name,
        processor_graph=processor_graph,
    )
    if processor2 in processor_graph[processor1]:
        processor_graph[processor1].remove(processor2)

    if processor1 in processor_graph[processor2]:
        processor_graph[processor2].remove(processor1)

    return processor_graph


def get_node_from_processor_graph(
    processor_name: str,
    processor_graph: Dict[BaseProcessor, Set[BaseProcessor]],
) -> BaseProcessor:
    for processor in processor_graph.keys():
        if processor.name == processor_name:
            return processor
