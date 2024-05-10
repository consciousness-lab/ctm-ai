import concurrent.futures
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..chunks import Chunk
from ..configs import BaseConsciousnessTuringMachineConfig
from ..fusers import BaseFuser
from ..processors import BaseProcessor
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor
from ..utils import calc_gist_sim


class BaseConsciousnessTuringMachine(object):
    def __init__(self, ctm_name: Optional[str] = None) -> None:
        super().__init__()
        if ctm_name:
            self.config = BaseConsciousnessTuringMachineConfig.from_ctm(
                ctm_name
            )
        else:
            self.config = BaseConsciousnessTuringMachineConfig()
        self.processor_list: List[Dict[str, Any]] = []
        self.processor_group_map: Dict[str, str] = defaultdict(str)
        self.processor_graph: Dict[str, Set[str]] = defaultdict(set)
        self.load_ctm()

    def __call__(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        video_frames: Optional[Any] = None,
    ) -> Tuple[str, float]:
        return self.forward(query, text, image, audio, video_frames)

    def add_processor(
        self, processor_name: str, group_name: Optional[str] = "default_group"
    ) -> None:
        processor_instance = BaseProcessor(processor_name)
        self.processor_list.append(
            {
                "processor_name": processor_name,
                "processor_instance": processor_instance,
            }
        )
        if group_name:
            self.processor_group_map[processor_name] = group_name

    def remove_processor(self, processor_name: str) -> None:
        for processor in self.processor_list:
            if processor["processor_name"] == processor_name:
                self.processor_list.remove(processor)
                break
        self.processor_group_map.pop(processor_name, None)

    def add_supervisor(self, supervisor_name: str) -> None:
        supervisor_instance = BaseSupervisor(supervisor_name)
        self.supervisor: Dict[str, Any] = {
            "supervisor_name": supervisor_name,
            "supervisor_instance": supervisor_instance,
        }

    def remove_supervisor(self, supervisor_name: str) -> None:
        self.supervisor.pop(supervisor_name, None)

    def add_scorer(self, scorer_name: str) -> None:
        scorer_instance = BaseScorer(scorer_name)
        self.scorer: Dict[str, Any] = {
            "scorer_name": scorer_name,
            "scorer_instance": scorer_instance,
        }

    def remove_scorer(self, scorer_name: str) -> None:
        self.scorer.pop(scorer_name, None)

    def add_fuser(self, fuser_name: str) -> None:
        fuser_instance = BaseFuser(fuser_name)
        self.fuser: Dict[str, Any] = {
            "fuser_name": fuser_name,
            "fuser_instance": fuser_instance,
        }

    def remove_fuser(self, fuser_name: str) -> None:
        self.fuser.pop(fuser_name, None)

    @staticmethod
    def ask_processor(
        processor: Dict[str, Any],
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        video_frames: Optional[Any] = None,
    ) -> Dict[str, Any]:
        processor_instance = processor["processor_instance"]
        processor_name = processor["processor_name"]
        print(processor_name)
        gist = processor_instance.ask(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        return {"processor_name": processor_name, "gist": gist}

    def ask_processors(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        video_frames: Optional[Any] = None,
    ) -> Dict[str, Dict[str, Any]]:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [
                executor.submit(
                    self.ask_processor,
                    processor,
                    query,
                    text,
                    image,
                    audio,
                    video_frames,
                )
                for processor in self.processor_list
            ]
            results = [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]

        output: Dict[str, Dict[str, Any]] = {}
        for result in results:
            output[result["processor_name"]] = {
                "gist": result["gist"],
            }

        assert len(output) == len(self.processor_list)
        return output

    def uptree_competition(self, chunks: List[Chunk]) -> Chunk:
        # Unpack processor outputs into lists for easier processing
        winning_chunks: List[Chunk] = []
        candidate_chunks: List[Chunk] = chunks
        for _ in range(len(chunks) - 1):
            for chunk1, chunk2 in zip(
                candidate_chunks[:-1], candidate_chunks[1:]
            ):
                winning_chunk = (
                    chunk1
                    if chunk1 > chunk2
                    else (
                        chunk2
                        if chunk1 < chunk2
                        else random.choice([chunk1, chunk2])
                    )
                )
                winning_chunks.append(winning_chunk)
            candidate_chunks = winning_chunks
            winning_chunks = []
        return candidate_chunks[0]

    def ask_scorer(
        self, processor_output: Dict[str, Dict[str, str]]
    ) -> Dict[str, Dict[str, float]]:
        processor_output_with_score: Dict[str, Dict[str, Any]] = {}
        for processor_name, processor_info in processor_output.items():
            processor_gist = processor_info["gist"]
            relevance, confidence, surrise = self.scorer[
                "scorer_instance"
            ].ask(query=processor_gist, gist=processor_gist, verbose=True)
            processor_output_with_score[processor_name] = {
                "processor_name": processor_name,
                "gist": processor_gist,
                "relevance": relevance,
                "confidence": confidence,
                "surprise": surrise,
            }
        return processor_output_with_score

    def ask_supervisor(self, query: str, chunk: Chunk) -> Tuple[str, float]:
        final_answer, score = self.supervisor["supervisor_instance"].ask(
            query, chunk.gist
        )
        return final_answer, score

    def downtree_broadcast(self, winning_chunk: Chunk) -> None:
        winning_processor_name = winning_chunk.processor_name
        winning_processor_gist = winning_chunk.gist
        for processor in self.processor_list:
            if processor["processor_name"] != winning_processor_name:
                processor["processor_instance"].update_info(
                    winning_processor_gist
                )
        return

    def link_form(self, chunks: List[Chunk]) -> None:
        sim = calc_gist_sim(chunks)
        print(sim)
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                if sim[i][j] > 0.5:
                    processor1_name = chunks[i].processor_name
                    processor2_name = chunks[j].processor_name
                    # link on the graph
                    self.processor_graph[processor1_name].add(processor2_name)
                    self.processor_graph[processor2_name].add(processor1_name)
                if sim[i][j] < 0.2:
                    processor1_name = chunks[i].processor_name
                    processor2_name = chunks[j].processor_name
                    # unlink on the graph
                    if (
                        processor2_name
                        in self.processor_graph[processor1_name]
                    ):
                        self.processor_graph[processor1_name].remove(
                            processor2_name
                        )
                    if (
                        processor1_name
                        in self.processor_graph[processor2_name]
                    ):
                        self.processor_graph[processor2_name].remove(
                            processor1_name
                        )
        return

    def collect_chunks(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        video_frames: Optional[Any] = None,
    ):
        processor_output = self.ask_processors(
            query=query,
            text=text,
            image=image,
            audio=audio,
            video_frames=video_frames,
        )
        processor_output_with_score = self.ask_scorer(processor_output)
        chunks: List[Chunk] = []
        for (
            processor_name,
            processor_info,
        ) in processor_output_with_score.items():
            processor_gist = processor_info["gist"]
            relevance, confidence, surprise = (
                processor_info["relevance"],
                processor_info["confidence"],
                processor_info["surprise"],
            )
            weight = relevance * confidence * surprise
            intensity = weight
            mood = weight
            chunk = Chunk(
                processor_name=processor_name,
                time_step=0,
                gist=processor_gist,
                relevance=relevance,
                confidence=confidence,
                surprise=surprise,
                weight=weight,
                intensity=intensity,
                mood=mood,
            )
            chunks.append(chunk)
        return chunks

    def processor_fuse(self, chunks: List[Chunk]) -> List[Chunk]:
        chunk_pairs: List[Tuple[Chunk, Chunk]] = []
        for chunk in chunks:
            src_chunk = chunk
            tgt_processor_names = self.processor_graph[
                src_chunk.processor_name
            ]
            chunk_pairs.extend(
                [
                    (src_chunk, chunk)
                    for chunk in chunks
                    if chunk.processor_name in tgt_processor_names
                ]
            )
        for chunk_pair in chunk_pairs:
            fused_chunk = self.fuser.fuse(chunk_pair)
            chunks.append(fused_chunk)
        chunks = random.shuffle(chunks)
        return chunks

    def forward(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[Any] = None,
        audio: Optional[Any] = None,
        video_frames: Optional[Any] = None,
    ) -> Tuple[str, float]:
        answer_threshold = 0.5
        max_iter = 3

        for i in range(max_iter):
            print("start the {}-th iteration".format(i + 1))
            chunks = self.collect_chunks(
                query=query,
                text=text,
                image=image,
                audio=audio,
                video_frames=video_frames,
            )
            chunks = self.processor_fuse(chunks)
            winning_chunk = self.uptree_competition(chunks)
            answer, confidence_score = self.ask_supervisor(
                query, winning_chunk
            )
            if confidence_score > answer_threshold:
                break
            else:
                self.downtree_broadcast(winning_chunk)
                self.link_form(chunks)
        return answer, confidence_score

    def load_ctm(self) -> None:
        for (
            group_name,
            processor_list,
        ) in self.config.groups_of_processors.items():
            for processor_name in processor_list:
                self.add_processor(processor_name, group_name=group_name)
        self.add_supervisor(self.config.supervisor)
        self.add_scorer(self.config.scorer)
