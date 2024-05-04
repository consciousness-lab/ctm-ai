import concurrent.futures
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..configs import BaseConsciousnessTuringMachineConfig
from ..processors import BaseProcessor
from ..scorers import BaseScorer
from ..supervisors import BaseSupervisor


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
        return {"name": processor_name, "gist": gist}

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
            output[result["name"]] = {
                "gist": result["gist"],
            }

        assert len(output) == len(self.processor_list)
        return output

    def uptree_competition(
        self, processor_output: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        # Unpack processor outputs into lists for easier processing
        gists: List[str] = []
        scores: List[float] = []
        names: List[str] = []

        for name, info in processor_output.items():
            gists.append(info["gist"])
            scores.append(info["score"])
            names.append(name)

        # Determine the unique group for each processor
        unique_groups: Set[str] = set(self.processor_group_map.values())

        # Prepare to track the best processor by group
        best_processor_by_group: Dict[str, Tuple[Optional[str], float]] = {
            group: (
                None,
                float("-inf"),
            )  # Use negative infinity as the initial lowest score
            for group in unique_groups
        }

        # Iterate through processors to find the best in each group
        for name, score in zip(names, scores):
            group = self.processor_group_map.get(name, "")
            if score > best_processor_by_group[group][1]:
                best_processor_by_group[group] = (name, score)

        # Select the overall best across groups
        best_overall: Tuple[Optional[str], float] = max(
            best_processor_by_group.values(), key=lambda x: x[1]
        )
        best_name: Optional[str] = best_overall[0]

        if best_name is None:
            raise ValueError(
                "No valid processor found."
            )  # Ensure best_name is not None

        index: int = names.index(
            best_name
        )  # Now best_name is guaranteed to be not None

        winning_info: Dict[str, Any] = {
            "name": best_name,
            "gist": gists[index],
            "score": scores[index],
        }
        return winning_info

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
                "gist": processor_gist,
                "relevance": relevance,
                "confidence": confidence,
                "surprise": surrise,
            }
        return processor_output_with_score

    def ask_supervisor(
        self, query: str, processor_info: Dict[str, Any]
    ) -> Tuple[str, float]:
        final_answer, score = self.supervisor["supervisor_instance"].ask(
            query, processor_info["gist"]
        )
        return final_answer, score

    def downtree_broadcast(self, winning_output: Dict[str, str]) -> None:
        winning_processor_name = winning_output["name"]
        winning_processor_gist = winning_output["gist"]
        for processor in self.processor_list:
            if processor["processor_name"] != winning_processor_name:
                processor["processor_instance"].update_info(
                    winning_processor_gist
                )
        return

    def calc_processor_sim(
        self, processor_output: Dict[str, Dict[str, float]]
    ) -> Any:
        processor_gists = [info["gist"] for info in processor_output.values()]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(processor_gists)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim

    def link_form(self, processor_output: Dict[str, Dict[str, float]]) -> None:
        sim = self.calc_processor_sim(processor_output)
        print(sim)
        # iterate on each sim pair
        # if sim > threshold, then link the two processors by combining them into the same group
        link_threshold = 0.5
        for i in range(len(sim)):
            for j in range(i + 1, len(sim)):
                if sim[i][j] > 0.5:
                    processor1_name = list(processor_output.keys())[i]
                    processor2_name = list(processor_output.keys())[j]
                    # choose the group that includes more processors
                    # processor_group_map is a dict with processor_name as key and group_name as value
                    group1 = self.processor_group_map[processor1_name]
                    group2 = self.processor_group_map[processor2_name]
                    # calculate the number of processors in each group
                    group1_count = sum(
                        [
                            1
                            for group in self.processor_group_map.values()
                            if group == group1
                        ]
                    )
                    group2_count = sum(
                        [
                            1
                            for group in self.processor_group_map.values()
                            if group == group2
                        ]
                    )
                    # choose the group with more processors
                    group_name = (
                        group1 if group1_count > group2_count else group2
                    )
                    self.processor_group_map[processor1_name] = group_name
                    self.processor_group_map[processor2_name] = group_name
        return

    def processor_fuse(
        self, infos: List[str], scores: List[float]
    ) -> Tuple[List[str], List[float]]:
        return infos, scores

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
            processor_output = self.ask_processors(
                query=query,
                text=text,
                image=image,
                audio=audio,
                video_frames=video_frames,
            )
            processor_output_with_score = self.ask_scorer(processor_output)
            winning_output = self.uptree_competition(
                processor_output_with_score
            )
            answer, confidence_score = self.ask_supervisor(
                query, winning_output
            )
            if confidence_score > answer_threshold:
                break
            else:
                self.downtree_broadcast(winning_output)
                self.link_form(processor_output_with_score)
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
