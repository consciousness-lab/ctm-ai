from processors.processor_base import BaseProcessor
from supervisors.supervisor_base import BaseSupervisor
import concurrent.futures
from collections import defaultdict
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class BaseConsciousnessTuringMachine(object):
    _ctm_registry = {}

    @classmethod
    def register_ctm(cls, ctm_name):
        def decorator(subclass):
            cls._ctm_registry[ctm_name] = subclass
            return subclass
        return decorator

    def __new__(cls, ctm_name, *args, **kwargs):
        if ctm_name not in cls._ctm_registry:
            raise ValueError(f"No CTM registered with name '{ctm_name}'")
        return super(BaseConsciousnessTuringMachine, cls).__new__(cls._ctm_registry[ctm_name])
    
    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def __init__(self, ctm_name, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.processor_list = []
        self.processor_group_map = defaultdict(list)

    def add_processor(self, processor_name, group_name=None):
        processor_instance = BaseProcessor(processor_name)
        self.processor_list.append({'processor_name': processor_name, 'processor_instance': processor_instance})
        if group_name:
            self.processor_group_map[processor_name] = group_name
    
    def add_supervisor(self, supervisor_name):
        supervisor_instance = BaseSupervisor(supervisor_name)
        self.supervisor = {'supervisor_name': supervisor_name, 'supervisor_instance': supervisor_instance}

    @staticmethod
    def ask_processor(processor, question, image_path):
        processor_instance = processor['processor_instance']
        processor_name = processor['processor_name']
        gist, score = processor_instance.ask(question, image_path)
        return {
            'name': processor_name,
            'gist': gist,
            'score': score
        }

    def ask_processors(self, question, image_path=None):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = [executor.submit(self.ask_processor, processor, question, image_path) for processor in self.processor_list]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]
        
        output = {}
        for result in results:
            output[result['name']] = {'gist': result['gist'], 'score': result['score']}

        assert len(output) == len(self.processor_list)
        return output

    def uptree_competition(self, processor_output):
        # Unpack processor outputs into lists for easier processing
        gists, scores, names = [], [], []
        for name, info in processor_output.items():
            gists.append(info['gist'])
            scores.append(info['score'])
            names.append(name)
        
        # Determine the unique group for each processor
        unique_groups = set(self.processor_group_map.values())
        
        # Prepare to track the best processor by group
        best_processor_by_group = {group: (None, -1) for group in unique_groups}  # (processor_name, score)
        
        # Iterate through processors to find the best in each group
        for name, score in zip(names, scores):
            group = self.processor_group_map[name]
            if score > best_processor_by_group[group][1]:
                best_processor_by_group[group] = (name, score)
        
        # Select the overall best across groups
        best_overall = max(best_processor_by_group.values(), key=lambda x: x[1])
        best_name = best_overall[0]
        index = names.index(best_name)
        
        winning_info = {
            'name': best_name, 
            'gist': gists[index], 
            'score': scores[index]
        }
        return winning_info

    def ask_supervisor(self, question, processor_info):
        final_answer, score = self.supervisor['supervisor_instance'].ask(question, processor_info['gist'])
        return final_answer, score

    def downtree_broadcast(self, winning_output):
        winning_processor_name = winning_output['name']
        winning_processor_gist = winning_output['gist']
        for processor in self.processor_list:
            if processor['processor_name'] != winning_processor_name:
                processor['processor_instance'].update_info(winning_processor_gist)
        return 
    
    def calc_processor_sim(self, processor_output):
        processor_gists = [info['gist'] for info in processor_output.values()]
        tfidf_vectorizer = TfidfVectorizer()
        tfidf_matrix = tfidf_vectorizer.fit_transform(processor_gists)
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return cosine_sim
        
    def link_form(self, processor_output):
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
                    group1_count = sum([1 for group in self.processor_group_map.values() if group == group1])
                    group2_count = sum([1 for group in self.processor_group_map.values() if group == group2])
                    # choose the group with more processors
                    group_name = group1 if group1_count > group2_count else group2
                    self.processor_group_map[processor1_name] = group_name
                    self.processor_group_map[processor2_name] = group_name
        return
    
    def processor_fuse(self, infos, scores):
        return infos, scores
    
    def forward(self, question=None, context=None, image_path=None, audio_path=None, video_path=None):
        answer_threshold = 0.5
        max_iter = 3
        
        for i in range(max_iter):
            print('start the {}-th iteration'.format(i + 1))
            processor_output = self.ask_processors(question=question, image_path=image_path)
            winning_output = self.uptree_competition(processor_output)
            answer, score = self.ask_supervisor(question, winning_output)
            if score > answer_threshold:
                break
            else:
                self.downtree_broadcast(winning_output)
                self.link_form(processor_output)
        return answer, score
    
    def craft_ctm(self):
        pass
