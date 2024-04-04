from processors.processor_base import BaseProcessor
import concurrent.futures
import random
import openai
from collections import defaultdict
import numpy as np
from collections import Counter
from openai import OpenAI

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
    
    def add_answer_generator(self, answer_generator_name):
        answer_generator_instance = BaseProcessor(answer_generator_name)
        self.answer_generator = {'processor_name': answer_generator_name, 'processor_instance': answer_generator_instance}

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

    def answer_generation(self, question, processor_info):
        final_answer, score = self.answer_generator['processor_instance'].ask(question, processor_info['gist'])
        return final_answer, score

    def downtree_broadcast(self, gist):
        for processor in self.processor_list:
            processor['processor_instance'].base_prompt.append('hello, world')
        return 
        
    def link_form(self, infos, scores):
        return infos, scores
    
    def processor_fuse(self, infos, scores):
        return infos, scores
    
    def forward(self, question=None, context=None, image_path=None, audio_path=None, video_path=None):
        answer_threshold = 0.5
        max_iter = 3
        
        for i in range(max_iter):
            print('start the {}-th iteration'.format(i + 1))
            self.downtree_broadcast('hello')
            processor_output = self.ask_processors(question=question, image_path=image_path)
            winning_output = self.uptree_competition(processor_output)
            answer, score = self.answer_generation(question, winning_output)
            if score > answer_threshold:
                break
            else:
                self.downtree_broadcast(winning_output)
                self.link_form()

        return answer, score