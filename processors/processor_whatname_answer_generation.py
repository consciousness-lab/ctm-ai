from .processor_base import BaseProcessor
from openai import OpenAI
from collections import Counter
import numpy as np


@BaseProcessor.register_processor('whatname_answer_generation_processor')
class WhatNameAnswerGenerationProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.client = OpenAI()
        self.base_prompt = []
        return

    def process(self, payload: dict) -> dict:
        return

    def ask_info(self, query: str, context: str = None) -> str:
        prompt = self.base_prompt + [{"role": "user", "content": f"The following is detailed information on the topic: {context}. Based on this information, answer the question: {query}. Answer with a few words:"}]
        try:
            responses = self.client.chat.completions.create(
                model="gpt-4-turbo-preview",
                messages=prompt,
                max_tokens=300,
                n=1
            )
            answer = responses.choices[0].message.content
        except Exception as e:
            print(f"An error occurred: {e}")
            answer = None
        return answer
    
    def ask_score(self, query, gist, verbose=False, *args, **kwargs):
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "user", "content": "How related is the information ({}) with the query ({})? We want to make sure that the information includes a person's name as the answer. Answer with a number from 0 to 5 and do not add any other thing.".format(gist, query)},
                    ],
                    max_tokens=50,
                )
                score = int(response.choices[0].message.content.strip()) / 5
                return score
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                else:
                    print("Max attempts reached. Returning default score.")
        return 0


if __name__ == "__main__":
    processor = BaseProcessor('cloth_fashion_processor')
    image_path = '../ctmai-test1.png'
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
