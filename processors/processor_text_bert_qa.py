from .processor_base import BaseProcessor
from huggingface_hub.inference_api import InferenceApi
import requests

@BaseProcessor.register_processor('text_bert_qa')
class TextBERTQAProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.TOKEN: str = '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'
        self.model_id: str = 'bert-large-uncased-whole-word-masking-finetuned-squad'
        self.api: InferenceApi = InferenceApi(repo_id=self.model_id, token=self.TOKEN)

    def process(self, payload: dict) -> dict:
        headers: dict = {"Authorization": f"Bearer {self.TOKEN}"}
        API_URL: str = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def score(self, payload: dict) -> dict:
        headers: dict = {"Authorization": f"Bearer {self.TOKEN}"}
        API_URL: str = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def ask_info(self, query: str, context: str) -> str:
        input_data: dict = {"inputs": {"question": query, "context": context}}
        output: dict = self.process(input_data)
        answer: str = output['answer']
        return answer

    def ask_score(self, query: str, context: str) -> float:
        input_data: dict = {"inputs": {"question": query, "context": context}}
        output: dict = self.score(input_data)
        score: float = output['score']
        assert 0 <= score <= 1
        return score


if __name__ == "__main__":
    processor = BaseProcessor('text_bert_qa')
    query: str = 'What is the capital of India?'
    context: str = 'The capital of India is New Delhi.'
    answer: str = processor.ask_info(query, context)
    score: float = processor.ask_score(query, context)
    print(answer, score)
