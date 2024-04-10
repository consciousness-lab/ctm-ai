from .processor_base import BaseProcessor
from huggingface_hub.inference_api import InferenceApi
import requests

@BaseProcessor.register_processor('text_bert_sst')
class TextBERTSSTProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.TOKEN: str = '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'
        self.model_id: str = 'nlptown/bert-base-multilingual-uncased-sentiment'
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
        input_data: dict = {"inputs": context}
        output: dict = self.process(input_data)
        sentiment: str = output['label']
        return sentiment

    def ask_score(self, query: str, context: str) -> float:
        input_data: dict = {"inputs": context}
        output: dict = self.score(input_data)
        score: float = max(output['scores'])
        assert 0 <= score <= 1
        return score

if __name__ == "__main__":
    processor = BaseProcessor('text_bert_sst')
    text: str = 'I love the new design of your product!'
    sentiment: str = processor.ask_info(text)
    confidence: float = processor.ask_score(text)
    print(f'Sentiment: {sentiment}, Confidence: {confidence}')
