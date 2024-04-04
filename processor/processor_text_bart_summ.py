from .processor_base import BaseProcessor
from huggingface_hub.inference_api import InferenceApi
import requests

@BaseProcessor.register_processor('text_bart_summarization')
class TextBARTSummProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.TOKEN: str = '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'
        self.model_id: str = 'facebook/bart-large-cnn'
        self.api: InferenceApi = InferenceApi(repo_id=self.model_id, token=self.TOKEN)

    def process(self, payload: dict) -> dict:
        headers: dict = {"Authorization": f"Bearer {self.TOKEN}"}
        API_URL: str = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()
    
    def weigh(self, payload: dict) -> dict:
        headers: dict = {"Authorization": f"Bearer {self.TOKEN}"}
        API_URL: str = f"https://api-inference.huggingface.co/models/{self.model_id}"
        response = requests.post(API_URL, headers=headers, json=payload)
        return response.json()

    def ask_info(self, query: str, context: str) -> str:
        input_data: dict = {"inputs": text}
        output: dict = self.process(input_data)
        summary: str = output[0]['summary_text']
        return summary
    
    def ask_weight(self, query: str, context: str) -> float:
        score = 0.5
        return score

if __name__ == "__main__":
    processor = BaseProcessor('text_bart_summarization')
    text: str = (
        "In a shocking turn of events, Hugging Face has released a new version of Transformers "
        "that brings several enhancements and bug fixes. Users are thrilled with the improvements "
        "and are finding the new version to be significantly better than the previous one. "
        "The Hugging Face team is thankful for the community's support and continues to work "
        "towards making the library the best it can be."
    )
    summary: str = processor.ask_info(query=None, context=text)
    print(summary)
