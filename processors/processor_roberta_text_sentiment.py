import sys
sys.path.append('..')
from processor_base import BaseProcessor
from messengers.messenger_base import BaseMessenger
from huggingface_hub.inference_api import InferenceApi
import os

@BaseProcessor.register_processor('roberta_text_sentiment_processor')
class RobertaTextSentimentProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.model = InferenceApi(
            token=os.environ['HF_TOKEN'], 
            repo_id='cardiffnlp/twitter-roberta-base-sentiment-latest'
        )
        self.messenger = BaseMessenger('roberta_text_sentiment_messenger')
        return

    def update_info(self, feedback: str):
        self.messenger.add_assistant_message(feedback)

    def ask_info(self, query: str, context: str = None, image_path: str = None, audio_path: str = None, video_path: str = None) -> str:
        if self.messenger.check_iter_round_num() == 0:
            self.messenger.add_user_message(context)

        response = self.model(self.messenger.get_messages())
        results = response[0]
        # choose the label with the highest score
        pos_score = 0
        neg_score = 0
        neutral_score = 0
        for result in results:
            if result['label'] == 'POSITIVE':
                pos_score = result['score']
            elif result['label'] == 'NEGATIVE':
                neg_score = result['score']
            else:
                neutral_score = result['score']
        if max(pos_score, neg_score, neutral_score) == pos_score:
            return 'This text is positive.'
        elif max(pos_score, neg_score, neutral_score) == neg_score:
            return 'This text is negative.'
        else:
            return 'This text is neutral.'


if __name__ == "__main__":
    processor = BaseProcessor('roberta_text_sentiment_processor')
    image_path = '../ctmai-test1.png'
    text: str = (
        "In a shocking turn of events, Hugging Face has released a new version of Transformers "
        "that brings several enhancements and bug fixes. Users are thrilled with the improvements "
        "and are finding the new version to be significantly better than the previous one. "
        "The Hugging Face team is thankful for the community's support and continues to work "
        "towards making the library the best it can be."
    )
    label = processor.ask_info(query=None, context=text, image_path=image_path)
    print(label)
