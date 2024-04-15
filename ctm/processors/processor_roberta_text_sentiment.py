import os

from huggingface_hub.inference_api import (
    InferenceApi,  # type: ignore[import] # FIX ME
)

from ctm.messengers.messenger_base import BaseMessenger
from ctm.processors.processor_base import BaseProcessor


@BaseProcessor.register_processor("roberta_text_sentiment_processor")  # type: ignore[no-untyped-call] # FIX ME
class RobertaTextSentimentProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_processor()  # type: ignore[no-untyped-call] # FIX ME

    def init_processor(self):  # type: ignore[no-untyped-def] # FIX ME
        self.model = InferenceApi(
            token=os.environ["HF_TOKEN"],
            repo_id="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )
        self.messenger = BaseMessenger("roberta_text_sentiment_messenger")  # type: ignore[no-untyped-call] # FIX ME
        return

    def update_info(self, feedback: str):  # type: ignore[no-untyped-def] # FIX ME
        self.messenger.add_assistant_message(feedback)

    def ask_info(  # type: ignore[override] # FIX ME
        self,
        query: str,
        text: str = None,  # type: ignore[assignment] # FIX ME
        *args,
        **kwargs,
    ) -> str:
        if self.messenger.check_iter_round_num() == 0:  # type: ignore[no-untyped-call] # FIX ME
            self.messenger.add_user_message(text)

        response = self.model(self.messenger.get_messages())  # type: ignore[no-untyped-call] # FIX ME
        results = response[0]
        # choose the label with the highest score
        pos_score = 0
        neg_score = 0
        neutral_score = 0
        for result in results:
            if result["label"] == "POSITIVE":
                pos_score = result["score"]
            elif result["label"] == "NEGATIVE":
                neg_score = result["score"]
            else:
                neutral_score = result["score"]
        if max(pos_score, neg_score, neutral_score) == pos_score:
            return "This text is positive."
        elif max(pos_score, neg_score, neutral_score) == neg_score:
            return "This text is negative."
        else:
            return "This text is neutral."


if __name__ == "__main__":
    processor = BaseProcessor("roberta_text_sentiment_processor")  # type: ignore[no-untyped-call] # FIX ME
    image_path = "../ctmai-test1.png"
    text: str = (
        "In a shocking turn of events, Hugging Face has released a new version of Transformers "
        "that brings several enhancements and bug fixes. Users are thrilled with the improvements "
        "and are finding the new version to be significantly better than the previous one. "
        "The Hugging Face team is thankful for the community's support and continues to work "
        "towards making the library the best it can be."
    )
    label = processor.ask_info(query=None, context=text, image_path=image_path)  # type: ignore[no-untyped-call] # FIX ME
    print(label)
