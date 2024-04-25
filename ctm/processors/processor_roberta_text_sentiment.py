import os
from typing import Any, Optional

from huggingface_hub import InferenceClient

from ..messengers.messenger_base import BaseMessenger
from .processor_base import BaseProcessor


@BaseProcessor.register_processor("roberta_text_sentiment_processor")
class RobertaTextSentimentProcessor(BaseProcessor):
    def init_processor(self) -> None:
        self.processor = InferenceClient(
            token=os.environ["HF_TOKEN"],
            model="cardiffnlp/twitter-roberta-base-sentiment-latest",
        )

    def init_messenger(self) -> None:
        self.messenger = BaseMessenger("roberta_text_sentiment_messenger")

    def update_info(self, feedback: str) -> None:
        self.messenger.add_assistant_message(feedback)

    def ask_info(
        self,
        query: Optional[str],
        text: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if text and self.messenger.check_iter_round_num() == 0:
            self.messenger.add_user_message(text)

        response = self.processor(self.messenger.get_messages())
        results = response[0]
        pos_score = (
            neg_score
        ) = neutral_score = 0.0  # Initialize scores as floats
        for result in results:
            if result["label"] == "POSITIVE":
                pos_score = result["score"]
            elif result["label"] == "NEGATIVE":
                neg_score = result["score"]
            else:
                neutral_score = result["score"]

        # Simplified decision structure
        max_score = max(pos_score, neg_score, neutral_score)
        if max_score == pos_score:
            return "This text is positive."
        elif max_score == neg_score:
            return "This text is negative."
        return "This text is neutral."
