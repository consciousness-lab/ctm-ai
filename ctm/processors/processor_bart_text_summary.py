import os
from typing import Any, Dict, Optional

from huggingface_hub.inference_api import InferenceApi

from ctm.messengers.messenger_base import BaseMessenger
from ctm.processors.processor_base import BaseProcessor


@BaseProcessor.register_processor("bart_text_summary_processor")
class BartTextSummaryProcessor(BaseProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args, **kwargs
        )  # Ensure base class is properly initialized
        self.init_processor()

    def init_processor(self) -> None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")
        self.model = InferenceApi(
            token=hf_token, repo_id="facebook/bart-large-cnn"
        )
        self.messenger = BaseMessenger("bart_text_summ_messenger")

    def update_info(self, feedback: str) -> None:
        self.messenger.add_assistant_message(feedback)

    def ask_info(
        self, context: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> str | Any:
        if context is None:
            raise ValueError("Context must not be None")
        if self.messenger.check_iter_round_num() == 0:
            self.messenger.add_user_message(context)

        response: Dict[str, Any] = self.model(self.messenger.get_messages())[0]
        return response["summary_text"]


if __name__ == "__main__":
    processor = BartTextSummaryProcessor()
    image_path = "../ctmai-test1.png"
    text = (
        "In a shocking turn of events, Hugging Face has released a new version of Transformers "
        "that brings several enhancements and bug fixes. Users are thrilled with the improvements "
        "and are finding the new version to be significantly better than the previous one. "
        "The Hugging Face team is thankful for the community's support and continues to work "
        "towards making the library the best it can be."
    )
    summary = processor.ask_info(context=text, image_path=image_path)
    print(summary)
