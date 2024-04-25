import json
import os
from typing import Any, Dict, Optional

from huggingface_hub import InferenceClient

from ctm.messengers.messenger_base import BaseMessenger
from ctm.processors.processor_base import BaseProcessor


@BaseProcessor.register_processor("bart_text_summary_processor")
class BartTextSummaryProcessor(BaseProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args, **kwargs
        )  # Ensure base class is properly initialized

    def init_executor(self) -> None:
        hf_token = os.getenv("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set")
        self.executor = InferenceClient(token=hf_token)

    def init_messenger(self) -> None:
        self.messenger = BaseMessenger("bart_text_summ_messenger")

    def init_task_info(self) -> None:
        pass

    def update_info(self, feedback: str) -> None:
        self.messenger.add_assistant_message(feedback)

    def ask_info(
        self, text: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> str | Any:
        if text is None:
            raise ValueError("Context must not be None")
        if self.messenger.check_iter_round_num() == 0:
            self.messenger.add_user_message(text)

        response: Dict[str, Any] = json.loads(
            self.executor.post(
                json={"inputs": self.messenger.get_messages()},
                model="facebook/bart-large-cnn",
            )
        )[0]
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
