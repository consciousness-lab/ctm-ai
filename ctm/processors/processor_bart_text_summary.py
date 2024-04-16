import os

from huggingface_hub.inference_api import InferenceApi

from ctm.messengers.messenger_base import BaseMessenger
from ctm.processors.processor_base import BaseProcessor


@BaseProcessor.register_processor("bart_text_summary_processor")  # type: ignore[no-untyped-call] # FIX ME
class BartTextSummaryProcessor(BaseProcessor):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_processor()  # type: ignore[no-untyped-call] # FIX ME

    def init_processor(self):  # type: ignore[no-untyped-def] # FIX ME
        self.model = InferenceApi(
            token=os.environ["HF_TOKEN"], repo_id="facebook/bart-large-cnn"
        )
        self.messenger = BaseMessenger("bart_text_summ_messenger")  # type: ignore[no-untyped-call] # FIX ME
        return

    def update_info(self, feedback: str):  # type: ignore[no-untyped-def] # FIX ME
        self.messenger.add_assistant_message(feedback)

    def ask_info(  # type: ignore[override] # FIX ME
        self,
        context: str = None,  # type: ignore[assignment] # FIX ME
        *args,
        **kwargs,
    ) -> str:

        if self.messenger.check_iter_round_num() == 0:  # type: ignore[no-untyped-call] # FIX ME
            self.messenger.add_user_message(context)

        response = self.model(self.messenger.get_messages())  # type: ignore[no-untyped-call] # FIX ME
        summary = response[0]["summary_text"]
        return summary  # type: ignore[no-any-return] # FIX ME


if __name__ == "__main__":
    processor = BaseProcessor("bart_text_summ_processor")  # type: ignore[no-untyped-call] # FIX ME
    image_path = "../ctmai-test1.png"
    text: str = (
        "In a shocking turn of events, Hugging Face has released a new version of Transformers "
        "that brings several enhancements and bug fixes. Users are thrilled with the improvements "
        "and are finding the new version to be significantly better than the previous one. "
        "The Hugging Face team is thankful for the community's support and continues to work "
        "towards making the library the best it can be."
    )
    summary: str = processor.ask_info(  # type: ignore[no-untyped-call] # FIX ME
        query=None, context=text, image_path=image_path
    )
    print(summary)
