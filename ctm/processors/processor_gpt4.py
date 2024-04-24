from typing import Any, Dict, Optional, Union

from openai import OpenAI

from ctm.messengers.messenger_base import BaseMessenger
from ctm.processors.processor_base import BaseProcessor
from ctm.utils.decorator import info_exponential_backoff


# Assuming the `register_processor` method has been updated to be properly typed:
@BaseProcessor.register_processor("gpt4_processor")
class GPT4Processor(BaseProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.init_processor()
        self.init_messenger()

    def init_processor(self) -> None:
        self.model = OpenAI()

    def init_messenger(self) -> None:
        self.messenger = BaseMessenger("gpt4_messenger")

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # Assume process should do something and return a dictionary
        return {}

    def update_info(self, feedback: str) -> None:
        self.messenger.add_assistant_message(feedback)

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def gpt4_request(self) -> Any:
        response = self.model.chat_completions.create(
            model="gpt-4-turbo-preview",
            messages=self.messenger.get_messages(),
            max_tokens=300,
        )
        return response

    def ask_info(
        self, query: str, text: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> str:
        if self.messenger.check_iter_round_num() == 0:
            initial_message = "The text information for the previously described task is as follows: "
            initial_message += (
                text if text is not None else "No text provided."
            )
            initial_message += (
                " Here is what you should do: " + self.task_instruction
            )
            self.messenger.add_user_message(initial_message)

        response = self.gpt4_request()
        description = response["choices"][0]["message"]["content"]
        return description


if __name__ == "__main__":
    processor = GPT4Processor()
    text = "Hugging Face has released a new version of Transformers that brings several enhancements."
    summary: str = processor.ask_info(
        query="Summarize the changes.", text=text
    )
    print(summary)
