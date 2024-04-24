from typing import Any, Dict, List, Optional, Union

from openai import OpenAI

from ctm.messengers.messenger_base import BaseMessenger
from ctm.processors.processor_base import BaseProcessor
from ctm.utils.decorator import info_exponential_backoff


# Ensure that BaseProcessor has a properly typed register_processor method:
@BaseProcessor.register_processor("gpt4v_processor")
class GPT4VProcessor(BaseProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # Properly initialize the base class
        self.init_processor()
        self.task_instruction: Optional[str] = None

    def init_processor(self) -> None:
        self.model = OpenAI()
        self.messenger = BaseMessenger("gpt4v_messenger")

    def process(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {}  # Return an empty dict or a meaningful response as required

    def update_info(self, feedback: str) -> None:
        self.messenger.add_assistant_message(feedback)

    @info_exponential_backoff(retries=5, base_wait_time=1)
    def gpt4v_request(self) -> str | Any:
        response = self.model.completions.create(
            model="gpt-4-vision-preview",
            messages=self.messenger.get_messages(),
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return description

    def ask_info(
        self,
        query: str,
        text: Optional[str] = None,
        image: Optional[str] = None,
        video_frames: Optional[str] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if self.messenger.check_iter_round_num() == 0:
            messages: List[Dict[str, Union[str, Dict[str, str]]]] = [
                {
                    "type": "text",
                    "text": self.task_instruction
                    or "No instruction provided.",
                },
            ]
            if image:
                messages.append(
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image}",
                    }
                )
            self.messenger.add_user_message(messages)

        description = self.gpt4v_request()
        return description


if __name__ == "__main__":
    processor = GPT4VProcessor()
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(
        query="Describe the image.", image=image_path
    )
    print(summary)
