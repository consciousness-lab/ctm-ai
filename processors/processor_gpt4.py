from openai import OpenAI

from messengers.messenger_base import BaseMessenger
from processors.processor_base import BaseProcessor
from utils.exponential_backoff import exponential_backoff


@BaseProcessor.register_processor("gpt4_processor")  # type: ignore[no-untyped-call] # FIX ME
class GPT4Processor(BaseProcessor):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_processor()  # type: ignore[no-untyped-call] # FIX ME
        self.task_instruction = None

    def init_processor(self):  # type: ignore[no-untyped-def] # FIX ME
        self.model = OpenAI()
        self.messenger = BaseMessenger("gpt4_messenger")  # type: ignore[no-untyped-call] # FIX ME
        return

    def process(self, payload: dict) -> dict:  # type: ignore[type-arg] # FIX ME
        return  # type: ignore[return-value] # FIX ME

    def update_info(self, feedback: str):  # type: ignore[no-untyped-def] # FIX ME
        self.messenger.add_assistant_message(feedback)

    @exponential_backoff(retries=5, base_wait_time=1)  # type: ignore[no-untyped-call] # FIX ME
    def gpt4_requst(self):  # type: ignore[no-untyped-def] # FIX ME
        response = self.model.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=self.messenger.get_messages(),  # type: ignore[no-untyped-call] # FIX ME
            max_tokens=300,
        )
        return response

    def ask_info(  # type: ignore[override] # FIX ME
        self,
        query: str,
        context: str = None,  # type: ignore[assignment] # FIX ME
        image_path: str = None,  # type: ignore[assignment] # FIX ME
        audio_path: str = None,  # type: ignore[assignment] # FIX ME
        video_path: str = None,  # type: ignore[assignment] # FIX ME
    ) -> str:
        if self.messenger.check_iter_round_num() == 0:  # type: ignore[no-untyped-call] # FIX ME
            self.messenger.add_user_message(
                "The text information for the previously described task is as follows: "
                + context
                + "Here is what you should do: "
                + self.task_instruction  # type: ignore[operator] # FIX ME
            )

        response = self.gpt4_requst()
        description = response.choices[0].message.content
        return description  # type: ignore[no-any-return] # FIX ME


if __name__ == "__main__":
    processor = BaseProcessor("ocr_processor")  # type: ignore[no-untyped-call] # FIX ME
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)  # type: ignore[no-untyped-call] # FIX ME
    print(summary)
