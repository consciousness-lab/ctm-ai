from processors.processor_base import BaseProcessor
from messengers.messenger_base import BaseMessenger
from openai import OpenAI


@BaseProcessor.register_processor('gpt4v_processor')
class GPT4VProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()
        self.task_instruction = None

    def init_processor(self):
        self.model = OpenAI()
        self.messenger = BaseMessenger('gpt4v_messenger')
        return

    def process(self, payload: dict) -> dict:
        return
    
    def update_info(self, feedback: str):
        self.messenger.add_assistant_message(feedback)

    def ask_info(self, query: str, image_path: str = None) -> str:
        if self.messenger.check_iter_round_num() == 0:
            image = self.process_image(image_path)
            self.messenger.add_user_message(
                [
                    {"type": "text", "text": self.task_instruction},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"},
                ]
            )

        response = self.model.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=self.messenger.get_messages(),
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return description


if __name__ == "__main__":
    processor = BaseProcessor('ocr_processor')
    image_path = '../ctmai-test1.png'
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
