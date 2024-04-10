from .processor_base import BaseProcessor
from .processor_messenger_base import BaseProcessorMessenger
from openai import OpenAI
from typing import Union, List, Dict


@BaseProcessorMessenger.register_messenger('scene_location_processor_messenger')
class SceneLocationProcessorMessenger(BaseProcessorMessenger):
    def __init__(self, role = None, content = None, *args, **kwargs):
        self.init_messenger(role, content)

    def init_messenger(self, role: str = None, content: Union[str, Dict, List] = None):
        self.messages = []
        if content and role:
            self.update_messages(role, content)

    def update_message(self, role: str, content: Union[str, Dict, List]):
        self.messages.append({
            "role": role,
            "content": content
        })

    def check_iter_round_num(self):
        return len(self.messages)


@BaseProcessor.register_processor('scene_location_processor')
class SceneLocationProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.client = OpenAI()
        self.messenger = BaseProcessorMessenger('scene_location_processor_messenger')
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
                    {"type": "text", "text": "Besides the main activity in the image, can you describe the potential location or the event that is going on within this picture?"},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"},
                ]
            )

        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=self.messenger.get_messages(),
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return description


if __name__ == "__main__":
    processor = BaseProcessor('scene_location_processor')
    image_path = '../ctmai-test1.png'
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
