from .processor_base import BaseProcessor
from openai import OpenAI


@BaseProcessor.register_processor('cloth_fashion_processor')
class ClothFashionProcessor(BaseProcessor):

    def __init__(self, *args, **kwargs):
        self.init_processor()

    def init_processor(self):
        self.client = OpenAI()
        self.base_prompt = []
        return

    def process(self, payload: dict) -> dict:
        return
    

    def ask_info(self, query: str, image_path: str = None) -> str:
        image = self.process_image(image_path)
        prompt = self.base_prompt + [{"role": "user", "content": [{"type": "text", "text": "Focus on the cloth of people in the image, describe the style of the cloth fashion. If there is no people detected, please answer with None."},{"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"}]}]
        import pdb; pdb.set_trace()
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=prompt,
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return description


if __name__ == "__main__":
    processor = BaseProcessor('cloth_fashion_processor')
    image_path = '../ctmai-test1.png'
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
