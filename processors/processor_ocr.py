from .processor_base import BaseProcessor
from openai import OpenAI


@BaseProcessor.register_processor('ocr_processor')
class OCRProcessor(BaseProcessor):

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
        response = self.client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "You should act like an OCR model. Please extract the text from the image. If there is no text detected, please answer with None."},
                    {"type": "image_url", "image_url": f"data:image/jpeg;base64,{image}"},
                ],
            }],
            max_tokens=300,
        )
        description = response.choices[0].message.content
        return description


if __name__ == "__main__":
    processor = BaseProcessor('ocr_processor')
    image_path = '../ctmai-test1.png'
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
