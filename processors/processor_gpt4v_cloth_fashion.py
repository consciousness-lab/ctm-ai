from processors.processor_gpt4v import GPT4VProcessor


@GPT4VProcessor.register_processor("gpt4v_cloth_fashion_processor")
class GPT4VClothFashionProcessor(GPT4VProcessor):
    def __init__(self, *args, **kwargs):
        self.init_processor()
        self.task_instruction = "Focus on the cloth of people in the image, describe the style of the cloth fashion. If there is no people detected, please answer with None."


if __name__ == "__main__":
    processor = GPT4VProcessor("close_fashion_processor")
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
