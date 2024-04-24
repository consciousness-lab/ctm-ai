from typing import Any

from ctm.processors.processor_gpt4v import GPT4VProcessor


# Assuming GPT4VProcessor has a properly typed `register_processor` method
@GPT4VProcessor.register_processor("gpt4v_cloth_fashion_processor")
class GPT4VClothFashionProcessor(GPT4VProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # Call to parent class constructor
        self.task_instruction = "Focus on the cloth of people in the image, describe the style of the cloth fashion. If there is no people detected, please answer with None."


if __name__ == "__main__":
    # Instantiate the specific subclass for the cloth fashion task
    processor = GPT4VClothFashionProcessor()
    image_path = "../ctmai-test1.png"
    # Providing a valid query and ensuring `ask_info` is correctly implemented in the base class
    summary: str = processor.ask_info(
        query="Describe the fashion style", image=image_path
    )
    print(summary)
