from typing import Any

from ctm.processors_deprecated.processor_gpt4v import GPT4VProcessor


# Correct the registration method to include type checking if possible in the GPT4VProcessor class
@GPT4VProcessor.register_processor("gpt4v_ocr_processor")
class GPT4VOCRProcessor(GPT4VProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args, **kwargs
        )  # Ensure the parent constructor is called properly

    def init_task_info(self) -> None:
        self.task_instruction = "You should act like an OCR model. Please extract the text from the image. If there is no text detected, please answer with None."


if __name__ == "__main__":
    # Ensure that we're instantiating the correct processor for the job
    processor = GPT4VOCRProcessor()
    image_path = "../ctmai-test1.png"
    # Provide a valid query string; ensure `ask_info` can handle all provided parameters
    summary: str = processor.ask_info(query="Extract text", image=image_path)
    print(summary)
