from typing import Any

from ctm.processors_deprecated.processor_gpt4v import GPT4VProcessor


# Assuming the GPT4VProcessor has a properly typed `register_processor` method:
@GPT4VProcessor.register_processor("gpt4v_posture_processor")
class GPT4VPostureProcessor(GPT4VProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args, **kwargs
        )  # Properly call the parent's constructor

    def init_task_info(self) -> None:
        self.task_instruction = "Besides the main scene in the image, can you describe the posture that is going on within this picture?"


if __name__ == "__main__":
    # Instantiate the specific subclass for the posture analysis task
    processor = GPT4VPostureProcessor()
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(
        query="Analyze the posture.", image=image_path
    )
    print(summary)
