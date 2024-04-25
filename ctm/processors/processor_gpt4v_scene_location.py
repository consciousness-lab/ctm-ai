from typing import Any

from ctm.processors.processor_gpt4v import GPT4VProcessor


# Assuming GPT4VProcessor has a properly typed `register_processor` method:
@GPT4VProcessor.register_processor("gpt4v_scene_location_processor")
class GPT4VSceneLocationProcessor(GPT4VProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # Initialize the parent processor

    def init_task_info(self):
        self.task_instruction = "Besides the main activity in the image, can you describe the potential location or the event that is going on within this picture?"


if __name__ == "__main__":
    # Instantiate the specific subclass for the scene location task
    processor = GPT4VSceneLocationProcessor()
    image_path = "../ctmai-test1.png"
    # The `ask_info` method should also be corrected to include all necessary parameters properly typed.
    summary: str = processor.ask_info(
        query="Describe the scene and location.", image=image_path
    )
    print(summary)
