from typing import Any

from ctm.processors_deprecated.processor_gpt4v import GPT4VProcessor


# Assume register_processor method has been properly typed
@GPT4VProcessor.register_processor("gpt4v_face_emotion_processor")
class GPT4VFaceEmotionProcessor(GPT4VProcessor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args, **kwargs
        )  # Properly initialize the parent class

    def init_task_info(self) -> None:
        self.task_instruction = "Besides the main scene in the image, can you describe the face emotion that is on people's faces within this picture?"


if __name__ == "__main__":
    # Instantiate the specific subclass for face emotion processing
    processor = GPT4VFaceEmotionProcessor()
    image_path = "../ctmai-test1.png"
    # Providing a valid query and ensuring that the method ask_info accepts the correct parameters
    summary: str = processor.ask_info(
        query="Describe face emotions", image=image_path
    )
    print(summary)
