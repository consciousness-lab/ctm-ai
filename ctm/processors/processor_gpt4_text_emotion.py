from typing import Any

from ctm.processors.processor_gpt4 import GPT4Processor


# Assuming GPT4Processor has a properly typed `register_processor` method
@GPT4Processor.register_processor("gpt4_text_emotion_processor")
class GPT4TextEmotionProcessor(GPT4Processor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)  # Call to parent class constructor

    def init_task_info(self) -> None:
        self.task_instruction = "You are a text emotion classifier. You can understand the emotion within the text and generate the emotion label. If there is no text detected, please answer with None."


if __name__ == "__main__":
    # Instantiate the specific subclass for text emotion processing
    processor = GPT4TextEmotionProcessor()
    text = "I am feeling great today! The sun is shining and I've got a lot of work done."
    summary: str = processor.ask_info(query="Identify the emotion.", text=text)
    print(summary)
