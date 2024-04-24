from typing import Any

from ctm.processors.processor_gpt4 import GPT4Processor


# Assuming GPT4Processor has a properly typed `register_processor` method
@GPT4Processor.register_processor("gpt4_speaker_intent_processor")
class GPT4SpeakerIntentProcessor(GPT4Processor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args, **kwargs
        )  # Ensure the parent constructor is called properly
        self.task_instruction = "You are a speaker intent predictor. You can understand the intent of the speaker and describe what is the speaker's intent for saying that. If there is no speaker detected, please answer with None."


if __name__ == "__main__":
    # Instantiate the specific subclass for speaker intent processing
    processor = GPT4SpeakerIntentProcessor()
    text = "I can't wait to see the results of the new project. We've put so much effort into it!"
    summary: str = processor.ask_info(
        query="What is the intent behind the speaker's statement?", text=text
    )
    print(summary)
