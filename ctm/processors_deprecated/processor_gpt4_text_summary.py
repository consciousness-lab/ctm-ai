from typing import Any

from ctm.processors_deprecated.processor_gpt4 import GPT4Processor


# Assuming GPT4Processor has a properly typed `register_processor` method
@GPT4Processor.register_processor("gpt4_text_summary_processor")
class GPT4TextSummaryProcessor(GPT4Processor):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(
            *args, **kwargs
        )  # Properly initialize the parent class

    def init_task_info(self) -> None:
        self.task_instruction = "You are a text summarizer. You can understand the meaning of the text and generate the summary."


if __name__ == "__main__":
    # Instantiate the specific subclass for the text summarization task
    processor = GPT4TextSummaryProcessor()
    text = "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet."
    summary: str = processor.ask_info(query="Summarize the text.", text=text)
    print(summary)
