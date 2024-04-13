from processors.processor_gpt4 import GPT4Processor


@GPT4Processor.register_processor("gpt4_text_summary_processor")  # type: ignore[no-untyped-call] # FIX ME
class GPT4TextSummaryProcessor(GPT4Processor):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_processor()  # type: ignore[no-untyped-call] # FIX ME
        self.task_instruction = "You are a text summarizer. You can understand the meaning of the text and generate the summary."


if __name__ == "__main__":
    processor = GPT4Processor("close_fashion_processor")  # type: ignore[no-untyped-call] # FIX ME
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)  # type: ignore[arg-type] # FIX ME
    print(summary)
