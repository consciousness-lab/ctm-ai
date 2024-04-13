from processors.processor_gpt4v import GPT4VProcessor


@GPT4VProcessor.register_processor("gpt4v_ocr_processor")  # type: ignore[no-untyped-call] # FIX ME
class GPT4VOCRProcessor(GPT4VProcessor):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_processor()  # type: ignore[no-untyped-call] # FIX ME
        self.task_instruction = "You should act like an OCR model. Please extract the text from the image. If there is no text detected, please answer with None."


if __name__ == "__main__":
    processor = GPT4VProcessor("ocr_processor")  # type: ignore[no-untyped-call] # FIX ME
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)  # type: ignore[arg-type] # FIX ME
    print(summary)
