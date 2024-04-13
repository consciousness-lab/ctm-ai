from ctm.processors.processor_gpt4 import GPT4Processor


@GPT4Processor.register_processor("gpt4_speaker_intent_processor")  # type: ignore[no-untyped-call] # FIX ME
class GPT4SpeakerIntentProcessor(GPT4Processor):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_processor()  # type: ignore[no-untyped-call] # FIX ME
        self.task_instruction = "You are a speaker intent predictor. You can understand the intent of the speaker and describe what is the speaker's intent for saying that. If there is no speaker detected, please answer with None."


if __name__ == "__main__":
    processor = GPT4Processor("close_fashion_processor")  # type: ignore[no-untyped-call] # FIX ME
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)  # type: ignore[arg-type] # FIX ME
    print(summary)
