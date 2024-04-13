from processors.processor_gpt4 import GPT4Processor


@GPT4Processor.register_processor("gpt4_speaker_intent_processor")
class GPT4SpeakerIntentProcessor(GPT4Processor):
    def __init__(self, *args, **kwargs):
        self.init_processor()
        self.task_instruction = "You are a speaker intent predictor. You can understand the intent of the speaker and describe what is the speaker's intent for saying that. If there is no speaker detected, please answer with None."


if __name__ == "__main__":
    processor = GPT4Processor("close_fashion_processor")
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
