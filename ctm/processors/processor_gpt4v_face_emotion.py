from ctm.processors.processor_gpt4v import GPT4VProcessor


@GPT4VProcessor.register_processor("gpt4v_face_emotion_processor")  # type: ignore[no-untyped-call] # FIX ME
class GPT4VFaceEmotionProcessor(GPT4VProcessor):
    def __init__(self, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        self.init_processor()  # type: ignore[no-untyped-call] # FIX ME
        self.task_instruction = "Besides the main scene in the image, can you describe the face emotion that is on people's faces within this picture?"


if __name__ == "__main__":
    processor = GPT4VProcessor("face_emotion_processor")  # type: ignore[no-untyped-call] # FIX ME
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)  # type: ignore[arg-type] # FIX ME
    print(summary)
