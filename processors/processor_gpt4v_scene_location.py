from processors.processor_gpt4v import GPT4VProcessor


@GPT4VProcessor.register_processor("gpt4v_scene_location_processor")
class GPT4VSceneLocationProcessor(GPT4VProcessor):
    def __init__(self, *args, **kwargs):
        self.init_processor()
        self.task_instruction = "Besides the main activity in the image, can you describe the potential location or the event that is going on within this picture?"


if __name__ == "__main__":
    processor = GPT4VProcessor("scene_location_processor")
    image_path = "../ctmai-test1.png"
    summary: str = processor.ask_info(query=None, image_path=image_path)
    print(summary)
