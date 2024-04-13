import base64


class BaseSupervisor(object):
    _supervisor_registry = {}

    @classmethod
    def register_supervisor(cls, supervisor_name):
        def decorator(subclass):
            cls._supervisor_registry[supervisor_name] = subclass
            return subclass

        return decorator

    def __new__(cls, supervisor_name, *args, **kwargs):
        if supervisor_name not in cls._supervisor_registry:
            raise ValueError(
                f"No supervisor registered with name '{supervisor_name}'"
            )
        return super(BaseSupervisor, cls).__new__(
            cls._supervisor_registry[supervisor_name]
        )

    def set_model(self):
        raise NotImplementedError(
            "The 'set_model' method must be implemented in derived classes."
        )

    @staticmethod
    def process_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def process_audio(audio_path):
        return None

    @staticmethod
    def process_video(video_path):
        return None

    def ask(self, query, image_path):
        gist = self.ask_info(query, image_path)
        score = self.ask_score(query, gist, verbose=True)
        return gist, score

    def ask_info(self, query: str, context: str = None) -> str:
        return None

    def ask_score(self, query: str, gist: str, verbose: bool = False) -> float:
        return None
