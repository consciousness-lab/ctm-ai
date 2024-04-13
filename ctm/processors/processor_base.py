import base64

from ctm.utils.exponential_backoff import exponential_backoff


class BaseProcessor(object):
    _processor_registry = {}  # type: ignore[var-annotated] # FIX ME

    @classmethod
    def register_processor(cls, processor_name):  # type: ignore[no-untyped-def] # FIX ME
        def decorator(subclass):  # type: ignore[no-untyped-def] # FIX ME
            cls._processor_registry[processor_name] = subclass
            return subclass

        return decorator

    def __new__(cls, processor_name, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        if processor_name not in cls._processor_registry:
            raise ValueError(
                f"No processor registered with name '{processor_name}'"
            )
        return super(BaseProcessor, cls).__new__(
            cls._processor_registry[processor_name]
        )

    def set_model(self):  # type: ignore[no-untyped-def] # FIX ME
        raise NotImplementedError(
            "The 'set_model' method must be implemented in derived classes."
        )

    @staticmethod
    def process_image(image_path):  # type: ignore[no-untyped-def] # FIX ME
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    @staticmethod
    def process_audio(audio_path):  # type: ignore[no-untyped-def] # FIX ME
        return None

    @staticmethod
    def process_video(video_path):  # type: ignore[no-untyped-def] # FIX ME
        return None

    def ask(self, query, context, image_path, audio_path, video_path):  # type: ignore[no-untyped-def] # FIX ME
        gist = self.ask_info(  # type: ignore[no-untyped-call] # FIX ME
            query, context, image_path, audio_path, video_path
        )
        score = self.ask_score(query, gist, verbose=True)  # type: ignore[no-untyped-call] # FIX ME
        return gist, score

    @exponential_backoff(retries=5, base_wait_time=1)  # type: ignore[misc, no-untyped-call] # FIX ME
    def ask_relevance(self, query: str, gist: str) -> float:
        response = self.model.chat.completions.create(  # type: ignore[attr-defined] # FIX ME
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": "How related is the information ({}) with the query ({})? Answer with a number from 0 to 5 and do not add any other thing.".format(
                        gist, query
                    ),
                },
            ],
            max_tokens=50,
        )
        score = int(response.choices[0].message.content.strip()) / 5
        return score

    @exponential_backoff(retries=5, base_wait_time=1)  # type: ignore[misc, no-untyped-call] # FIX ME
    def ask_confidence(self, query: str, gist: str) -> float:
        response = self.model.chat.completions.create(  # type: ignore[attr-defined] # FIX ME
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": "How confidence do you think the information ({}) is a mustk? Answer with a number from 0 to 5 and do not add any other thing.".format(  # type: ignore[str-format] # FIX ME
                        gist, query
                    ),
                },
            ],
            max_tokens=50,
        )
        score = int(response.choices[0].message.content.strip()) / 5
        return score

    @exponential_backoff(retries=5, base_wait_time=1)  # type: ignore[misc, no-untyped-call] # FIX ME
    def ask_surprise(
        self, query: str, gist: str, history_gists: str = None  # type: ignore[assignment] # FIX ME
    ) -> float:
        response = self.model.chat.completions.create(  # type: ignore[attr-defined] # FIX ME
            model="gpt-4-0125-preview",
            messages=[
                {
                    "role": "user",
                    "content": "How surprise do you think the information ({}) is as an output of the processor? Answer with a number from 0 to 5 and do not add any other thing.".format(  # type: ignore[str-format] # FIX ME
                        gist, query
                    ),
                },
            ],
            max_tokens=50,
        )
        score = int(response.choices[0].message.content.strip()) / 5
        return score

    def ask_score(self, query, gist, verbose=False, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        relevance = self.ask_relevance(query, gist, *args, **kwargs)
        confidence = self.ask_confidence(query, gist, *args, **kwargs)
        surprise = self.ask_surprise(query, gist, *args, **kwargs)
        if verbose:
            print(
                f"Relevance: {relevance}, Confidence: {confidence}, Surprise: {surprise}"
            )
        return relevance * confidence * surprise

    def ask_info(self, query, image_path, *args, **kwargs):  # type: ignore[no-untyped-def] # FIX ME
        raise NotImplementedError(
            "The 'ask_information' method must be implemented in derived classes."
        )
