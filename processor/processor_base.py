import base64

class BaseProcessor(object):
    _processor_registry = {}

    @classmethod
    def register_processor(cls, processor_name):
        def decorator(subclass):
            cls._processor_registry[processor_name] = subclass
            return subclass
        return decorator

    def __new__(cls, processor_name, *args, **kwargs):
        if processor_name not in cls._processor_registry:
            raise ValueError(f"No processor registered with name '{processor_name}'")
        return super(BaseProcessor, cls).__new__(cls._processor_registry[processor_name])

    def set_model(self):
        raise NotImplementedError("The 'set_model' method must be implemented in derived classes.")

    @staticmethod
    def process_image(image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def process_audio(audio_path):
        return None
    
    @staticmethod
    def process_video(video_path):
        return None

    def ask(self, query, image_path):
        gist = self.ask_info(query, image_path)
        #gist = 'she appears to be in mid-sentence with an expression that could be interpreted as frustration or emphasis, often associated with making a strong point or expressing a complaint. The eyebrows are raised, and the mouth is open mid-speech, which emphasizes the expressive nature of the moment.'
        #gist = 'The images appear to be from a sitcom setting, likely taken from a scene occurring in a coffee shop or café, which is a common setting for many television shows. The background is a bit blurred, but we can see other patrons and a bar with stools, which suggests a casual, social environment. The woman\'s attire and the décor suggest a contemporary, possibly urban setting. This kind of venue is often used in television shows to create a relaxed atmosphere where characters can have conversations and interact in a more personal and informal way.'
        #gist = 'her expression intensifies to one that could be interpreted as either frustration or animated emphasis, as she appears to be speaking passionately or vehemently about something. Her eyebrows are drawn together and upwards, her eyes are slightly narrowed, and her mouth is open wider, all of which contribute to an expression that suggests a strong emotional response.'
        #gist = 'The audio is said in neutral tone.'
        #gist = 'When someone says, "So now I have to go so he\'ll think that I\'m totally ok with seeing him!", they are likely expressing a sense of obligation or pressure to behave in a particular way, despite their true feelings or desires. This statement suggests that the person feels the need to appear unaffected, indifferent, or even positive about the prospect of meeting or seeing someone else, possibly after a conflict, breakup, or other emotionally charged event. The use of "have to" implies that the person doesn\'t genuinely want to go but believes it\'s necessary to maintain a certain appearance or facade. This could be for various reasons, such as wanting to avoid drama, not wanting to show vulnerability, or trying to make a specific impression on the other person or on others who might be aware of the situation.'
        #gist = 'Taking into account the additional context provided, the woman\'s facial expression seems to capture a mix of emotions, including reluctance, exasperation, and resignation. Her grimace and furrowed brow can now be interpreted as a non-verbal expression of her internal struggle with having to put on a facade. The forced smile and exposed teeth may not indicate genuine happiness but rather an awkward attempt to seem okay with the situation she is describing. This aligns with the sentiment of feeling obligated to maintain appearances, despite contrary personal feelings.'
        #gist = 'Taking into account the new information provided, it seems that the woman in the image could be expressing frustration or annoyance about a situation involving a social obligation, potentially with an ex-partner or someone she has a complicated relationship with. The scene being in a coffee house—a common meeting place in many sitcoms for friends to discuss personal issues—suggests she may be confiding in her friends or seeking advice on how to handle the situation. Her expression, which comes across as exasperated, further hints that she is venting about the pretense she feels compelled to maintain in an upcoming encounter. This setting is typically used as a comedic backdrop where characters share their relationship woes and the paradoxes of their social lives, often leading to humorous or exaggerated exchanges.'
        gist = 'Given the context of the statement, it appears the woman is portraying a feeling of forced composure or a strained attempt at showing she is fine when she is not. Her facial expression conveys a mix of irritation and the stress of having to put on a facade. She seems to be in a situation where she is preparing to interact with someone under pretenses that do not align with her true emotions.'
        score = self.ask_score(query, gist, verbose=True)
        print(score)
        import pdb; pdb.set_trace()
        return gist, score

    def ask_relevance(self, query: str, gist: str) -> float:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "user", "content": "How related is the information ({}) with the query ({})? Answer with a number from 0 to 5 and do not add any other thing.".format(gist, query)},
                    ],
                    max_tokens=50,
                )
                score = int(response.choices[0].message.content.strip()) / 5
                return score
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                else:
                    print("Max attempts reached. Returning default score.")
        return 0

    def ask_confidence(self, query: str, gist: str) -> float:
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "user", "content": "How confidence do you think the information ({}) is a mustk? Answer with a number from 0 to 5 and do not add any other thing.".format(gist, query)},
                    ],
                    max_tokens=50,
                )
                score = int(response.choices[0].message.content.strip()) / 5
                return score
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                else:
                    print("Max attempts reached. Returning default score.")
        return 0

    def ask_surprise(self, query: str, gist: str, history_gists: str = None) -> float:
        # TODO: need to add cached history data
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4-0125-preview",
                    messages=[
                        {"role": "user", "content": "How surprise do you think the information ({}) is as an output of the processor? Answer with a number from 0 to 5 and do not add any other thing.".format(gist, query)},
                    ],
                    max_tokens=50,
                )
                score = int(response.choices[0].message.content.strip()) / 5
                return score
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_attempts - 1:
                    print("Retrying...")
                else:
                    print("Max attempts reached. Returning default score.")
        return 0

    def ask_score(self, query, gist, verbose=False, *args, **kwargs):
        relevance = self.ask_relevance(query, gist, *args, **kwargs) 
        confidence = self.ask_confidence(query, gist, *args, **kwargs)
        surprise = self.ask_surprise(query, gist, *args, **kwargs)
        if verbose:
            print(f"Relevance: {relevance}, Confidence: {confidence}, Surprise: {surprise}")
        return relevance * confidence * surprise
    
    def ask_info(self, query, image_path, *args, **kwargs):
        raise NotImplementedError("The 'ask_information' method must be implemented in derived classes.")