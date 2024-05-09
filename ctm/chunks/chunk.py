class Chunk(object):
    def __init__(
        self,
        processor_name: str = None,
        timestep: int = -1,
        gist: str = None,
        relevance: float = None,
        confidence: float = None,
        surprise: float = None,
        weight: float = None,
        intensity: float = None,
        mood: float = None,
    ):
        self.processor_name: str = processor_name
        self.timestep: int = timestep
        self.gist: str = gist
        self.relevance: float = relevance
        self.confidence: float = confidence
        self.surprise: float = surprise
        self.weight: float = weight
        self.intensity: float = intensity
        self.mood: float = mood
