from typing import Any, Dict, Tuple, Type

from ..chunks import Chunk


class BaseFuser(object):
    _fuser_registry: Dict[str, Type['BaseFuser']] = {}

    @classmethod
    def register_fuser(cls, fuser_name: str) -> Any:
        def decorator(
            subclass: Type['BaseFuser'],
        ) -> Type['BaseFuser']:
            cls._fuser_registry[fuser_name] = subclass
            return subclass

        return decorator

    def __new__(cls, fuser_name: str, *args: Any, **kwargs: Any) -> Any:
        if fuser_name not in cls._fuser_registry:
            raise ValueError(f"No fuser registered with name '{fuser_name}'")
        return super(BaseFuser, cls).__new__(cls._fuser_registry[fuser_name])

    def init_fuser(
        self,
    ) -> None:
        raise NotImplementedError(
            "The 'set_model' method must be implemented in derived classes."
        )

    def fuse(self, chunk1: Chunk, chunk2: Chunk) -> Tuple[str, float]:
        gist1, gist2 = chunk1.gist, chunk2.gist
        gist = self.fuse_info(gist1, gist2)
        relevance, confidence, surprise = self.fuse_score(gist1, gist2, verbose=True)
        weight = relevance * confidence * surprise
        chunk = Chunk(
            processor_name='{}_{}_fuse'.format(
                chunk1.processor_name, chunk2.processor_name
            ),
            gist=gist,
            time_step=max(chunk1.time_step, chunk2.time_step) + 1,
            relevance=relevance,
            confidence=confidence,
            surprise=surprise,
            weight=weight,
            intensity=weight,
            mood=weight,
        )
        return chunk

    def fuse_info(self, chunk1: Chunk, chunk2: Chunk) -> str:
        raise NotImplementedError(
            "The 'fuse_info' method must be implemented in derived classes."
        )

    def fuse_score(self, chunk1: Chunk, chunk2: Chunk, verbose: bool = False) -> float:
        raise NotImplementedError(
            "The 'fuse_score' method must be implemented in derived classes."
        )
