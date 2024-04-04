from .processor_text_bert_qa import TextBERTQAProcessor
from .processor_text_bert_sst import TextBERTSSTProcessor
from .processor_text_bart_summ import TextBARTSummProcessor
from .processor_scene_location import SceneLocationProcessor
from .processor_ocr import OCRProcessor
from .processor_cloth_fashion import ClothFashionProcessor
from .processor_whatname_answer_generation import WhatNameAnswerGenerationProcessor

__all__ = [
    'BaseProcessor',
    'TextBERTQAProcessor', 
    'TextBERTSSTProcessor', 
    'TextBARTSummProcessor', 
    'SceneLocationProcessor',
    'OCRProcessor',
    'ClothFashionProcessor',
    'WhatNameAnswerGenerationProcessor',
]