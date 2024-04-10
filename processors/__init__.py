from .processor_text_bert_qa import TextBERTQAProcessor
from .processor_text_bert_sst import TextBERTSSTProcessor
from .processor_text_bart_summ import TextBARTSummProcessor
from .processor_gpt4v_scene_location import GPT4VSceneLocationProcessor
from .processor_gpt4v_ocr import GPT4VOCRProcessor
from .processor_gpt4v_cloth_fashion import GPT4VClothFashionProcessor
from .processor_gpt4v_face_emotion import GPT4VFaceEmotionProcessor
from .processor_gpt4v_posture import GPT4VPostureProcessor
from .processor_whatname_answer_generation import WhatNameAnswerGenerationProcessor

__all__ = [
    'BaseProcessor',
    'TextBERTQAProcessor', 
    'TextBERTSSTProcessor', 
    'TextBARTSummProcessor', 
    'GPT4VProcessor',
    'GPT4VSceneLocationProcessor',
    'GPT4VOCRProcessor',
    'GPT4VClothFashionProcessor',
    'GPT4VFaceEmotionProcessor',
    'GPT4VPostureProcessor',
    'WhatNameAnswerGenerationProcessor',
]