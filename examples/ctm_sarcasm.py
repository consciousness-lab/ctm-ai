import sys
sys.path.append('..')
from ctm.ctm_base import BaseConsciousnessTuringMachine

@BaseConsciousnessTuringMachine.register_ctm('sarcasm_ctm')
class SarcasmConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(self, ctm_name, *args, **kwargs):
        super().__init__(ctm_name, *args, **kwargs)


if __name__ == "__main__":
    ctm = BaseConsciousnessTuringMachine('sarcasm_ctm')
    ctm.add_processor("gpt4v_scene_location_processor", group_name="group_1")
    ctm.add_processor("gpt4v_cloth_fashion_processor", group_name="group_1")
    ctm.add_processor("gpt4v_posture_processor", group_name="group_2")
    ctm.add_processor("gpt4v_ocr_processor", group_name="group_3")
    ctm.add_answer_generator("whatname_answer_generation_processor")

    answer_threshold = 0.5
    max_iter = 3
    question = 'Is the person saying sarcasm or not?'
    image_path = '../images/sarcasm_example1.png'
    
    answer = ctm(question=question, image_path=image_path)
    print(answer)

    import pdb; pdb.set_trace()