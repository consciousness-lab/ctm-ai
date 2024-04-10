from ctms.ctm_base import BaseConsciousnessTuringMachine

@BaseConsciousnessTuringMachine.register_ctm('sarcasm_ctm')
class SarcasmConsciousnessTuringMachine(BaseConsciousnessTuringMachine):
    def __init__(self, ctm_name, *args, **kwargs):
        super().__init__(ctm_name, *args, **kwargs)
    
    def craft(self):
        self.add_processor("gpt4v_scene_location_processor", group_name="group_1")
        self.add_processor("gpt4v_cloth_fashion_processor", group_name="group_1")
        self.add_processor("gpt4v_posture_processor", group_name="group_2")
        self.add_processor("gpt4v_ocr_processor", group_name="group_3")
        self.add_supervisor("gpt4_supervisor")
