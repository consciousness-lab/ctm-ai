from ctm_base import (
    BaseConsciousnessTuringMachine,  # type: ignore[import] # FIX ME
)

if __name__ == "__main__":
    ctm = BaseConsciousnessTuringMachine("whatname_ctm")
    ctm.add_processor("scene_location_processor", group_name="group_1")
    ctm.add_processor("cloth_fashion_processor", group_name="group_1")
    ctm.add_processor("ocr_processor", group_name="group_2")
    ctm.add_answer_generator("whatname_answer_generation_processor")

    answer_threshold = 0.5
    max_iter = 3
    question = "what is the name of its professor?"
    image_path = "./ctmai-test1.png"

    answer = ctm(question=question, image_path=image_path)
    print(answer)

    import pdb

    pdb.set_trace()
