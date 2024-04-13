import sys

sys.path.append("..")
from ctm.ctms.ctm_base import (
    BaseConsciousnessTuringMachine,  # type: ignore[import] # FIX ME
)

if __name__ == "__main__":
    ctm = BaseConsciousnessTuringMachine("sarcasm_ctm")
    question = "Is the person saying sarcasm or not?"
    image_path = "../images/sarcasm_example1.png"
    ctm = BaseConsciousnessTuringMachine("sarcasm_ctm")
    question = "Is the person saying sarcasm or not?"
    image_path = "../images/sarcasm_example1.png"
    answer = ctm(question=question, image_path=image_path)
    print(answer)
