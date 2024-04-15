import sys

sys.path.append("..")
from ctm.ctms.ctm_base import (
    BaseConsciousnessTuringMachine,  # type: ignore[import] # FIX ME
)
from ctm.utils import load_image

if __name__ == "__main__":
    ctm = BaseConsciousnessTuringMachine("sarcasm_ctm")
    query = "Is the person saying sarcasm or not?"
    text = "You have no idea what you are talking about!"
    image_path = "../images/sarcasm_example1.png"
    image = load_image(image_path)
    answer = ctm(
        query=query,
        text=text,
        image=image,
    )
    print(answer)
