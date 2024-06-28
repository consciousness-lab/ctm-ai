import sys

from ctm_ai.ctms.ctm_base import ConsciousnessTuringMachine
from ctm_ai.utils import load_image

sys.path.append('..')

if __name__ == '__main__':
    ctm = ConsciousnessTuringMachine('sarcasm_ctm')
    query = 'Is the person saying sarcasm or not?'
    text = 'You have no idea what you are talking about!'
    image_path = '../assets/sarcasm_example1.png'
    image = load_image(image_path)
    answer = ctm(
        query=query,
        text=text,
        image=image,
    )
    print(answer)
