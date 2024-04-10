import sys
sys.path.append('..')
from ctms.ctm_base import BaseConsciousnessTuringMachine


if __name__ == "__main__":
    ctm = BaseConsciousnessTuringMachine('sarcasm_ctm')
    ctm.craft()

    answer_threshold = 0.5
    max_iter = 3
    question = 'Is the person saying sarcasm or not?'
    image_path = '../images/sarcasm_example1.png'
    
    answer = ctm(question=question, image_path=image_path)
    print(answer)

    import pdb; pdb.set_trace()