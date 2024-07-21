import sys

from ctm_ai.ctms.ctm import ConsciousnessTuringMachine
from ctm_ai.supervisors.supervisor_base import BaseSupervisor
from ctm_ai.utils import load_image

sys.path.append('..')

if __name__ == '__main__':
    ctm = ConsciousnessTuringMachine('sarcasm_ctm')
    supervisor = BaseSupervisor('gpt4_supervisor')
    query = 'Is the person saying sarcasm or not?'
    text = 'You have no idea what you are talking about!'
    label = 'yes'
    image_path = '../assets/sarcasm_example1.png'
    image = load_image(image_path)
    info = ctm(query=query, text=text, image=image)
    prediction, score = supervisor.ask(query, info)
    feedback = prediction == label
    ctm.backward(feedback)
    print(feedback)
