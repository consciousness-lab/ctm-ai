import sys

from ctm_ai.ctms.ctm import ConsciousTuringMachine
from ctm_ai.utils import load_image

sys.path.append('..')

if __name__ == '__main__':
    ctm = ConsciousTuringMachine('sarcasm_ctm')
    query = 'Find relevant information about what did Paul Liang posts, Paul is a assistant professor at MIT.'
    text = 'Paul Liang is a assistant professor at MIT.'
    answer = ctm(
        query=query,
        text=text,
    )
    print(answer)
