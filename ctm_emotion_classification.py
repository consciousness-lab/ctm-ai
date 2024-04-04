from ctm_base import BaseConsciousnessTuringMachine

@BaseConsciousnessTuringMachine.register_ctm('EmotionCTM')
class EmotionConsciousnessTuringMachine(BaseConsciousnessTuringMachine):



if __name__ == "__main__":
    ctm = ConsciousnessTuringMachine()
    ctm.add_processor("gpt-3.5-turbo")
    ctm.add_processor("gpt-3.5-turbo")

    document1 = ['One man wearing white T-shirt is standing in a classroom', 'He is talking loudly about machine learning.']
    query = 'what is the name of its professor?'

    answers = ctm.ask_processors(query)
    print(answers)
