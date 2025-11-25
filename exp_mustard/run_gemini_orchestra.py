import json
import sys
import time

from exp_baselines import GeminiMultimodalLLM

sys.path.append('..')

# Prompts for different roles
PLANNER_PROMPT = (
    "You are a Sarcasm Investigation Planner.\n"
    "Your task is NOT to decide if it is sarcasm yet, but to create a plan to find out.\n"
    "Analyze the input briefly and generate 3 specific, targeted questions that an investigator should answer to determine if the punchline is sarcastic.\n"
    "Focus on contrasting modalities (e.g., 'Does the facial expression match the positive words?').\n"
    "Output ONLY the 3 questions, numbered 1, 2, and 3."
)

EXECUTOR_PROMPT = (
    "You are a Sarcasm Investigator.\n"
    "You have been given 3 specific questions to answer based on the multimodal inputs.\n"
    "Please answer each question in detail based on what you see in the video, hear in the audio, and read in the text.\n"
    "Questions to answer:\n{plan}\n"
)

SYNTHESIZER_PROMPT = (
    "You are a Sarcasm Synthesizer.\n"
    "Review the Investigator's observations regarding the punchline.\n"
    "Investigator's Report:\n{observations}\n\n"
    "Based on these findings, determine if the punchline is sarcastic.\n"
    "Your final answer must start with 'Yes' or 'No', followed by your reasoning."
)


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file='baseline_orchestra.jsonl'):
    dataset = load_data('./mustard_dataset_test_sample_100.json')
    target_sentence = dataset[test_file]['utterance']
    
    text_list = dataset[test_file]['context']
    text_list.append(target_sentence)
    fullContext = ''
    for i in range(len(text_list)):
        currentUtterance = f'{text_list[i]} \n'
        fullContext += currentUtterance
    audio_path = f'mustard_audios/{test_file}_audio.mp4'
    video_frames_path = f'mustard_frames/{test_file}_frames'
    
    # Helper to create LLM instance
    def create_llm(query):
        return GeminiMultimodalLLM(
            file_name=test_file,
            image_frames_folder=video_frames_path,
            audio_file_path=audio_path,
            context=fullContext,
            query=query,
            model_name='gemini-2.0-flash-lite',
        )

    # Phase 1: Planning
    print(f"--- Orchestra Phase 1: Planning ({test_file}) ---")
    planner_query = f"{PLANNER_PROMPT}\n\n punchline:'{target_sentence}' "
    planner_agent = create_llm(planner_query)
    plan = planner_agent.generate_response()
    print(f"Plan generated:\n{plan}")
    
    # Phase 2: Execution
    print(f"--- Orchestra Phase 2: Execution ({test_file}) ---")
    executor_query_text = EXECUTOR_PROMPT.format(plan=plan)
    executor_query = f"{executor_query_text}\n\n punchline:'{target_sentence}' "
    executor_agent = create_llm(executor_query)
    observations = executor_agent.generate_response()

    # Phase 3: Synthesis
    print(f"--- Orchestra Phase 3: Synthesis ({test_file}) ---")
    synthesizer_query_text = SYNTHESIZER_PROMPT.format(observations=observations)
    synthesizer_query = f"{synthesizer_query_text}\n\n punchline:'{target_sentence}' "
    synthesizer_agent = create_llm(synthesizer_query)
    final_verdict = synthesizer_agent.generate_response()

    print('------------------------------------------')
    print(f"Verdict: {final_verdict}")
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [final_verdict],
            'plan': plan,
            'observations': observations,
            'label': dataset[test_file]['sarcasm'],
            'method': 'multi_agent_orchestra'
        }
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    dataset_path = './mustard_dataset_test_sample_100.json'
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f'Total Test Cases: {len(test_list)}')

    for test_file in test_list:
        run_instance(test_file)
        time.sleep(5)

