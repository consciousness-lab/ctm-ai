import json
import sys
import time
import statistics

from exp_baselines import GeminiMultimodalLLM

sys.path.append('..')

SYS_PROMPT = (
    'Please analyze the inputs provided to determine the punchline provided sarcasm or not.'
    "Your answer should start with 'Yes' or 'No'."
    "If you think these inputs includes exaggerated description or its real meaning is not aligned with the original one, please answer 'Yes'."
    "If you think these inputs is neutral or its true meaning is not different from its original one, please answer 'No'."
    'You should also provide your reason for your answer.'
)

# Pricing for Gemini 2.0 Flash Lite (Estimated based on Flash tiers)
# Adjust these values as per actual pricing
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30

class StatsTracker:
    def __init__(self):
        self.times = []
        self.input_tokens = []
        self.output_tokens = []
        self.costs = []

    def add(self, duration, input_tok, output_tok):
        cost = (input_tok / 1_000_000 * COST_INPUT_PER_1M) + \
               (output_tok / 1_000_000 * COST_OUTPUT_PER_1M)
        
        self.times.append(duration)
        self.input_tokens.append(input_tok)
        self.output_tokens.append(output_tok)
        self.costs.append(cost)

    def print_summary(self):
        if not self.times:
            print("No stats to report.")
            return

        avg_time = statistics.mean(self.times)
        avg_input = statistics.mean(self.input_tokens)
        avg_output = statistics.mean(self.output_tokens)
        avg_cost = statistics.mean(self.costs)
        total_cost = sum(self.costs)

        print("\n" + "="*40)
        print("PERFORMANCE & COST SUMMARY")
        print("="*40)
        print(f"Total Samples Processed: {len(self.times)}")
        print("-" * 30)
        print(f"Average Time per Request: {avg_time:.2f} seconds")
        print(f"Average Input Tokens:     {avg_input:.1f}")
        print(f"Average Output Tokens:    {avg_output:.1f}")
        print(f"Average Cost per Request: ${avg_cost:.6f}")
        print("-" * 30)
        print(f"Total Cost for Run:       ${total_cost:.6f}")
        print("="*40 + "\n")

tracker = StatsTracker()


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file='baseline1.jsonl'):
    start_time = time.time()
    
    dataset = load_data('./mustard_dataset_test_sample_100.json')
    target_sentence = dataset[test_file]['utterance']
    query = f"{SYS_PROMPT}\n\n punchline:'{target_sentence}' "
    text_list = dataset[test_file]['context']
    text_list.append(target_sentence)
    fullContext = ''
    for i in range(len(text_list)):
        currentUtterance = f'{text_list[i]} \n'
        fullContext += currentUtterance
    audio_path = f'mustard_audios/{test_file}_audio.mp4'
    video_frames_path = f'mustard_frames/{test_file}_frames'
    
    gemini_llm = GeminiMultimodalLLM(
        file_name=test_file,
        image_frames_folder=video_frames_path,
        audio_file_path=audio_path,
        context=fullContext,
        query=query,
        model_name='gemini-2.0-flash-lite',
    )
    
    answer, usage = gemini_llm.generate_response()
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Record stats
    tracker.add(duration, usage['prompt_tokens'], usage['completion_tokens'])
    
    print('------------------------------------------')
    print(f"Processed {test_file} in {duration:.2f}s")
    print(f"Tokens: In={usage['prompt_tokens']}, Out={usage['completion_tokens']}")
    print(answer)
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [answer],
            'label': dataset[test_file]['sarcasm'],
            'usage': usage,
            'latency': duration
        }
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    dataset_path = './mustard_dataset_test_sample_100.json'
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f'Total Test Cases: {len(test_list)}')

    try:
        for test_file in test_list:
            run_instance(test_file)
            time.sleep(2) # Reduced wait time as we have rate limit handling now
    finally:
        # Ensure stats are printed even if interrupted
        tracker.print_summary()
