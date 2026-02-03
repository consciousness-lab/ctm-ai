import json
import sys
import time
import statistics
import concurrent.futures
from collections import Counter

from exp_baselines import GeminiMultimodalLLM

sys.path.append('..')

# Number of votes
N_VOTES = 3

# Higher temperature for diversity in voting (only for voting baseline)
# 2.0 is the max for Gemini to get more diverse outputs
TEMPERATURE = 1.0

SYS_PROMPT = (
    'Please analyze the inputs provided to determine the punchline provided sarcasm or not.'
    "Your answer should start with 'Yes' or 'No'."
    "If you think these inputs includes exaggerated description or its real meaning is not aligned with the original one, please answer 'Yes'."
    "If you think these inputs is neutral or its true meaning is not different from its original one, please answer 'No'."
    'You should also provide your reason for your answer.'
)

# Pricing for Gemini 2.0 Flash Lite
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30


class StatsTracker:
    def __init__(self):
        self.times = []
        self.input_tokens = []
        self.output_tokens = []
        self.costs = []
        self.api_calls = []

    def add(self, duration, input_tok, output_tok, num_api_calls):
        cost = (input_tok / 1_000_000 * COST_INPUT_PER_1M) + \
               (output_tok / 1_000_000 * COST_OUTPUT_PER_1M)
        
        self.times.append(duration)
        self.input_tokens.append(input_tok)
        self.output_tokens.append(output_tok)
        self.costs.append(cost)
        self.api_calls.append(num_api_calls)

    def print_summary(self):
        if not self.times:
            print("No stats to report.")
            return

        avg_time = statistics.mean(self.times)
        avg_input = statistics.mean(self.input_tokens)
        avg_output = statistics.mean(self.output_tokens)
        avg_cost = statistics.mean(self.costs)
        total_cost = sum(self.costs)
        total_api_calls = sum(self.api_calls)
        avg_api_calls = statistics.mean(self.api_calls)

        print("\n" + "="*40)
        print(f"PERFORMANCE & COST SUMMARY (Voting N={N_VOTES})")
        print("="*40)
        print(f"Total Samples Processed: {len(self.times)}")
        print(f"Total API Calls:         {total_api_calls}")
        print("-" * 30)
        print(f"Average Time per Sample:  {avg_time:.2f} seconds")
        print(f"Average API Calls/Sample: {avg_api_calls:.1f}")
        print(f"Average Input Tokens:     {avg_input:.1f}")
        print(f"Average Output Tokens:    {avg_output:.1f}")
        print(f"Total Input Tokens:       {sum(self.input_tokens)}")
        print(f"Total Output Tokens:      {sum(self.output_tokens)}")
        print("-" * 30)
        print(f"Average Cost per Sample:  ${avg_cost:.6f}")
        print(f"Total Cost for Run:       ${total_cost:.6f}")
        print("="*40 + "\n")


tracker = StatsTracker()


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def load_processed_keys(output_file):
    """Load already processed keys from output file for resume functionality."""
    processed = set()
    try:
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    processed.update(result.keys())
    except FileNotFoundError:
        pass
    return processed


def extract_vote(answer):
    """Extract Yes/No from the beginning of answer string."""
    if answer is None:
        return 'Unknown'
    answer_lower = answer.strip().lower()
    
    # Check if answer starts with Yes or No
    if answer_lower.startswith('yes'):
        return 'Yes'
    elif answer_lower.startswith('no'):
        return 'No'
    
    # Fallback: check the first 50 characters
    first_part = answer_lower[:50]
    if 'yes' in first_part and 'no' not in first_part:
        return 'Yes'
    elif 'no' in first_part and 'yes' not in first_part:
        return 'No'
    
    # Last resort: check anywhere
    if 'yes' in answer_lower:
        return 'Yes'
    elif 'no' in answer_lower:
        return 'No'
    
    return 'Unknown'


def majority_vote(votes):
    """Return the majority vote result."""
    vote_counts = Counter(votes)
    most_common = vote_counts.most_common(1)
    if most_common:
        return most_common[0][0]
    return 'Unknown'


def run_instance(test_file, dataset, output_file='baseline_voting_0128.jsonl'):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0
    
    target_sentence = dataset[test_file]['utterance']
    query = f"{SYS_PROMPT}\n\n punchline:'{target_sentence}' "
    
    text_list = dataset[test_file]['context'].copy()
    text_list.append(target_sentence)
    fullContext = ''
    for i in range(len(text_list)):
        currentUtterance = f'{text_list[i]} \n'
        fullContext += currentUtterance
    
    audio_path = f'mustard_audios/{test_file}_audio.mp4'
    video_frames_path = f'mustard_frames/{test_file}_frames'
    
    def create_llm():
        return GeminiMultimodalLLM(
            file_name=test_file,
            image_frames_folder=video_frames_path,
            audio_file_path=audio_path,
            context=fullContext,
            query=query,
            model_name='gemini-2.0-flash-lite',
            temperature=TEMPERATURE,
        )
    
    def single_vote(vote_idx):
        """Execute a single vote."""
        llm = create_llm()
        answer, usage = llm.generate_response()
        return vote_idx, answer, usage

    print(f"--- Voting ({N_VOTES} votes) for {test_file} ---")
    
    all_answers = []
    all_votes = []
    
    # Use ThreadPoolExecutor for parallel voting
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_VOTES) as executor:
        futures = {executor.submit(single_vote, i): i for i in range(N_VOTES)}
        
        for future in concurrent.futures.as_completed(futures):
            vote_idx = futures[future]
            try:
                idx, answer, usage = future.result()
                all_answers.append(answer)
                vote = extract_vote(answer)
                all_votes.append(vote)
                
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                num_api_calls += 1
                
                print(f"  Vote {idx + 1}: {vote}")
                
            except Exception as exc:
                print(f'Vote {vote_idx + 1} generated an exception: {exc}')
                all_answers.append("Error")
                all_votes.append("Unknown")
    
    # Majority voting
    final_vote = majority_vote(all_votes)
    final_verdict = f"{final_vote}. (Votes: {all_votes})"
    
    end_time = time.time()
    duration = end_time - start_time
    
    tracker.add(duration, total_prompt_tokens, total_completion_tokens, num_api_calls)

    print('------------------------------------------')
    print(f"Final Verdict: {final_verdict}")
    print(f"Time: {duration:.2f}s | API Calls: {num_api_calls} | Tokens: {total_prompt_tokens} in, {total_completion_tokens} out")
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [final_verdict],
            'individual_answers': all_answers,
            'votes': all_votes,
            'final_vote': final_vote,
            'label': dataset[test_file]['sarcasm'],
            'method': f'voting_n{N_VOTES}',
            'usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens,
                'api_calls': num_api_calls
            },
            'latency': duration
        }
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    output_file = 'baseline_voting_0128_v2.jsonl'
    dataset_path = './mustard_dataset/mustard_dataset_test.json'
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f'Total Test Cases: {len(test_list)}')
    
    # Load already processed keys for resume
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f'Resuming: {len(processed_keys)} already processed, skipping...')

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
            try:
                run_instance(test_file, dataset, output_file)
            except Exception as e:
                print(f"[ERROR] Failed to process {test_file}: {e}")
                print(f"[INFO] Skipping and continuing with next sample...")
                continue
            time.sleep(2)
    finally:
        tracker.print_summary()
