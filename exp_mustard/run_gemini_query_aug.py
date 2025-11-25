import json
import sys
import time
import statistics
import concurrent.futures

from exp_baselines import GeminiMultimodalLLM

sys.path.append('..')

# Explicit sub-queries to augment the context with more dimensions
SUB_QUERIES = {
    "context_analysis": "Analyze the conversation context and the literal meaning of the punchline. Is the statement logically consistent with the previous dialogue?",
    "visual_analysis": "Focus ONLY on the video frames. Describe the speaker's facial expressions and body language during the punchline. Do they look sincere, mocking, deadpan, or exaggerated?",
    "audio_analysis": "Focus ONLY on the audio. Describe the speaker's tone, pitch, and speed. Is there a drawl, a sudden pitch change, or a monotonous delivery that contrasts with the words?",
    "emotional_shift": "Analyze the emotional trajectory. How does the emotion in the punchline compare to the emotion in the context? Is there a sudden, unexpected shift (e.g., from serious to playful, or from angry to overly polite)?",
    "audience_reaction": "Observe the reaction of other characters in the scene (if any) or listen for background laughter/reactions. Do others seem confused, amused, or offended? Does background laughter suggest a joke?",
    "intent_analysis": "Infer the speaker's intent. Are they trying to be informative, hurtful, funny, or complaining? Does the statement seem to serve a communicative goal other than its literal meaning?",
    "contradiction_check": "Compare the literal meaning of the words with the non-verbal cues (visual and audio). Is there a contradiction? For example, saying something positive with a negative tone or expression."
}

FINAL_DECISION_PROMPT = (
    "You are an expert sarcasm detector. You have analyzed the input from multiple perspectives. Here are your findings:\n\n"
    "1. Context Analysis: {context_analysis}\n"
    "2. Visual Analysis: {visual_analysis}\n"
    "3. Audio Analysis: {audio_analysis}\n"
    "4. Emotional Shift: {emotional_shift}\n"
    "5. Audience/Environment Reaction: {audience_reaction}\n"
    "6. Intent Analysis: {intent_analysis}\n"
    "7. Contradiction Check: {contradiction_check}\n\n"
    "Based on these detailed findings, determine if the punchline is sarcastic.\n"
    "Your final answer must start with 'Yes' or 'No', followed by your reasoning."
)

# Pricing for Gemini 2.0 Flash Lite (Estimated)
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
        print("PERFORMANCE & COST SUMMARY (Augmentation - Parallel)")
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


def run_instance(test_file, output_file='baseline_aug_parallel.jsonl'):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    
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
    
    # Helper to create LLM instance (not thread-safe if reusing same object, but we create new per call)
    def create_llm(query):
        return GeminiMultimodalLLM(
            file_name=test_file,
            image_frames_folder=video_frames_path,
            audio_file_path=audio_path,
            context=fullContext,
            query=query,
            model_name='gemini-2.0-flash-lite',
        )

    print(f"--- Augmentation Phase: Analyzing {test_file} from multiple angles (Parallel) ---")
    
    augmentations = {}

    # Define function for parallel execution
    def execute_sub_query(key, question):
        full_query = f"{question}\n\n punchline:'{target_sentence}' "
        agent = create_llm(full_query)
        response, usage = agent.generate_response()
        return key, response, usage

    # Use ThreadPoolExecutor for parallel calls
    # 7 threads for 7 queries
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(SUB_QUERIES)) as executor:
        future_to_query = {
            executor.submit(execute_sub_query, key, question): key 
            for key, question in SUB_QUERIES.items()
        }
        
        for future in concurrent.futures.as_completed(future_to_query):
            key = future_to_query[future]
            try:
                key, response, usage = future.result()
                augmentations[key] = response
                
                # Accumulate usage (thread-safe because we sum after completion)
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
                
            except Exception as exc:
                print(f'{key} generated an exception: {exc}')
                augmentations[key] = "Error during analysis."

    # Final Decision Phase
    print(f"--- Final Phase: Synthesizing Decision ---")
    final_query_text = FINAL_DECISION_PROMPT.format(**augmentations)
    final_query = f"{final_query_text}\n\n punchline:'{target_sentence}' "
    
    final_agent = create_llm(final_query)
    final_verdict, usage = final_agent.generate_response()
    
    total_prompt_tokens += usage.get('prompt_tokens', 0)
    total_completion_tokens += usage.get('completion_tokens', 0)

    end_time = time.time()
    duration = end_time - start_time
    
    tracker.add(duration, total_prompt_tokens, total_completion_tokens)

    print('------------------------------------------')
    print(f"Verdict: {final_verdict}")
    print(f"Time: {duration:.2f}s | Tokens: {total_prompt_tokens} in, {total_completion_tokens} out")
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [final_verdict],
            'augmentations': augmentations,
            'label': dataset[test_file]['sarcasm'],
            'method': 'query_decomposition_augmentation_parallel',
            'usage': {
                'prompt_tokens': total_prompt_tokens,
                'completion_tokens': total_completion_tokens
            },
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
            time.sleep(2)
    finally:
        tracker.print_summary()
