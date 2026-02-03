import json
import sys
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed

from exp_baselines import GeminiMultimodalLLM

sys.path.append('..')

ROUNDS = 3  # Number of questioning rounds

# Controller prompts
CONTROLLER_INIT_PROMPT = (
    "You are a Humor Detection Controller. Your task is to coordinate three modality experts "
    "(Video, Audio, Text) to determine if the punchline is humorous.\n\n"
    "The punchline is: '{punchline}'\n\n"
    "For Round 1, generate THREE specific questions - one for each expert:\n"
    "1. A question for the VIDEO expert about visual cues related to humor\n"
    "2. A question for the AUDIO expert about vocal cues related to humor\n"
    "3. A question for the TEXT expert about linguistic cues related to humor\n\n"
    "Format your response exactly as:\n"
    "VIDEO_QUESTION: [your question]\n"
    "AUDIO_QUESTION: [your question]\n"
    "TEXT_QUESTION: [your question]"
)

CONTROLLER_FOLLOWUP_PROMPT = (
    "You are a Humor Detection Controller coordinating three modality experts.\n\n"
    "The punchline is: '{punchline}'\n\n"
    "Here are the responses from the previous round:\n"
    "- Video Expert: {video_response}\n"
    "- Audio Expert: {audio_response}\n"
    "- Text Expert: {text_response}\n\n"
    "Based on these responses, generate follow-up questions for Round {round_num} to dig deeper "
    "or clarify any uncertainties. Focus on areas where the evidence is unclear or conflicting.\n\n"
    "Format your response exactly as:\n"
    "VIDEO_QUESTION: [your question]\n"
    "AUDIO_QUESTION: [your question]\n"
    "TEXT_QUESTION: [your question]"
)

CONTROLLER_DECISION_PROMPT = (
    "You are a Humor Detection Controller. You have gathered evidence from three modality experts "
    "over {num_rounds} rounds of questioning.\n\n"
    "The punchline is: '{punchline}'\n\n"
    "Here is the complete conversation history:\n{conversation_history}\n\n"
    "Based on all the evidence gathered, make your final determination.\n"
    "If you think these inputs include exaggerated description or are expressing sarcastic meaning, answer 'Yes'.\n"
    "If you think these inputs are neutral or just common meaning, answer 'No'.\n"
    "Your answer must start with 'Yes' or 'No', followed by your reasoning."
)

# Agent prompts
VIDEO_AGENT_PROMPT = (
    "You are a Video Analysis Expert. Analyze the video frames to answer the following question.\n"
    "Focus ONLY on visual cues: facial expressions, body language, gestures, comedic timing.\n\n"
    "Question: {question}\n\n"
    "Provide a concise but detailed answer based on what you observe in the video."
)

AUDIO_AGENT_PROMPT = (
    "You are an Audio Analysis Expert. Analyze the audio to answer the following question.\n"
    "Focus ONLY on audio cues: tone of voice, comedic delivery, timing, audience reactions.\n\n"
    "Question: {question}\n\n"
    "Provide a concise but detailed answer based on what you hear in the audio."
)

TEXT_AGENT_PROMPT = (
    "You are a Text Analysis Expert. Analyze the text context to answer the following question.\n"
    "Focus ONLY on linguistic cues: wordplay, puns, setup-punchline structure, unexpected twists.\n\n"
    "Question: {question}\n\n"
    "Provide a concise but detailed answer based on the text."
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

        print("\n" + "="*50)
        print(f"PERFORMANCE & COST SUMMARY (Query Aug - Controller)")
        print(f"  Rounds: {ROUNDS} | Agents: Video, Audio, Text")
        print("="*50)
        print(f"Total Samples Processed: {len(self.times)}")
        print(f"Total API Calls:         {total_api_calls}")
        print("-" * 40)
        print(f"Average Time per Sample:  {avg_time:.2f} seconds")
        print(f"Average API Calls/Sample: {avg_api_calls:.1f}")
        print(f"Average Input Tokens:     {avg_input:.1f}")
        print(f"Average Output Tokens:    {avg_output:.1f}")
        print(f"Total Input Tokens:       {sum(self.input_tokens)}")
        print(f"Total Output Tokens:      {sum(self.output_tokens)}")
        print("-" * 40)
        print(f"Average Cost per Sample:  ${avg_cost:.6f}")
        print(f"Total Cost for Run:       ${total_cost:.6f}")
        print("="*50 + "\n")


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


def parse_controller_questions(response):
    """Parse controller response to extract questions for each agent."""
    questions = {'video': '', 'audio': '', 'text': ''}
    if response is None:
        return questions
    
    lines = response.split('\n')
    for line in lines:
        line_lower = line.lower()
        if 'video_question:' in line_lower:
            questions['video'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif 'audio_question:' in line_lower:
            questions['audio'] = line.split(':', 1)[1].strip() if ':' in line else ''
        elif 'text_question:' in line_lower:
            questions['text'] = line.split(':', 1)[1].strip() if ':' in line else ''
    
    # Fallback: if parsing failed, use generic questions
    if not questions['video']:
        questions['video'] = "What visual cues do you observe that might indicate humor?"
    if not questions['audio']:
        questions['audio'] = "What audio cues do you observe that might indicate humor?"
    if not questions['text']:
        questions['text'] = "What textual cues do you observe that might indicate humor?"
    
    return questions


def run_instance(test_file, dataset, output_file='urfunny_aug_parallel_0128_v2.jsonl'):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0
    
    sample = dataset[test_file]
    punchline = sample['punchline_sentence']
    
    text_list = sample['context_sentences'].copy()
    text_list.append(punchline)
    fullContext = '\n'.join(text_list)
    
    audio_path = f'urfunny_audios/{test_file}_audio.mp4'
    video_frames_path = f'urfunny_frames/{test_file}_frames'
    
    def create_llm(query):
        return GeminiMultimodalLLM(
            file_name=test_file,
            image_frames_folder=video_frames_path,
            audio_file_path=audio_path,
            context=fullContext,
            query=query,
            model_name='gemini-2.0-flash-lite',
        )

    print(f"--- Controller-Agent Query Aug for {test_file} ({ROUNDS} Rounds) ---")
    
    conversation_history = ""
    video_response = ""
    audio_response = ""
    text_response = ""

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Round {round_num} ---")
        round_history = f"\n=== Round {round_num} ===\n"
        
        # Controller generates questions
        if round_num == 1:
            controller_query = CONTROLLER_INIT_PROMPT.format(punchline=punchline)
        else:
            controller_query = CONTROLLER_FOLLOWUP_PROMPT.format(
                punchline=punchline,
                video_response=video_response[:500],
                audio_response=audio_response[:500],
                text_response=text_response[:500],
                round_num=round_num
            )
        
        controller_llm = create_llm(controller_query)
        controller_response, usage = controller_llm.generate_response()
        num_api_calls += 1
        total_prompt_tokens += usage.get('prompt_tokens', 0)
        total_completion_tokens += usage.get('completion_tokens', 0)
        
        questions = parse_controller_questions(controller_response)
        round_history += f"Controller Questions:\n{controller_response}\n\n"
        print(f"  [Controller] Generated questions")
        
        # Run all three agents in parallel
        def run_agent(agent_type, query):
            agent = create_llm(query)
            response, usage = agent.generate_response()
            return agent_type, response, usage
        
        video_query = VIDEO_AGENT_PROMPT.format(question=questions['video'])
        audio_query = AUDIO_AGENT_PROMPT.format(question=questions['audio'])
        text_query = TEXT_AGENT_PROMPT.format(question=questions['text'])
        
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_agent, 'video', video_query): 'video',
                executor.submit(run_agent, 'audio', audio_query): 'audio',
                executor.submit(run_agent, 'text', text_query): 'text',
            }
            
            results = {}
            for future in as_completed(futures):
                agent_type, response, usage = future.result()
                results[agent_type] = (response, usage)
                num_api_calls += 1
                total_prompt_tokens += usage.get('prompt_tokens', 0)
                total_completion_tokens += usage.get('completion_tokens', 0)
        
        video_response, _ = results['video']
        audio_response, _ = results['audio']
        text_response, _ = results['text']
        
        round_history += f"Video Expert: {video_response}\n\n"
        round_history += f"Audio Expert: {audio_response}\n\n"
        round_history += f"Text Expert: {text_response}\n\n"
        
        print(f"  [Video] {video_response[:80] if video_response else 'None'}...")
        print(f"  [Audio] {audio_response[:80] if audio_response else 'None'}...")
        print(f"  [Text] {text_response[:80] if text_response else 'None'}...")
        
        conversation_history += round_history

    # Controller makes final decision
    print(f"--- Controller Final Decision ---")
    decision_query = CONTROLLER_DECISION_PROMPT.format(
        punchline=punchline,
        num_rounds=ROUNDS,
        conversation_history=conversation_history
    )
    decision_llm = create_llm(decision_query)
    final_verdict, usage = decision_llm.generate_response()
    num_api_calls += 1
    total_prompt_tokens += usage.get('prompt_tokens', 0)
    total_completion_tokens += usage.get('completion_tokens', 0)

    end_time = time.time()
    duration = end_time - start_time
    
    tracker.add(duration, total_prompt_tokens, total_completion_tokens, num_api_calls)

    print('------------------------------------------')
    print(f"Verdict: {final_verdict}")
    print(f"Time: {duration:.2f}s | API Calls: {num_api_calls} | Tokens: {total_prompt_tokens} in, {total_completion_tokens} out")
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [final_verdict],
            'conversation_history': conversation_history,
            'label': sample['label'],
            'method': 'controller_agent_query_aug',
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
    output_file = 'urfunny_aug_parallel_0128_v2.jsonl'
    dataset_path = './urfunny_dataset_test.json'
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
