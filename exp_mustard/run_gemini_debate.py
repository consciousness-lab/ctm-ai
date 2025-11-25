import json
import sys
import time
import statistics

from exp_baselines import GeminiMultimodalLLM

sys.path.append('..')

ROUNDS = 5  # Number of debate rounds

# Prompts for different agents
PRO_PROMPT_INIT = (
    "You are a Sarcasm Advocate. Your goal is to convince the judge that the following punchline IS sarcastic.\n"
    "Analyze the text, audio, and video carefully. Find ANY subtle cue—a slight change in tone, a micro-expression, or contextual irony—that supports the theory that this is sarcasm.\n"
    "Ignore evidence to the contrary. Focus ONLY on proving it is sarcasm.\n"
    "Provide a concise opening argument."
)

PRO_PROMPT_REBUTTAL = (
    "You are a Sarcasm Advocate. You are debating a Sincerity Advocate.\n"
    "Here is the argument from the Sincerity Advocate (claiming it's NOT sarcasm):\n"
    "\"{opponent_arg}\"\n\n"
    "Your task: Refute their points using evidence from the text, audio, or video. Strengthen your case that this IS sarcasm.\n"
    "Provide a concise rebuttal."
)

CON_PROMPT_INIT = (
    "You are a Sincerity Advocate. Your goal is to convince the judge that the following punchline is NOT sarcastic (it is sincere, neutral, or just emotional).\n"
    "Analyze the text, audio, and video. Argue that the speaker means exactly what they say, or that any emotion shown is genuine, not ironic.\n"
    "Ignore evidence to the contrary. Focus ONLY on proving it is NOT sarcasm.\n"
    "Provide a concise opening argument."
)

CON_PROMPT_REBUTTAL = (
    "You are a Sincerity Advocate. You are debating a Sarcasm Advocate.\n"
    "Here is the argument from the Sarcasm Advocate (claiming it IS sarcasm):\n"
    "\"{opponent_arg}\"\n\n"
    "Your task: Refute their points. Explain why the cues they mentioned are actually genuine emotions or misunderstandings. Strengthen your case that this is NOT sarcasm.\n"
    "Provide a concise rebuttal."
)

JUDGE_PROMPT = (
    "You are an impartial Judge. You have watched a multi-round debate between a Sarcasm Advocate and a Sincerity Advocate regarding the punchline below.\n"
    "Here is the full transcript of their debate:\n\n"
    "{debate_history}\n\n"
    "Review the original context and the arguments presented. Who made the more compelling case based on the multimodal evidence?\n"
    "Your final answer must start with 'Yes' (if it is sarcasm) or 'No' (if it is not), followed by your verdict explaining which specific arguments convinced you."
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
        print("PERFORMANCE & COST SUMMARY (Debate)")
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


def run_instance(test_file, output_file='baseline_debate.jsonl'):
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

    debate_history = ""
    last_pro_arg = ""
    last_con_arg = ""

    print(f"--- Starting Debate for {test_file} ({ROUNDS} Rounds) ---")

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Round {round_num} ---")
        
        # --- Pro-Sarcasm Turn ---
        if round_num == 1:
            pro_query_text = PRO_PROMPT_INIT
        else:
            pro_query_text = PRO_PROMPT_REBUTTAL.format(opponent_arg=last_con_arg)
        
        pro_query = f"{pro_query_text}\n\n punchline:'{target_sentence}' "
        pro_agent = create_llm(pro_query)
        pro_arg, usage = pro_agent.generate_response()
        
        total_prompt_tokens += usage.get('prompt_tokens', 0)
        total_completion_tokens += usage.get('completion_tokens', 0)
        
        debate_history += f"Round {round_num} - Sarcasm Advocate: {pro_arg}\n\n"
        last_pro_arg = pro_arg
        print(f"[Pro]: {pro_arg[:100]}...")

        # --- Con-Sarcasm Turn ---
        if round_num == 1:
            con_query_text = CON_PROMPT_INIT
        else:
            con_query_text = CON_PROMPT_REBUTTAL.format(opponent_arg=last_pro_arg)
            
        con_query = f"{con_query_text}\n\n punchline:'{target_sentence}' "
        con_agent = create_llm(con_query)
        con_arg, usage = con_agent.generate_response()
        
        total_prompt_tokens += usage.get('prompt_tokens', 0)
        total_completion_tokens += usage.get('completion_tokens', 0)
        
        debate_history += f"Round {round_num} - Sincerity Advocate: {con_arg}\n\n"
        last_con_arg = con_arg
        print(f"[Con]: {con_arg[:100]}...")

    # --- Judge Turn ---
    print(f"--- Judge Verdict ({test_file}) ---")
    judge_query_text = JUDGE_PROMPT.format(debate_history=debate_history)
    judge_query = f"{judge_query_text}\n\n punchline:'{target_sentence}' "
    judge_agent = create_llm(judge_query)
    final_verdict, usage = judge_agent.generate_response()
    
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
            'debate_history': debate_history,
            'label': dataset[test_file]['sarcasm'],
            'method': 'multi_agent_debate_multiround',
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
