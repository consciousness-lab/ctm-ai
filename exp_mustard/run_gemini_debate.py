import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
from gemini_utils import (
    check_gemini_api_key,
    load_data,
    load_processed_keys,
    load_images_as_base64,
    prepare_audio_for_gemini,
    call_gemini_with_content,
    StatsTracker,
    save_result_to_jsonl,
)

sys.path.append("..")

# Check API key
check_gemini_api_key()
litellm.set_verbose = False

ROUNDS = 3  # Number of debate rounds

# Modality-specific agent prompts for sarcasm detection
VIDEO_AGENT_INIT = (
    "You are a Video Analysis Expert. You will analyze whether the person in the video is being sarcastic or not.\n"
    "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
)

AUDIO_AGENT_INIT = (
    "You are an Audio Analysis Expert. You will analyze whether the person in the audio is being sarcastic or not.\n"
    "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
)

TEXT_AGENT_INIT = (
    "You are a Text Analysis Expert. You will be given a punchline that was said by a person, and analysis whether the person is being sarcastic or not.\n"
    "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
)

VIDEO_AGENT_REFINE = (
    "You are a Video Analysis Expert. You previously analyzed the video of this punchline.\n"
    "Your previous answer: {own_answer}\n\n"
    "Here are the analyses from other experts:\n"
    "- Audio Expert: {audio_answer}\n"
    "- Text Expert: {text_answer}\n\n"
    "First, consider their perspectives and re-examine the video evidence carefully.\n"
    "Then, determine if this punchline is sarcastic or not.\n"
    "End your response with 'My Answer: Yes' or 'My Answer: No'."
)

AUDIO_AGENT_REFINE = (
    "You are an Audio Analysis Expert. You previously analyzed the audio of this punchline.\n"
    "Your previous answer: {own_answer}\n\n"
    "Here are the analyses from other experts:\n"
    "- Video Expert: {video_answer}\n"
    "- Text Expert: {text_answer}\n\n"
    "First, consider their perspectives and re-examine the audio evidence carefully.\n"
    "Then, determine if this punchline is sarcastic or not.\n"
    "End your response with 'My Answer: Yes' or 'My Answer: No'."
)

TEXT_AGENT_REFINE = (
    "You are a Text Analysis Expert. You previously analyzed this punchline.\n"
    "Your previous answer: {own_answer}\n\n"
    "Here are the analyses from other experts:\n"
    "- Video Expert: {video_answer}\n"
    "- Audio Expert: {audio_answer}\n\n"
    "First, consider their perspectives and re-examine the text evidence of the punchline carefully.\n"
    "Then, determine if this punchline is sarcastic or not.\n"
    "End your response with 'My Answer: Yes' or 'My Answer: No'."
)

JUDGE_PROMPT = (
    "You are an impartial Judge. Three experts have debated whether this punchline is sarcastic or not.\n"
    "Here is the discussion:\n\n{debate_history}\n\n"
    "Based on all evidence from the video, audio, and text analyses, determine if this punchline is sarcastic or not.\n"
    "Your answer must start with 'Yes' or 'No', followed by your reasoning."
)

# Pricing for Gemini 2.0 Flash Lite
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30

# Initialize tracker
tracker = StatsTracker(
    cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
)


def extract_answer(response):
    """Extract Yes/No from agent response."""
    if response is None:
        return "Unknown"
    response_lower = response.lower()
    if "my answer: yes" in response_lower:
        return "Yes"
    elif "my answer: no" in response_lower:
        return "No"
    # Fallback
    if response_lower.strip().endswith("yes"):
        return "Yes"
    elif response_lower.strip().endswith("no"):
        return "No"
    return "Unknown"


def run_instance(test_file, dataset, output_file="mustard_debate_0207.jsonl"):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0

    target_sentence = dataset[test_file]["utterance"]

    text_list = dataset[test_file]["context"].copy()
    text_list.append(target_sentence)
    fullContext = "\n".join(text_list)

    audio_path = f"mustard_audios/{test_file}_audio.mp4"
    video_frames_path = f"mustard_frames/{test_file}_frames"

    # Load multimodal data once
    video_frames = load_images_as_base64(video_frames_path, max_frames=10)
    audio_file = prepare_audio_for_gemini(audio_path)

    model_name = "gemini/gemini-2.0-flash-lite"

    debate_history = ""

    # Store answers from each agent
    video_answer = ""
    audio_answer = ""
    text_answer = ""

    print(f"--- Multimodal Debate for {test_file} ({ROUNDS} Rounds) ---")

    def run_agent(agent_type, query):
        """Helper function to run a single agent and return results."""
        if agent_type == "video":
            # Video agent: only video frames
            response, usage = call_gemini_with_content(
                query=query,
                images=video_frames,
                audio=None,
                context=None,
                model=model_name,
                temperature=1.0,
            )
        elif agent_type == "audio":
            # Audio agent: only audio
            response, usage = call_gemini_with_content(
                query=query,
                images=None,
                audio=audio_file,
                context=None,
                model=model_name,
                temperature=1.0,
            )
        elif agent_type == "text":
            # Text agent: only text context
            response, usage = call_gemini_with_content(
                query=query,
                images=None,
                audio=None,
                context=fullContext,
                model=model_name,
                temperature=1.0,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_type, response, usage

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Round {round_num} ---")
        round_history = f"=== Round {round_num} ===\n\n"

        # Prepare queries for all three agents
        if round_num == 1:
            video_query = VIDEO_AGENT_INIT
            audio_query = AUDIO_AGENT_INIT
            text_query = f"{TEXT_AGENT_INIT}\n\npunchline: '{target_sentence}'"
        else:
            video_query = VIDEO_AGENT_REFINE.format(
                own_answer=video_answer,
                audio_answer=audio_answer,
                text_answer=text_answer,
            )
            audio_query = AUDIO_AGENT_REFINE.format(
                own_answer=audio_answer,
                video_answer=video_answer,
                text_answer=text_answer,
            )
            text_query = f"{TEXT_AGENT_REFINE.format(own_answer=text_answer, video_answer=video_answer, audio_answer=audio_answer)}\n\npunchline: '{target_sentence}'"

        # Run all three agents in parallel
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {
                executor.submit(run_agent, "video", video_query): "video",
                executor.submit(run_agent, "audio", audio_query): "audio",
                executor.submit(run_agent, "text", text_query): "text",
            }

            results = {}
            for future in as_completed(futures):
                agent_type, response, usage = future.result()
                results[agent_type] = (response, usage)
                num_api_calls += 1
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)

        # Extract results
        video_response, _ = results["video"]
        audio_response, _ = results["audio"]
        text_response, _ = results["text"]

        video_answer = video_response
        audio_answer = audio_response
        text_answer = text_response

        video_vote = extract_answer(video_response)
        audio_vote = extract_answer(audio_response)
        text_vote = extract_answer(text_response)

        round_history += f"[Video Expert]: {video_response}\n\n"
        round_history += f"[Audio Expert]: {audio_response}\n\n"
        round_history += f"[Text Expert]: {text_response}\n\n"

        print(
            f"  [Video]: {video_vote} - {video_response[:80] if video_response else 'None'}..."
        )
        print(
            f"  [Audio]: {audio_vote} - {audio_response[:80] if audio_response else 'None'}..."
        )
        print(
            f"  [Text]:  {text_vote} - {text_response[:80] if text_response else 'None'}..."
        )

        debate_history += round_history

    # --- Judge ---
    print("--- Judge Verdict ---")
    judge_query = f"{JUDGE_PROMPT.format(debate_history=debate_history)}\n\npunchline: '{target_sentence}'"
    # Judge uses text modality with context
    final_verdict, usage = call_gemini_with_content(
        query=judge_query,
        images=None,
        audio=None,
        context=fullContext,
        model=model_name,
        temperature=1.0,
    )
    num_api_calls += 1
    total_prompt_tokens += usage.get("prompt_tokens", 0)
    total_completion_tokens += usage.get("completion_tokens", 0)

    end_time = time.time()
    duration = end_time - start_time

    tracker.add(duration, total_prompt_tokens, total_completion_tokens, num_api_calls)

    print("------------------------------------------")
    print(f"Verdict: {final_verdict}")
    print(
        f"Time: {duration:.2f}s | API Calls: {num_api_calls} | Tokens: {total_prompt_tokens} in, {total_completion_tokens} out"
    )
    print("------------------------------------------")

    result = {
        test_file: {
            "answer": [final_verdict],
            "debate_history": debate_history,
            "final_votes": {
                "video": extract_answer(video_answer),
                "audio": extract_answer(audio_answer),
                "text": extract_answer(text_answer),
            },
            "label": dataset[test_file]["sarcasm"],
            "method": "multimodal_debate_3agents",
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "api_calls": num_api_calls,
            },
            "latency": duration,
        }
    }

    save_result_to_jsonl(result, output_file)


if __name__ == "__main__":
    output_file = "mustard_debate_0207.jsonl"
    dataset_path = "./mustard_dataset/mustard_dataset_test.json"
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f"Total Test Cases: {len(test_list)}")

    # Load already processed keys for resume
    processed_keys = load_processed_keys(output_file)
    if processed_keys:
        print(f"Resuming: {len(processed_keys)} already processed, skipping...")

    try:
        for test_file in test_list:
            if test_file in processed_keys:
                continue
            try:
                run_instance(test_file, dataset, output_file)
            except Exception as e:
                print(f"[ERROR] Failed to process {test_file}: {e}")
                print("[INFO] Skipping and continuing with next sample...")
                continue
            time.sleep(2)
    finally:
        tracker.print_summary(f"Multimodal Debate - Rounds: {ROUNDS}")
