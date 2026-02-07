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
litellm.set_verbose = False  # Set to True for debugging

ROUNDS = 3  # Number of questioning rounds

# Controller prompts - similar style to debate prompts
CONTROLLER_INIT_PROMPT = (
    "You are a Humor Detection Controller. Your task is to coordinate three modality experts "
    "(Video, Audio, Text) to determine if the person is being humorous or not.\n\n"
    "For Round 1, generate THREE specific questions - one for each expert:\n"
    "1. A question for the VIDEO expert to analyze visual expressions and gestures\n"
    "2. A question for the AUDIO expert to analyze vocal tone and speech patterns\n"
    "3. A question for the TEXT expert to analyze the punchline text\n\n"
    "The questions should guide experts to first analyze, then determine if the person is humorous.\n\n"
    "Format your response exactly as:\n"
    "VIDEO_QUESTION: [your question]\n"
    "AUDIO_QUESTION: [your question]\n"
    "TEXT_QUESTION: [your question]"
)

CONTROLLER_FOLLOWUP_PROMPT = (
    "You are a Humor Detection Controller coordinating three modality experts.\n\n"
    "Here are the responses from Round {prev_round}:\n"
    "- Video Expert: {video_response}\n"
    "- Audio Expert: {audio_response}\n"
    "- Text Expert: {text_response}\n\n"
    "Based on these responses, generate follow-up questions for Round {round_num} to dig deeper.\n"
    "Focus on areas where the evidence is unclear or where experts might disagree.\n\n"
    "Format your response exactly as:\n"
    "VIDEO_QUESTION: [your question]\n"
    "AUDIO_QUESTION: [your question]\n"
    "TEXT_QUESTION: [your question]"
)

CONTROLLER_DECISION_PROMPT = (
    "You are a Humor Detection Controller. You have gathered evidence from three modality experts "
    "over {num_rounds} rounds of questioning.\n\n"
    "Here is the complete conversation history:\n{conversation_history}\n\n"
    "Based on all the evidence gathered from video, audio, and text analyses, "
    "determine if this person is humorous or not.\n"
    "Your answer must start with 'Yes' or 'No', followed by your reasoning."
)

# Agent prompts - similar to debate agent prompts
VIDEO_AGENT_PROMPT = (
    "You are a Video Analysis Expert. You will analyze video frames showing a person.\n"
    "Question: {question}\n\n"
    "First, carefully observe and describe the person's visual expressions, gestures, and body language.\n"
    "Then, based on your analysis, provide your answer to the question."
)

AUDIO_AGENT_PROMPT = (
    "You are an Audio Analysis Expert. You will analyze audio of a person speaking.\n"
    "Question: {question}\n\n"
    "First, carefully listen and describe the person's vocal tone, intonation, speech patterns, and any audio cues.\n"
    "Then, based on your analysis, provide your answer to the question."
)

TEXT_AGENT_PROMPT = (
    "You are a Text Analysis Expert. You will be given a punchline that was said by a person.\n"
    "Question: {question}\n\n"
    "Punchline: '{punchline}'\n\n"
    "First, carefully read and analyze the text content, word choice, tone, and any linguistic patterns.\n"
    "Then, based on your analysis of what this person said, provide your answer to the question."
)

# Pricing for Gemini 2.0 Flash Exp
COST_INPUT_PER_1M = 0.075
COST_OUTPUT_PER_1M = 0.30

# Initialize tracker
tracker = StatsTracker(
    cost_input_per_1m=COST_INPUT_PER_1M, cost_output_per_1m=COST_OUTPUT_PER_1M
)


def parse_controller_questions(response):
    """Parse controller response to extract questions for each agent."""
    questions = {"video": "", "audio": "", "text": ""}
    if response is None:
        return questions

    lines = response.split("\n")
    for line in lines:
        line_lower = line.lower()
        if "video_question:" in line_lower:
            questions["video"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif "audio_question:" in line_lower:
            questions["audio"] = line.split(":", 1)[1].strip() if ":" in line else ""
        elif "text_question:" in line_lower:
            questions["text"] = line.split(":", 1)[1].strip() if ":" in line else ""

    # Fallback: if parsing failed, use generic questions
    if not questions["video"]:
        questions["video"] = (
            "What visual cues do you observe that might indicate humor?"
        )
    if not questions["audio"]:
        questions["audio"] = "What audio cues do you observe that might indicate humor?"
    if not questions["text"]:
        questions["text"] = (
            "What textual cues do you observe that might indicate humor?"
        )

    return questions


def run_instance(test_file, dataset, output_file="urfunny_aug_0205.jsonl"):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0

    sample = dataset[test_file]
    punchline = sample["punchline_sentence"]

    audio_path = f"test_inputs/urfunny_audios/{test_file}_audio.mp4"
    video_frames_path = f"test_inputs/urfunny_frames/{test_file}_frames"

    # Load multimodal data once
    video_frames = load_images_as_base64(video_frames_path, max_frames=10)
    audio_file = prepare_audio_for_gemini(audio_path)

    model_name = "gemini/gemini-2.0-flash-exp"

    print(f"--- Controller-Agent Query Aug for {test_file} ({ROUNDS} Rounds) ---")

    conversation_history = ""
    video_response = ""
    audio_response = ""
    text_response = ""

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
            # Text agent: only punchline (no context)
            response, usage = call_gemini_with_content(
                query=query,
                images=None,
                audio=None,
                context=None,
                model=model_name,
                temperature=1.0,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_type, response, usage

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Round {round_num} ---")
        round_history = f"\n=== Round {round_num} ===\n"

        # Controller generates questions
        if round_num == 1:
            controller_query = CONTROLLER_INIT_PROMPT
        else:
            controller_query = CONTROLLER_FOLLOWUP_PROMPT.format(
                prev_round=round_num - 1,
                video_response=video_response[:500] if video_response else "None",
                audio_response=audio_response[:500] if audio_response else "None",
                text_response=text_response[:500] if text_response else "None",
                round_num=round_num,
            )

        # Controller uses text modality with context
        controller_response, usage = call_gemini_with_content(
            query=controller_query,
            images=None,
            audio=None,
            context=None,
            model=model_name,
            temperature=1.0,
        )
        num_api_calls += 1
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_completion_tokens += usage.get("completion_tokens", 0)

        questions = parse_controller_questions(controller_response)
        round_history += f"[Controller Questions]:\n{controller_response}\n\n"
        print("  [Controller] Generated questions")

        # Prepare queries for all three agents
        video_query = VIDEO_AGENT_PROMPT.format(question=questions["video"])
        audio_query = AUDIO_AGENT_PROMPT.format(question=questions["audio"])
        text_query = TEXT_AGENT_PROMPT.format(
            question=questions["text"], punchline=punchline
        )

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

        video_response, _ = results["video"]
        audio_response, _ = results["audio"]
        text_response, _ = results["text"]

        round_history += f"[Video Expert]: {video_response}\n\n"
        round_history += f"[Audio Expert]: {audio_response}\n\n"
        round_history += f"[Text Expert]: {text_response}\n\n"

        print(f"  [Video] {video_response[:80] if video_response else 'None'}...")
        print(f"  [Audio] {audio_response[:80] if audio_response else 'None'}...")
        print(f"  [Text] {text_response[:80] if text_response else 'None'}...")

        conversation_history += round_history

    # Controller makes final decision
    print("--- Controller Final Decision ---")
    decision_query = CONTROLLER_DECISION_PROMPT.format(
        num_rounds=ROUNDS, conversation_history=conversation_history
    )

    # Controller uses text modality with full context for final decision
    final_verdict, usage = call_gemini_with_content(
        query=decision_query,
        images=None,
        audio=None,
        context=None,
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
            "conversation_history": conversation_history,
            "label": sample["label"],
            "method": "controller_agent_query_aug",
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
    output_file = "urfunny_aug_0205.jsonl"
    dataset_path = "./data_raw/urfunny_dataset_test.json"
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
        tracker.print_summary(f"Query Augmentation - Rounds: {ROUNDS}")
