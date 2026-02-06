import json
import sys
import time
import statistics
import os
import glob
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional

import litellm

sys.path.append("..")

# Set Gemini API key from environment variable
if not os.getenv("GEMINI_API_KEY"):
    raise ValueError("GEMINI_API_KEY environment variable not set")

litellm.set_verbose = False  # Set to True for debugging

ROUNDS = 3  # Number of debate rounds (initial + refinements)

# Modality-specific agent prompts for humor detection - using original balanced prompt style
VIDEO_AGENT_INIT = (
    "You are a Video Analysis Expert. You will analyze whether the person in the video is being humorous or not.\n"
    "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
)

AUDIO_AGENT_INIT = (
    "You are an Audio Analysis Expert. You will analyze whether the person in the video is being humorous or not.\n"
    "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
)

TEXT_AGENT_INIT = (
    "You are a Text Analysis Expert. You will be given a punchline that was said by a person, and analysis whether the person is being humorous or not.\n"
    "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
)

VIDEO_AGENT_REFINE = (
    "You are a Video Analysis Expert. You previously analyzed the video frames of this person.\n"
    "Your previous answer: {own_answer}\n\n"
    "Here are the analyses from other experts:\n"
    "- Audio Expert: {audio_answer}\n"
    "- Text Expert: {text_answer}\n\n"
    "First, consider their perspectives and re-examine the video evidence carefully.\n"
    "Then, determine if this person is being humorous or not.\n"
    "End your response with 'My Answer: Yes' or 'My Answer: No'."
)

AUDIO_AGENT_REFINE = (
    "You are an Audio Analysis Expert. You previously analyzed the audio of this person speaking.\n"
    "Your previous answer: {own_answer}\n\n"
    "Here are the analyses from other experts:\n"
    "- Video Expert: {video_answer}\n"
    "- Text Expert: {text_answer}\n\n"
    "First, consider their perspectives and re-examine the audio evidence carefully.\n"
    "Then, determine if this person is being humorous or not.\n"
    "End your response with 'My Answer: Yes' or 'My Answer: No'."
)

TEXT_AGENT_REFINE = (
    "You are a Text Analysis Expert. You previously analyzed the punchline that was said by this person.\n"
    "Your previous answer: {own_answer}\n\n"
    "Here are the analyses from other experts:\n"
    "- Video Expert: {video_answer}\n"
    "- Audio Expert: {audio_answer}\n\n"
    "First, consider their perspectives and re-examine the text evidence of the punchline carefully.\n"
    "Then, determine if this person is being humorous or not.\n"
    "End your response with 'My Answer: Yes' or 'My Answer: No'."
)

JUDGE_PROMPT = (
    "You are an impartial Judge. Three experts have debated whether this person is being humorous or not.\n"
    "Here is the discussion:\n\n{debate_history}\n\n"
    "Based on all evidence from the video, audio, and text analyses, determine if this person is humorous or not.\n"
    "Your answer must start with 'Yes' or 'No', followed by your reasoning."
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
        cost = (input_tok / 1_000_000 * COST_INPUT_PER_1M) + (
            output_tok / 1_000_000 * COST_OUTPUT_PER_1M
        )

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

        print("\n" + "=" * 50)
        print("PERFORMANCE & COST SUMMARY (Multimodal Debate)")
        print(f"  Rounds: {ROUNDS} | Agents: Video, Audio, Text + Judge")
        print("=" * 50)
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
        print("=" * 50 + "\n")


tracker = StatsTracker()


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def load_processed_keys(output_file):
    """Load already processed keys from output file for resume functionality."""
    processed = set()
    try:
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    result = json.loads(line)
                    processed.update(result.keys())
    except FileNotFoundError:
        pass
    return processed


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


def load_images_as_base64(
    image_folder: str, max_frames: int = 10
) -> List[Dict[str, str]]:
    """Load images from folder and convert to base64 format for litellm."""
    if not image_folder or not os.path.exists(image_folder):
        return []

    image_pattern = os.path.join(image_folder, "*.jpg")
    image_paths = sorted(glob.glob(image_pattern))

    if not image_paths:
        return []

    # Sample frames evenly if too many
    if len(image_paths) > max_frames:
        step = len(image_paths) / max_frames
        image_paths = [image_paths[int(i * step)] for i in range(max_frames)]

    images = []
    for img_path in image_paths:
        try:
            with open(img_path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
                images.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_data}"},
                    }
                )
        except Exception as e:
            print(f"Warning: Failed to load image {img_path}: {e}")
            continue

    return images


def prepare_audio_for_gemini(audio_path: str) -> Optional[Dict]:
    """Load audio file and convert to base64 format for Gemini."""
    if not audio_path or not os.path.exists(audio_path):
        return None

    try:
        # Read audio file and encode to base64
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()

        encoded_data = base64.b64encode(audio_bytes).decode("utf-8")

        # Return in the correct format for litellm + Gemini
        return {
            "type": "file",
            "file": {
                "file_data": f"data:audio/mp4;base64,{encoded_data}",
            },
        }
    except Exception as e:
        print(f"Warning: Failed to load audio {audio_path}: {e}")
        return None


def call_gemini_with_content(
    query: str,
    images: Optional[List[Dict]] = None,
    audio: Optional[Dict] = None,
    context: Optional[str] = None,
    model: str = "gemini/gemini-2.0-flash-exp",
) -> tuple[Optional[str], Dict[str, int]]:
    """Call Gemini API using litellm with multimodal content."""

    # Build the content list
    content = []

    # Add text query
    if context:
        text_content = f"### Context:\n{context}\n\n### Query:\n{query}"
    else:
        text_content = f"### Query:\n{query}"

    content.append({"type": "text", "text": text_content})

    # Add images if provided
    if images:
        content.extend(images)

    # Add audio if provided (already in correct format from prepare_audio_for_gemini)
    if audio:
        content.append(audio)

    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=1.0,
        )

        text = response.choices[0].message.content
        usage = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

        return text, usage

    except Exception as e:
        print(f"Error calling Gemini API: {e}")
        return None, {"prompt_tokens": 0, "completion_tokens": 0}


def run_instance(test_file, dataset, output_file="urfunny_debate_0205.jsonl"):
    start_time = time.time()
    total_prompt_tokens = 0
    total_completion_tokens = 0
    num_api_calls = 0

    sample = dataset[test_file]
    punchline = sample["punchline_sentence"]

    text_list = sample["context_sentences"].copy()
    text_list.append(punchline)
    fullContext = "\n".join(text_list)

    audio_path = f"test_inputs/urfunny_audios/{test_file}_audio.mp4"
    video_frames_path = f"test_inputs/urfunny_frames/{test_file}_frames"

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
            )
        elif agent_type == "audio":
            # Audio agent: only audio
            response, usage = call_gemini_with_content(
                query=query,
                images=None,
                audio=audio_file,
                context=None,
                model=model_name,
            )
        elif agent_type == "text":
            # Text agent: only text context
            response, usage = call_gemini_with_content(
                query=query,
                images=None,
                audio=None,
                context=fullContext,
                model=model_name,
            )
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")

        return agent_type, response, usage

    for round_num in range(1, ROUNDS + 1):
        print(f"--- Round {round_num} ---")
        round_history = f"=== Round {round_num} ===\n\n"

        # Prepare queries for all three agents
        # Only text agent receives punchline text
        if round_num == 1:
            video_query = VIDEO_AGENT_INIT
            audio_query = AUDIO_AGENT_INIT
            text_query = f"{TEXT_AGENT_INIT}\n\npunchline: '{punchline}'"
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
            text_query = f"{TEXT_AGENT_REFINE.format(own_answer=text_answer, video_answer=video_answer, audio_answer=audio_answer)}\n\npunchline: '{punchline}'"

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
    judge_query = f"{JUDGE_PROMPT.format(debate_history=debate_history)}\n\npunchline: '{punchline}'"
    # Judge uses text modality with context
    final_verdict, usage = call_gemini_with_content(
        query=judge_query,
        images=None,
        audio=None,
        context=fullContext,
        model=model_name,
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
            "label": sample["label"],
            "method": "multimodal_debate_3agents",
            "usage": {
                "prompt_tokens": total_prompt_tokens,
                "completion_tokens": total_completion_tokens,
                "api_calls": num_api_calls,
            },
            "latency": duration,
        }
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    output_file = "urfunny_debate_0205.jsonl"
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
        tracker.print_summary()
