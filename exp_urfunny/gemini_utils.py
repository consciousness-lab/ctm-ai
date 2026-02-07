"""
Utility functions for Gemini multimodal experiments.
Shared across debate, query augmentation, and voting experiments.
"""

import base64
import glob
import json
import os
import statistics
from typing import Dict, List, Optional

import litellm

# ============================================================================
# Environment Setup
# ============================================================================


def check_gemini_api_key():
    """Check if GEMINI_API_KEY environment variable is set."""
    if not os.getenv('GEMINI_API_KEY'):
        raise ValueError('GEMINI_API_KEY environment variable not set')


# ============================================================================
# Data Loading
# ============================================================================


def load_data(file_path: str) -> dict:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def load_processed_keys(output_file: str) -> set:
    """
    Load already processed keys from output file for resume functionality.

    Args:
        output_file: Path to JSONL output file

    Returns:
        Set of already processed test file keys
    """
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


# ============================================================================
# Multimodal Data Processing
# ============================================================================


def load_images_as_base64(
    image_folder: str, max_frames: int = 10
) -> List[Dict[str, str]]:
    """
    Load images from folder and convert to base64 format for litellm.

    Args:
        image_folder: Path to folder containing .jpg images
        max_frames: Maximum number of frames to load (evenly sampled)

    Returns:
        List of image dictionaries in litellm format
    """
    if not image_folder or not os.path.exists(image_folder):
        return []

    image_pattern = os.path.join(image_folder, '*.jpg')
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
            with open(img_path, 'rb') as f:
                img_data = base64.b64encode(f.read()).decode('utf-8')
                images.append(
                    {
                        'type': 'image_url',
                        'image_url': {'url': f'data:image/jpeg;base64,{img_data}'},
                    }
                )
        except Exception as e:
            print(f'Warning: Failed to load image {img_path}: {e}')
            continue

    return images


def prepare_audio_for_gemini(audio_path: str) -> Optional[Dict]:
    """
    Load audio file and convert to base64 format for Gemini.

    Args:
        audio_path: Path to audio file (.mp4)

    Returns:
        Audio dictionary in litellm + Gemini format, or None if file not found
    """
    if not audio_path or not os.path.exists(audio_path):
        return None

    try:
        # Read audio file and encode to base64
        with open(audio_path, 'rb') as f:
            audio_bytes = f.read()

        encoded_data = base64.b64encode(audio_bytes).decode('utf-8')

        # Return in the correct format for litellm + Gemini
        return {
            'type': 'file',
            'file': {
                'file_data': f'data:audio/mp4;base64,{encoded_data}',
            },
        }
    except Exception as e:
        print(f'Warning: Failed to load audio {audio_path}: {e}')
        return None


# ============================================================================
# Gemini API Calls
# ============================================================================


def call_gemini_with_content(
    query: str,
    images: Optional[List[Dict]] = None,
    audio: Optional[Dict] = None,
    context: Optional[str] = None,
    model: str = 'gemini/gemini-2.0-flash-exp',
    temperature: float = 1.0,
) -> tuple[Optional[str], Dict[str, int]]:
    """
    Call Gemini API using litellm with multimodal content.

    Args:
        query: Text query/prompt
        images: List of images in litellm format (from load_images_as_base64)
        audio: Audio dictionary in litellm format (from prepare_audio_for_gemini)
        context: Optional context text to prepend to query
        model: Gemini model name (default: gemini-2.0-flash-exp)
        temperature: Sampling temperature (default: 1.0)

    Returns:
        Tuple of (response_text, usage_dict)
    """
    # Build the content list
    content = []

    # Add text query
    if context:
        text_content = f'### Context:\n{context}\n\n### Query:\n{query}'
    else:
        text_content = f'### Query:\n{query}' if query.strip() else query

    content.append({'type': 'text', 'text': text_content})

    # Add images if provided
    if images:
        content.extend(images)

    # Add audio if provided
    if audio:
        content.append(audio)

    try:
        response = litellm.completion(
            model=model,
            messages=[{'role': 'user', 'content': content}],
            temperature=temperature,
        )

        text = response.choices[0].message.content
        usage = {
            'prompt_tokens': response.usage.prompt_tokens,
            'completion_tokens': response.usage.completion_tokens,
        }

        return text, usage

    except Exception as e:
        print(f'Error calling Gemini API: {e}')
        return None, {'prompt_tokens': 0, 'completion_tokens': 0}


# ============================================================================
# Label Normalization
# ============================================================================


def normalize_label(label) -> str:
    """
    Normalize label to "Yes"/"No" format.

    Args:
        label: Label value (can be int, float, or string)

    Returns:
        Normalized label as "Yes" or "No"
    """
    if isinstance(label, (int, float)):
        # Numeric label: 1 -> "Yes", 0 -> "No"
        return 'Yes' if label == 1 else 'No'
    else:
        # String label: use as is
        return str(label)


# ============================================================================
# Statistics Tracking
# ============================================================================


class StatsTracker:
    """
    Track performance statistics across multiple samples.
    Used for cost estimation and performance analysis.
    """

    def __init__(
        self, cost_input_per_1m: float = 0.075, cost_output_per_1m: float = 0.30
    ):
        """
        Initialize tracker.

        Args:
            cost_input_per_1m: Cost per 1M input tokens (default: Gemini 2.0 Flash)
            cost_output_per_1m: Cost per 1M output tokens (default: Gemini 2.0 Flash)
        """
        self.times = []
        self.input_tokens = []
        self.output_tokens = []
        self.costs = []
        self.api_calls = []
        self.cost_input_per_1m = cost_input_per_1m
        self.cost_output_per_1m = cost_output_per_1m

    def add(self, duration: float, input_tok: int, output_tok: int, num_api_calls: int):
        """Add statistics for one sample."""
        cost = (input_tok / 1_000_000 * self.cost_input_per_1m) + (
            output_tok / 1_000_000 * self.cost_output_per_1m
        )

        self.times.append(duration)
        self.input_tokens.append(input_tok)
        self.output_tokens.append(output_tok)
        self.costs.append(cost)
        self.api_calls.append(num_api_calls)

    def print_summary(self, method_name: str = 'Experiment'):
        """Print summary statistics."""
        if not self.times:
            print('No stats to report.')
            return

        avg_time = statistics.mean(self.times)
        avg_input = statistics.mean(self.input_tokens)
        avg_output = statistics.mean(self.output_tokens)
        avg_cost = statistics.mean(self.costs)
        total_cost = sum(self.costs)
        total_api_calls = sum(self.api_calls)
        avg_api_calls = statistics.mean(self.api_calls)

        print('\n' + '=' * 50)
        print(f'PERFORMANCE & COST SUMMARY ({method_name})')
        print('=' * 50)
        print(f'Total Samples Processed: {len(self.times)}')
        print(f'Total API Calls:         {total_api_calls}')
        print('-' * 40)
        print(f'Average Time per Sample:  {avg_time:.2f} seconds')
        print(f'Average API Calls/Sample: {avg_api_calls:.1f}')
        print(f'Average Input Tokens:     {avg_input:.1f}')
        print(f'Average Output Tokens:    {avg_output:.1f}')
        print(f'Total Input Tokens:       {sum(self.input_tokens)}')
        print(f'Total Output Tokens:      {sum(self.output_tokens)}')
        print('-' * 40)
        print(f'Average Cost per Sample:  ${avg_cost:.6f}')
        print(f'Total Cost for Run:       ${total_cost:.6f}')
        print('=' * 50 + '\n')


# ============================================================================
# File I/O
# ============================================================================


def save_result_to_jsonl(result: dict, output_file: str):
    """
    Save result dictionary to JSONL file.

    Args:
        result: Result dictionary to save
        output_file: Path to output JSONL file
    """
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
