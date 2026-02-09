"""
Utility module for multimodal LLM experiments.

Supports both Gemini and Qwen providers with 4 agent types:
  - TextAgent: text-only input
  - AudioAgent: audio-only input (no video)
  - VideoAgent: video-only input (no audio, muted video)
  - MultimodalAgent: all modalities (text + video with audio)

Shared across debate, query augmentation, voting, and baseline experiments.
"""

import base64
import json
import os
import statistics
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import litellm

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODELS = {
    'gemini': 'gemini/gemini-2.0-flash-lite',
    'qwen': 'openai/qwen2.5-omni-7b',
}

# Default data paths (relative to exp_mustard/)
DATA_PATHS = {
    'full_video': 'mmsd_raw_data/utterances_final',  # video + audio
    'audio_only': 'mustard_audios',  # audio only
    'video_only': 'mustard_muted_videos',  # muted video only
}


# ============================================================================
# Environment Setup
# ============================================================================


def check_api_key(provider: str = 'gemini'):
    """Check if required API key environment variable is set."""
    if provider == 'gemini':
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError('GEMINI_API_KEY environment variable not set')
    elif provider == 'qwen':
        # Qwen may use DASHSCOPE_API_KEY or OPENAI_API_KEY depending on setup
        pass


# ============================================================================
# Data Loading
# ============================================================================


def load_data(file_path: str) -> dict:
    """Load JSON data from file."""
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def load_processed_keys(output_file: str) -> set:
    """Load already processed keys from JSONL output file for resume."""
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
# File Path Helpers
# ============================================================================


def get_audio_path(test_file: str, base_dir: str = DATA_PATHS['audio_only']) -> str:
    """Get audio file path for a test sample."""
    return os.path.join(base_dir, f'{test_file}_audio.mp4')


def get_muted_video_path(
    test_file: str, base_dir: str = DATA_PATHS['video_only']
) -> str:
    """Get muted video file path for a test sample."""
    return os.path.join(base_dir, f'{test_file}.mp4')


def get_full_video_path(
    test_file: str, base_dir: str = DATA_PATHS['full_video']
) -> str:
    """Get full video (with audio) file path for a test sample."""
    return os.path.join(base_dir, f'{test_file}.mp4')


# ============================================================================
# Media Encoding Helpers
# ============================================================================


def encode_file_base64(file_path: str) -> str:
    """Read file and return base64-encoded string."""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def make_black_video_with_audio(audio_path: str, output_path: str) -> bool:
    """Convert audio to black-screen video (required for Qwen audio processing)."""
    cmd = [
        'ffmpeg',
        '-y',
        '-f',
        'lavfi',
        '-i',
        'color=c=black:s=320x240:r=1',
        '-i',
        audio_path,
        '-shortest',
        '-c:v',
        'libx264',
        '-tune',
        'stillimage',
        '-c:a',
        'aac',
        '-pix_fmt',
        'yuv420p',
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def get_video_mime_type(video_path: str) -> str:
    """Get MIME type from video file extension."""
    ext = os.path.splitext(video_path)[1].lower()
    mime_map = {
        '.mp4': 'video/mp4',
        '.mpeg': 'video/mpeg',
        '.mov': 'video/mov',
        '.avi': 'video/avi',
        '.webm': 'video/webm',
        '.wmv': 'video/wmv',
        '.3gp': 'video/3gpp',
    }
    return mime_map.get(ext, 'video/mp4')


def get_audio_mime_type(audio_path: str) -> str:
    """Get MIME type from audio file extension."""
    ext = os.path.splitext(audio_path)[1].lower().lstrip('.')
    mime_map = {
        'mp3': 'audio/mp3',
        'wav': 'audio/wav',
        'aac': 'audio/aac',
        'flac': 'audio/flac',
        'mp4': 'audio/mp4',
    }
    return mime_map.get(ext, 'audio/mp4')


# ============================================================================
# Provider-Aware Content Building
# ============================================================================


def build_text_content(query: str, context: Optional[str] = None) -> List[Dict]:
    """Build text-only content."""
    if context:
        text = f'### Context:\n{context}\n\n### Query:\n{query}'
    else:
        text = query
    return [{'type': 'text', 'text': text}]


def build_audio_content(audio_path: str, provider: str = 'gemini') -> List[Dict]:
    """Build audio content block (provider-aware).

    - Gemini: uses 'image_url' type with audio MIME (litellm uses this for all media)
    - Qwen: embeds audio in a black-screen video via ffmpeg
    """
    if not audio_path or not os.path.exists(audio_path):
        return []

    if provider == 'qwen':
        tmp_video = tempfile.mktemp(suffix='.mp4')
        try:
            if not make_black_video_with_audio(audio_path, tmp_video):
                print(f'Warning: ffmpeg failed for {audio_path}')
                return []
            encoded = encode_file_base64(tmp_video)
        finally:
            if os.path.exists(tmp_video):
                os.unlink(tmp_video)
        return [
            {
                'type': 'video_url',
                'video_url': {'url': f'data:video/mp4;base64,{encoded}'},
            }
        ]
    else:
        # Gemini via litellm: use image_url type for audio (litellm convention)
        mime_type = get_audio_mime_type(audio_path)
        encoded = encode_file_base64(audio_path)
        return [
            {
                'type': 'image_url',
                'image_url': {'url': f'data:{mime_type};base64,{encoded}'},
            }
        ]


def build_video_content(video_path: str, provider: str = 'gemini') -> List[Dict]:
    """Build video content block (provider-aware).

    - Gemini: uses 'image_url' type with video MIME
    - Qwen: uses 'video_url' type
    """
    if not video_path or not os.path.exists(video_path):
        return []

    mime_type = get_video_mime_type(video_path)
    encoded = encode_file_base64(video_path)
    data_url = f'data:{mime_type};base64,{encoded}'

    if provider == 'qwen':
        return [{'type': 'video_url', 'video_url': {'url': data_url}}]
    else:
        # Gemini uses image_url type for video
        return [{'type': 'image_url', 'image_url': {'url': data_url}}]


# ============================================================================
# Agent Classes
# ============================================================================


class BaseAgent:
    """Base agent for LLM API calls with provider support."""

    AGENT_TYPE = 'base'

    def __init__(
        self,
        provider: str = 'gemini',
        model: Optional[str] = None,
        temperature: float = 1.0,
    ):
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS['gemini'])
        self.temperature = temperature

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        """Build content list. Override in subclasses."""
        raise NotImplementedError

    def call(self, query: str, **kwargs: Any) -> Tuple[Optional[str], Dict[str, int]]:
        """Make an LLM API call with the agent's modality."""
        content = self._build_content(query, **kwargs)

        try:
            response = litellm.completion(
                model=self.model,
                messages=[{'role': 'user', 'content': content}],
                temperature=self.temperature,
            )
            text = response.choices[0].message.content
            usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
            }
            return text, usage

        except Exception as e:
            print(f'Error calling {self.provider} API ({self.AGENT_TYPE}): {e}')
            return None, {'prompt_tokens': 0, 'completion_tokens': 0}

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(provider={self.provider}, model={self.model})'
        )


class TextAgent(BaseAgent):
    """Text-only input agent.

    kwargs:
        context (str): Text context to include with the query.
    """

    AGENT_TYPE = 'text'

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        context = kwargs.get('context')
        return build_text_content(query, context)


class AudioAgent(BaseAgent):
    """Audio-only input agent (no video).

    kwargs:
        audio_path (str): Path to the audio file (.mp4).
    """

    AGENT_TYPE = 'audio'

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        audio_path = kwargs.get('audio_path')
        content = build_text_content(query)
        audio_blocks = build_audio_content(audio_path, self.provider)
        content.extend(audio_blocks)
        return content


class VideoAgent(BaseAgent):
    """Video-only input agent (muted video, no audio).

    kwargs:
        video_path (str): Path to the muted video file (.mp4).
    """

    AGENT_TYPE = 'video'

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        video_path = kwargs.get('video_path')
        content = build_text_content(query)
        video_blocks = build_video_content(video_path, self.provider)
        content.extend(video_blocks)
        return content


class MultimodalAgent(BaseAgent):
    """Full multimodal agent (text + video with audio).

    Uses the full video file (which contains both video and audio streams).

    kwargs:
        context (str): Text context to include with the query.
        video_path (str): Path to the full video file with audio (.mp4).
    """

    AGENT_TYPE = 'multimodal'

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        context = kwargs.get('context')
        video_path = kwargs.get('video_path')
        content = build_text_content(query, context)
        # Full video contains both video and audio streams
        video_blocks = build_video_content(video_path, self.provider)
        content.extend(video_blocks)
        return content


# ============================================================================
# Agent Factory
# ============================================================================

AGENT_CLASSES = {
    'text': TextAgent,
    'audio': AudioAgent,
    'video': VideoAgent,
    'multimodal': MultimodalAgent,
}

AGENT_ROLES = {
    'text': 'Text Analysis',
    'audio': 'Audio Analysis',
    'video': 'Video Analysis',
    'multimodal': 'Multimodal Analysis',
}


def create_agent(
    agent_type: str,
    provider: str = 'gemini',
    model: Optional[str] = None,
    temperature: float = 1.0,
) -> BaseAgent:
    """Factory function to create an agent by type."""
    if agent_type not in AGENT_CLASSES:
        raise ValueError(
            f'Unknown agent type: {agent_type}. '
            f'Choose from: {list(AGENT_CLASSES.keys())}'
        )
    return AGENT_CLASSES[agent_type](
        provider=provider, model=model, temperature=temperature
    )


def create_all_agents(
    provider: str = 'gemini',
    model: Optional[str] = None,
    temperature: float = 1.0,
) -> Dict[str, BaseAgent]:
    """Create all 4 agents with the same provider/model/temperature."""
    return {
        name: create_agent(name, provider, model, temperature) for name in AGENT_CLASSES
    }


def get_agent_kwargs(test_file: str, dataset: dict) -> Dict[str, Dict[str, Any]]:
    """Build kwargs for each agent type given a test sample.

    Returns dict mapping agent_type -> kwargs for agent.call().
    """
    target_sentence = dataset[test_file]['utterance']
    text_list = dataset[test_file]['context'].copy()
    text_list.append(target_sentence)
    full_context = '\n'.join(text_list)

    return {
        'text': {'context': full_context},
        'audio': {'audio_path': get_audio_path(test_file)},
        'video': {'video_path': get_muted_video_path(test_file)},
        'multimodal': {
            'context': full_context,
            'video_path': get_full_video_path(test_file),
        },
    }


# ============================================================================
# Label Normalization
# ============================================================================


def normalize_label(label) -> str:
    """Normalize label to 'Yes'/'No' format."""
    if isinstance(label, (int, float)):
        return 'Yes' if label == 1 else 'No'
    return str(label)


# ============================================================================
# Statistics Tracking
# ============================================================================


class StatsTracker:
    """Track performance statistics across multiple samples."""

    def __init__(
        self,
        cost_input_per_1m: float = 0.075,
        cost_output_per_1m: float = 0.30,
    ):
        self.times: List[float] = []
        self.input_tokens: List[int] = []
        self.output_tokens: List[int] = []
        self.costs: List[float] = []
        self.api_calls: List[int] = []
        self.cost_input_per_1m = cost_input_per_1m
        self.cost_output_per_1m = cost_output_per_1m

    def add(
        self,
        duration: float,
        input_tok: int,
        output_tok: int,
        num_api_calls: int,
    ):
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
    """Append result dictionary to JSONL file."""
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


# ============================================================================
# Common CLI Arguments
# ============================================================================


def add_common_args(parser):
    """Add common CLI arguments to an argparse parser."""
    parser.add_argument(
        '--provider',
        type=str,
        default='gemini',
        choices=['gemini', 'qwen'],
        help='LLM provider (default: gemini)',
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Model name for litellm (default: auto based on provider)',
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mustard_dataset/mustard_dataset_test.json',
        help='Path to dataset JSON file',
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSONL file path',
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Sampling temperature (default: 1.0)',
    )
    return parser
