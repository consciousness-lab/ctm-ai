"""
Unified LLM utility module - supports multi-dataset affective computing experiments

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
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import litellm
from dataset_configs import get_dataset_config

# ============================================================================
# Constants
# ============================================================================

DEFAULT_MODELS = {
    'gemini': 'gemini/gemini-2.5-flash-lite',  # Use 1.5-flash (confirmed audio support in official example)
    'qwen': 'qwen/qwen3-omni-flash',  # Fixed: use qwen/ prefix
}

QWEN_API_BASE = 'https://dashscope-intl.aliyuncs.com/compatible-mode/v1'


# ============================================================================
# Environment Setup
# ============================================================================


def check_api_key(
    provider: str = 'gemini', keys: Optional[Sequence[str]] = None
):
    """Check that at least one API key is available.

    If `keys` is provided, use those; otherwise fall back to the standard
    environment variable (GEMINI_API_KEY / DASHSCOPE_API_KEY).
    """
    if keys:
        if any(k and k.strip() for k in keys):
            return
        raise ValueError(f'No valid {provider} API keys provided')
    if provider == 'gemini':
        if not os.getenv('GEMINI_API_KEY'):
            raise ValueError('GEMINI_API_KEY environment variable not set')
    elif provider == 'qwen':
        if not os.getenv('DASHSCOPE_API_KEY'):
            raise ValueError('DASHSCOPE_API_KEY environment variable not set')


class KeyRotator:
    """Thread-safe round-robin API key rotator.

    Each call to `next()` returns the next key in the pool; rotation is
    atomic so concurrent threads always get distinct sequential keys.
    """

    def __init__(self, keys: Sequence[str]):
        self.keys: List[str] = [k.strip() for k in keys if k and k.strip()]
        if not self.keys:
            raise ValueError('KeyRotator: no non-empty keys provided')
        self._idx = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            k = self.keys[self._idx % len(self.keys)]
            self._idx += 1
            return k

    def __len__(self) -> int:
        return len(self.keys)


def parse_api_keys(
    api_keys_arg: Optional[str], provider: str = 'gemini'
) -> List[str]:
    """Parse --api-keys CLI arg (comma-separated) with env var fallback."""
    if api_keys_arg:
        keys = [k.strip() for k in api_keys_arg.split(',') if k.strip()]
        if keys:
            return keys
    env_name = 'GEMINI_API_KEY' if provider == 'gemini' else 'DASHSCOPE_API_KEY'
    env_val = os.getenv(env_name)
    return [env_val] if env_val else []


# ============================================================================
# Data Loading
# ============================================================================


def load_data(file_path: str) -> dict:
    """Load JSON data from file"""
    with open(file_path, 'r', encoding='utf-8') as json_file:
        return json.load(json_file)


def load_processed_keys(output_file: str) -> set:
    """Load already processed keys from JSONL output file for resume"""
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


def get_audio_path(test_file: str, dataset_name: str) -> str:
    """Get audio file path for a test sample"""
    config = get_dataset_config(dataset_name)
    data_paths = config.get_data_paths()
    base_dir = data_paths['audio_only']
    filename = config.get_video_filename(test_file, 'audio')
    return os.path.join(base_dir, filename)


def get_muted_video_path(test_file: str, dataset_name: str) -> str:
    """Get muted video file path for a test sample"""
    config = get_dataset_config(dataset_name)
    data_paths = config.get_data_paths()
    base_dir = data_paths['video_only']
    filename = config.get_video_filename(test_file, 'muted')
    return os.path.join(base_dir, filename)


def get_full_video_path(test_file: str, dataset_name: str) -> str:
    """Get full video (with audio) file path for a test sample"""
    config = get_dataset_config(dataset_name)
    data_paths = config.get_data_paths()
    base_dir = data_paths['full_video']
    filename = config.get_video_filename(test_file, 'full')
    return os.path.join(base_dir, filename)


# ============================================================================
# Media Encoding Helpers
# ============================================================================


def encode_file_base64(file_path: str) -> str:
    """Read file and return base64-encoded string"""
    with open(file_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def make_black_video_with_audio(audio_path: str, output_path: str) -> bool:
    """Convert audio to black-screen video (required for Qwen audio processing)"""
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
    """Get MIME type from video file extension"""
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
    """Get MIME type from audio file extension"""
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
    """Build text-only content"""
    if context:
        text = f'### Context:\n{context}\n\n### Query:\n{query}'
    else:
        text = query
    return [{'type': 'text', 'text': text}]


def build_audio_content(audio_path: str, provider: str = 'gemini') -> List[Dict]:
    """Build audio content block (provider-aware)

    - Gemini: uses 'file' type with audio MIME (correct format for Gemini API)
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
        # Gemini: use 'file' type for audio (matches processor_audio.py implementation)
        mime_type = get_audio_mime_type(audio_path)
        encoded = encode_file_base64(audio_path)
        return [
            {
                'type': 'file',
                'file': {'file_data': f'data:{mime_type};base64,{encoded}'},
            }
        ]


def build_video_content(video_path: str, provider: str = 'gemini') -> List[Dict]:
    """Build video content block (provider-aware)

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
    """Base agent for LLM API calls with provider support.

    Thread-safe: a single agent instance can be shared across worker threads,
    and each call rotates the API key via the shared KeyRotator (if provided).
    """

    AGENT_TYPE = 'base'

    def __init__(
        self,
        provider: str = 'gemini',
        model: Optional[str] = None,
        temperature: float = 1.0,
        api_key: Optional[str] = None,
        key_rotator: Optional[KeyRotator] = None,
    ):
        self.provider = provider
        self.model = model or DEFAULT_MODELS.get(provider, DEFAULT_MODELS['gemini'])
        self.temperature = temperature
        self.api_key = api_key
        self.key_rotator = key_rotator

    def _resolve_api_key(self) -> Optional[str]:
        """Get an API key for this call. Rotates if a KeyRotator is set."""
        if self.key_rotator is not None:
            return self.key_rotator.next()
        if self.api_key is not None:
            return self.api_key
        if self.provider == 'gemini':
            return os.getenv('GEMINI_API_KEY')
        if self.provider == 'qwen':
            return os.getenv('DASHSCOPE_API_KEY')
        return None

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        """Build content list. Override in subclasses"""
        raise NotImplementedError

    def call(
        self, query: str, max_retries: int = 3, **kwargs: Any
    ) -> Tuple[Optional[str], Dict[str, int]]:
        """Make an LLM API call with the agent's modality, retry up to max_retries on failure.

        On retry, the API key is re-resolved — so if multiple keys are provided
        via KeyRotator, a failing key will be rotated away on the next attempt.
        """
        content = self._build_content(query, **kwargs)

        for attempt in range(1, max_retries + 1):
            try:
                call_kwargs = {
                    'model': self.model,
                    'messages': [{'role': 'user', 'content': content}],
                    'temperature': self.temperature,
                    # Hard per-request cap so a stalled Gemini multimodal
                    # request can't wedge the whole pipeline.
                    'timeout': 90,
                    'api_key': self._resolve_api_key(),
                }

                if self.provider == 'qwen':
                    call_kwargs['api_base'] = QWEN_API_BASE
                    call_kwargs['modalities'] = ['text']

                response = litellm.completion(**call_kwargs)
                text = response.choices[0].message.content
                usage = {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                }

                if text:
                    return text, usage

                # Response is None or empty, retry
                print(
                    f'Warning: empty response from {self.provider} ({self.AGENT_TYPE}), '
                    f'attempt {attempt}/{max_retries}'
                )

            except Exception as e:
                print(
                    f'Error calling {self.provider} API ({self.AGENT_TYPE}), '
                    f'attempt {attempt}/{max_retries}: {e}'
                )

        # All retries exhausted
        return None, {'prompt_tokens': 0, 'completion_tokens': 0}

    def __repr__(self):
        return (
            f'{self.__class__.__name__}(provider={self.provider}, model={self.model})'
        )


class TextAgent(BaseAgent):
    """Text-only input agent

    kwargs:
        context (str): Text context to include with the query
    """

    AGENT_TYPE = 'text'

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        context = kwargs.get('context')
        return build_text_content(query, context)


class AudioAgent(BaseAgent):
    """Audio-only input agent (no video)

    kwargs:
        audio_path (str): Path to the audio file (.mp4)
    """

    AGENT_TYPE = 'audio'

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        audio_path = kwargs.get('audio_path')
        content = build_text_content(query)
        audio_blocks = build_audio_content(audio_path, self.provider)
        content.extend(audio_blocks)
        return content


class VideoAgent(BaseAgent):
    """Video-only input agent (muted video, no audio)

    kwargs:
        video_path (str): Path to the muted video file (.mp4)
    """

    AGENT_TYPE = 'video'

    def _build_content(self, query: str, **kwargs: Any) -> List[Dict]:
        video_path = kwargs.get('video_path')
        content = build_text_content(query)
        video_blocks = build_video_content(video_path, self.provider)
        content.extend(video_blocks)
        return content


class MultimodalAgent(BaseAgent):
    """Full multimodal agent (text + video with audio)

    Uses the full video file (which contains both video and audio streams)

    kwargs:
        context (str): Text context to include with the query
        video_path (str): Path to the full video file with audio (.mp4)
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


def create_agent(
    agent_type: str,
    provider: str = 'gemini',
    model: Optional[str] = None,
    temperature: float = 1.0,
    api_key: Optional[str] = None,
    key_rotator: Optional[KeyRotator] = None,
) -> BaseAgent:
    """Factory function to create an agent by type"""
    if agent_type not in AGENT_CLASSES:
        raise ValueError(
            f'Unknown agent type: {agent_type}. '
            f'Choose from: {list(AGENT_CLASSES.keys())}'
        )
    return AGENT_CLASSES[agent_type](
        provider=provider,
        model=model,
        temperature=temperature,
        api_key=api_key,
        key_rotator=key_rotator,
    )


def load_sample_inputs(
    test_file: str, dataset: dict, dataset_name: str
) -> Dict[str, Any]:
    """Load all inputs for a sample: text, audio, video paths and metadata.

    Returns a dict with:
        - target_sentence: the target text from sample
        - system_prompt: task-specific system prompt
        - label: ground truth label
        - full_video_path: path to full video (with audio)
        - muted_video_path: path to muted video (no audio)
        - audio_path: path to audio file
        - config: dataset config object
    """
    config = get_dataset_config(dataset_name)
    sample = dataset[test_file]

    return {
        'target_sentence': config.get_text_field(sample),
        'system_prompt': config.get_system_prompt(),
        'label': config.get_label_field(sample),
        'full_video_path': get_full_video_path(test_file, dataset_name),
        'muted_video_path': get_muted_video_path(test_file, dataset_name),
        'audio_path': get_audio_path(test_file, dataset_name),
        'config': config,
    }


# ============================================================================
# Label Normalization
# ============================================================================


def normalize_label(label) -> str:
    """Normalize label to 'Yes'/'No' format"""
    if isinstance(label, bool):
        return 'Yes' if label else 'No'
    if isinstance(label, (int, float)):
        return 'Yes' if label == 1 else 'No'
    if isinstance(label, str):
        lower = label.strip().lower()
        if lower in ('yes', 'true', '1'):
            return 'Yes'
        if lower in ('no', 'false', '0'):
            return 'No'
        return label
    return str(label)


# ============================================================================
# Statistics Tracking
# ============================================================================


class StatsTracker:
    """Track performance statistics across multiple samples (thread-safe)"""

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
        self._lock = threading.Lock()

    def add(
        self,
        duration: float,
        input_tok: int,
        output_tok: int,
        num_api_calls: int,
    ):
        """Add statistics for one sample"""
        cost = (input_tok / 1_000_000 * self.cost_input_per_1m) + (
            output_tok / 1_000_000 * self.cost_output_per_1m
        )
        with self._lock:
            self.times.append(duration)
            self.input_tokens.append(input_tok)
            self.output_tokens.append(output_tok)
            self.costs.append(cost)
            self.api_calls.append(num_api_calls)

    def print_summary(self, method_name: str = 'Experiment'):
        """Print summary statistics"""
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


_jsonl_write_locks: Dict[str, threading.Lock] = {}
_jsonl_write_locks_lock = threading.Lock()


def _get_jsonl_lock(output_file: str) -> threading.Lock:
    """Return a per-file threading lock (created lazily)."""
    with _jsonl_write_locks_lock:
        lock = _jsonl_write_locks.get(output_file)
        if lock is None:
            lock = threading.Lock()
            _jsonl_write_locks[output_file] = lock
        return lock


def save_result_to_jsonl(result: dict, output_file: str):
    """Append result dictionary to JSONL file (thread-safe per output_file)."""
    lock = _get_jsonl_lock(output_file)
    line = json.dumps(result, ensure_ascii=False) + '\n'
    with lock:
        with open(output_file, 'a', encoding='utf-8') as f:
            f.write(line)


# ============================================================================
# Common CLI Arguments
# ============================================================================


def add_common_args(parser):
    """Add common CLI arguments to an argparse parser"""
    parser.add_argument(
        '--dataset_name',
        type=str,
        default='urfunny',
        choices=['urfunny', 'mustard'],
        help='Dataset name (default: urfunny)',
    )
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
        default=None,
        help='Path to dataset JSON file (default: auto based on dataset_name)',
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
    parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Number of samples to process concurrently (default: 4)',
    )
    parser.add_argument(
        '--api-keys',
        type=str,
        default=None,
        help=(
            'Comma-separated list of API keys to rotate across. '
            'If omitted, falls back to the GEMINI_API_KEY / DASHSCOPE_API_KEY env var.'
        ),
    )
    return parser
