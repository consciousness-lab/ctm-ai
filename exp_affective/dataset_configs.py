"""
Dataset configuration file - supports different affective computing datasets

Each dataset configuration contains:
- Dataset name and task type
- Text field mappings
- Label field names
- File path prefixes
- Task-related prompts and queries

Per-modality system prompts for the multi-agent baselines (debate / orchestra /
ensemble) are loaded from the CTM config JSONs under ``ctm_conf/`` so that all
methods share the same source of truth for modality expertise. Prompts in this
file are intentionally thin task wrappers (Yes/No format, refine/question
templates, controller/judge meta-prompts) that get concatenated with those CTM
system prompts at runtime.
"""

import json
import os
import warnings
from typing import Any, Dict, Optional

# ---------------------------------------------------------------------------
# CTM config loader helpers
# ---------------------------------------------------------------------------

_CTM_CONF_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'ctm_conf'
)

_PROCESSOR_TO_MODALITY = {
    'video_processor': 'video',
    'audio_processor': 'audio',
    'language_processor': 'text',
}

# Shared commit instruction applied to all three modality agents in the
# ensemble / orchestra task wrappers. The CTM per-modality system_prompts
# tune agents for CTM's weighted uptree fusion — they encode confidence
# thresholds so low-confidence votes get small weight. Majority voting /
# controller / judge aggregation have no such weighting, so we explicitly
# force each agent into a forced-choice binary decision and tell it to
# treat confidence rubrics as internal reasoning only (no abstention).
# Empirically this produced urfunny ensemble F1_pos=0.6875 (+1.5 over unified)
# in 100-sample experiments — slightly worse than pure-majority-vote on
# F1_macro but better on F1_pos, which is the standard metric for humor/
# sarcasm positive-class detection.
_COMMIT_INSTRUCTION = (
    'This is a forced-choice binary decision task: you must output either Yes or No. '
    'Treat any confidence-threshold rubric in your guidelines as internal reasoning '
    'only — it must NOT bias you toward "No" at moderate confidence. '
    'Pick Yes whenever a plausible humor/sarcasm technique could apply, even if the '
    'evidence is subtle; pick No only when the content is clearly straightforward with '
    'no reasonable humor/sarcasm interpretation. Do not abstain, do not hedge.'
)

_TEXT_COMMIT_INSTRUCTION = _COMMIT_INSTRUCTION
_AV_COMMIT_INSTRUCTION = _COMMIT_INSTRUCTION


def _resolve_ctm_config_path(path: str) -> str:
    """Resolve a CTM config path, accepting either a bare filename or absolute path."""
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    return os.path.join(_CTM_CONF_DIR, path)


def load_ctm_modality_prompts(config_path: str) -> Dict[str, str]:
    """Load per-modality system prompts from a CTM config JSON file.

    Returns a dict keyed by ``'video'`` / ``'audio'`` / ``'text'``. Only
    modalities that are present in the given config (and have a non-empty
    ``system_prompt``) appear in the returned dict.
    """
    resolved = _resolve_ctm_config_path(config_path)
    with open(resolved, 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    processors = cfg.get('processors_config', {}) or {}
    prompts: Dict[str, str] = {}
    for proc_key, modality in _PROCESSOR_TO_MODALITY.items():
        proc_cfg = processors.get(proc_key) or {}
        sp = proc_cfg.get('system_prompt')
        if sp:
            prompts[modality] = sp
    return prompts


class DatasetConfig:
    """Base class for dataset configuration"""

    # Per-provider default CTM config filenames (relative to ctm_conf/).
    # Subclasses override this. ``None`` value means no default is defined for
    # that provider and a fallback has to be resolved explicitly.
    DEFAULT_CTM_CONFIGS: Dict[str, Optional[str]] = {}

    # Per-modality fallback CTM configs, used when a modality is missing from
    # the primary config for a given provider (e.g. mustard+qwen has no
    # audio_processor, so we borrow the audio prompt from the gemini variant).
    # Keyed by provider -> modality -> config filename.
    MODALITY_FALLBACK_CONFIGS: Dict[str, Dict[str, str]] = {}

    def __init__(self, name: str, task_type: str):
        self.name = name
        self.task_type = task_type

    def get_text_field(self, sample: dict) -> str:
        """Get the target text from sample"""
        raise NotImplementedError

    def get_context_field(self, sample: dict) -> str:
        """Get the context text from sample"""
        raise NotImplementedError

    def get_label_field(self, sample: dict):
        """Get the label from sample"""
        raise NotImplementedError

    def get_video_filename(self, test_file: str, video_type: str) -> str:
        """Get video filename

        Args:
            test_file: Test file ID
            video_type: 'full' (full video), 'muted' (muted video), 'audio' (audio)
        """
        raise NotImplementedError

    def get_data_paths(self) -> Dict[str, str]:
        """Get data path configuration"""
        raise NotImplementedError

    def get_default_dataset_path(self) -> str:
        """Get default dataset path"""
        raise NotImplementedError

    def get_task_query(self) -> str:
        """Get task query statement"""
        raise NotImplementedError

    def get_system_prompt(self) -> str:
        """Get system prompt"""
        raise NotImplementedError

    def get_debate_prompts(self) -> Dict[str, str]:
        """Get task-wrapper prompts for debate experiments.

        These are thin task instructions (Yes/No format, refine template,
        judge prompt). Per-modality expertise lives in the CTM config system
        prompts returned by :meth:`get_ctm_modality_prompts`.
        """
        raise NotImplementedError

    def get_query_aug_prompts(self) -> Dict[str, str]:
        """Get task-wrapper prompts for query augmentation / orchestra.

        Controller meta-prompts and per-expert question wrappers live here;
        per-modality expertise comes from the CTM config.
        """
        raise NotImplementedError

    def get_ensemble_task_instructions(self) -> Dict[str, str]:
        """Get per-modality task wrappers for the modality ensemble baseline."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # CTM config integration
    # ------------------------------------------------------------------

    def get_ctm_config_path(self, provider: str) -> Optional[str]:
        """Return the default CTM config filename for this dataset/provider."""
        return self.DEFAULT_CTM_CONFIGS.get(provider)

    def get_ctm_modality_prompts(
        self,
        provider: str,
        override_path: Optional[str] = None,
    ) -> Dict[str, str]:
        """Load per-modality CTM system prompts for the given provider.

        If ``override_path`` is supplied, it is used as the primary config. The
        primary config is consulted first; any modality missing from it is
        filled in from the provider's ``MODALITY_FALLBACK_CONFIGS`` entry (if
        defined). A warning is emitted for each fallback or missing modality.
        """
        primary_path = override_path or self.get_ctm_config_path(provider)
        prompts: Dict[str, str] = {}
        if primary_path:
            try:
                prompts.update(load_ctm_modality_prompts(primary_path))
            except FileNotFoundError:
                warnings.warn(
                    f'CTM config not found: {primary_path}. '
                    f'No CTM modality prompts loaded from primary config.'
                )

        fallback_map = self.MODALITY_FALLBACK_CONFIGS.get(provider, {})
        for modality in ('video', 'audio', 'text'):
            if modality in prompts:
                continue
            fallback_path = fallback_map.get(modality)
            if not fallback_path:
                continue
            try:
                fb_prompts = load_ctm_modality_prompts(fallback_path)
            except FileNotFoundError:
                warnings.warn(
                    f'CTM fallback config not found: {fallback_path}. '
                    f'Leaving {modality} modality prompt empty.'
                )
                continue
            if modality in fb_prompts:
                warnings.warn(
                    f'[{self.name}/{provider}] {modality} modality prompt not '
                    f'present in primary CTM config; borrowing from '
                    f'{os.path.basename(fallback_path)} for fair comparison.'
                )
                prompts[modality] = fb_prompts[modality]
        return prompts


class URFunnyConfig(DatasetConfig):
    """URFunny dataset configuration - humor detection"""

    DEFAULT_CTM_CONFIGS = {
        'qwen': 'urfunny_qwen3vl_thinking_v28_config.json',
        'gemini': 'urfunny_test_gemini_v28_config.json',
    }

    def __init__(self):
        super().__init__('urfunny', 'humor_detection')

    def get_text_field(self, sample: dict) -> str:
        return sample['punchline_sentence']

    def get_context_field(self, sample: dict) -> str:
        """Return full context (context + punchline)"""
        context_sentences = sample.get('context_sentences', [])
        punchline = sample['punchline_sentence']
        full_context = '\n'.join(context_sentences)
        if full_context:
            full_context += f'\n\nPunchline: {punchline}'
        else:
            full_context = punchline
        return full_context

    def get_label_field(self, sample: dict):
        return sample['label']

    def get_video_filename(self, test_file: str, video_type: str) -> str:
        if video_type == 'full':
            return f'{test_file}.mp4'
        elif video_type == 'muted':
            return f'{test_file}.mp4'
        elif video_type == 'audio':
            return f'{test_file}_audio.mp4'
        else:
            raise ValueError(f'Unknown video type: {video_type}')

    def get_data_paths(self) -> Dict[str, str]:
        return {
            'full_video': 'data/urfunny/urfunny_videos',
            'audio_only': 'data/urfunny/urfunny_audios',
            'video_only': 'data/urfunny/urfunny_muted_videos',
        }

    def get_default_dataset_path(self) -> str:
        return 'data/urfunny/data_raw/urfunny_dataset_test.json'

    def get_task_query(self) -> str:
        return 'Is the person being humorous or not?'

    def get_system_prompt(self) -> str:
        return (
            'Please analyze the inputs provided to determine if the person is being humorous or not.\n'
            "If you think the input includes exaggerated description or it is expressing sarcastic meaning, please answer 'Yes'. "
            "If you think the input is neutral or just common meaning, please answer 'No'. "
            "Your answer should begin with either 'Yes' or 'No', followed by your reasoning. "
            'If you are not sure, please provide your best guess and do not say that you are not sure.'
        )

    def get_debate_prompts(self) -> Dict[str, str]:
        return {
            'video_init': (
                'Analyze the visual cues in the muted video to determine whether the '
                'punchline is humorous.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' "
                "or 'My Answer: No'."
            ),
            'audio_init': (
                'Analyze the audio to determine whether the punchline is humorous.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' "
                "or 'My Answer: No'."
            ),
            'text_init': (
                'Analyze the punchline text to determine whether it is humorous.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' "
                "or 'My Answer: No'."
            ),
            'video_refine': (
                'You previously analyzed the visual cues for this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from the other experts:\n'
                '- Audio Expert: {audio_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'Re-examine the visual evidence in light of their perspectives and decide '
                'whether this punchline is humorous.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_refine': (
                'You previously analyzed the audio for this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from the other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'Re-examine the audio evidence in light of their perspectives and decide '
                'whether this punchline is humorous.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_refine': (
                'You previously analyzed the text of this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from the other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Audio Expert: {audio_answer}\n\n'
                'Re-examine the text evidence in light of their perspectives and decide '
                'whether this punchline is humorous.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'judge': (
                'You are an impartial Judge. Three experts have debated whether this '
                'punchline is humorous or not.\n'
                'Here is the discussion:\n\n{debate_history}\n\n'
                'Based on all evidence from the video, audio, and text analyses, determine '
                'if this punchline is humorous or not.\n'
                "Your answer must start with 'Yes' or 'No', followed by your reasoning."
            ),
        }

    def get_query_aug_prompts(self) -> Dict[str, str]:
        return {
            'controller_init': (
                'You are a Humor Detection Controller. Your task is to coordinate three modality experts '
                '(Video, Audio, Text) to determine if the person is being humorous or not.\n\n'
                'For Round 1, generate THREE specific questions - one for each expert:\n'
                '1. A question for the VIDEO expert to analyze visual expressions and gestures\n'
                '2. A question for the AUDIO expert to analyze vocal tone and speech patterns\n'
                '3. A question for the TEXT expert to analyze the punchline text\n\n'
                'The questions should guide experts to first analyze, then determine if the person is being humorous.\n\n'
                'Format your response exactly as:\n'
                'VIDEO_QUESTION: [your question]\n'
                'AUDIO_QUESTION: [your question]\n'
                'TEXT_QUESTION: [your question]'
            ),
            'controller_followup': (
                'You are a Humor Detection Controller coordinating three modality experts.\n\n'
                'Here are the responses from Round {prev_round}:\n'
                '- Video Expert: {video_response}\n'
                '- Audio Expert: {audio_response}\n'
                '- Text Expert: {text_response}\n\n'
                'Based on these responses, generate follow-up questions for Round {round_num} to dig deeper.\n'
                'Focus on areas where the evidence is unclear or where experts might disagree.\n\n'
                'Format your response exactly as:\n'
                'VIDEO_QUESTION: [your question]\n'
                'AUDIO_QUESTION: [your question]\n'
                'TEXT_QUESTION: [your question]'
            ),
            'controller_decision': (
                'You are the final decision maker integrating analyses from three '
                'modality experts (video, audio, text) over {num_rounds} rounds of '
                'questioning. The experts follow strict scoring rubrics that make '
                'them conservative individually — a single modality rarely has '
                'enough evidence alone to commit Yes, so they often hedge.\n\n'
                'Conversation history:\n{conversation_history}\n\n'
                'AGGREGATION RULE:\n'
                '- Answer Yes if ANY single modality clearly identified a humor '
                'technique, OR if two or more modalities leaned toward humor (even '
                'without high individual confidence).\n'
                '- Answer No only if all three modalities clearly indicated no '
                'humor signal.\n'
                '- Do NOT require unanimous high confidence from the experts — '
                'their conservatism is by design; your job is to synthesize their '
                'combined evidence decisively.\n'
                "Your answer must start with 'Yes' or 'No', followed by your reasoning."
            ),
            'video_agent': (
                'Question from the controller: {question}\n\n'
                'Analyze the visual cues in the muted video and answer the question.'
            ),
            'audio_agent': (
                'Question from the controller: {question}\n\n'
                'Analyze the audio and answer the question.'
            ),
            'text_agent': (
                'Question from the controller: {question}\n\n'
                "Punchline text: '{text}'\n\n"
                'Analyze the text and answer the question.'
            ),
        }

    def get_ensemble_task_instructions(self) -> Dict[str, str]:
        return {
            'video': (
                'Task: Decide whether the punchline is humorous based on the visual '
                'cues in the muted video.\n'
                f'{_AV_COMMIT_INSTRUCTION}\n'
                "Begin your answer with 'Yes' or 'No', followed by a brief justification."
            ),
            'audio': (
                'Task: Decide whether the punchline is humorous based on the audio.\n'
                f'{_AV_COMMIT_INSTRUCTION}\n'
                "Begin your answer with 'Yes' or 'No', followed by a brief justification."
            ),
            'text': (
                'Task: Decide whether the text below is humorous (a joke or comedic '
                'remark intended to amuse the audience).\n'
                f'{_TEXT_COMMIT_INSTRUCTION}\n'
                "Begin your answer with 'Yes' or 'No', followed by a brief justification.\n\n"
                "Text: '{text}'"
            ),
        }


class MustardConfig(DatasetConfig):
    """MUStARD dataset configuration - sarcasm detection"""

    DEFAULT_CTM_CONFIGS = {
        'qwen': 'mustard_qwen3vl_thinking_config.json',
        'gemini': 'sarcasm_ctm_config.json',
    }

    # mustard + qwen uses qwen3-vl (video+language only); there's no audio
    # processor in that CTM config. For a fair audio baseline we borrow the
    # sarcasm audio_processor system_prompt from the gemini variant — same
    # task, same domain, model-agnostic prompt.
    MODALITY_FALLBACK_CONFIGS = {
        'qwen': {
            'audio': 'sarcasm_ctm_config.json',
        },
    }

    def __init__(self):
        super().__init__('mustard', 'sarcasm_detection')

    def get_text_field(self, sample: dict) -> str:
        context_list = sample.get('context', [])
        context_speakers = sample.get('context_speakers', [])
        utterance = sample['utterance']
        speaker = sample.get('speaker', 'SPEAKER')
        parts = []
        for i, ctx in enumerate(context_list):
            sp = context_speakers[i] if i < len(context_speakers) else 'OTHER'
            parts.append(f'{sp}: {ctx}')
        parts.append(f'{speaker}: {utterance}')
        return '\n'.join(parts)

    def get_context_field(self, sample: dict) -> str:
        """Return full context (context + utterance)"""
        context_list = sample.get('context', [])
        utterance = sample['utterance']
        text_list = context_list.copy()
        text_list.append(utterance)
        return '\n'.join(text_list)

    def get_label_field(self, sample: dict):
        return sample['sarcasm']

    def get_video_filename(self, test_file: str, video_type: str) -> str:
        if video_type == 'full':
            return f'{test_file}.mp4'
        elif video_type == 'muted':
            return f'{test_file}.mp4'
        elif video_type == 'audio':
            return f'{test_file}_audio.mp4'
        else:
            raise ValueError(f'Unknown video type: {video_type}')

    def get_data_paths(self) -> Dict[str, str]:
        return {
            'full_video': 'data/mustard/mmsd_raw_data/utterances_final',
            'audio_only': 'data/mustard/mustard_audios',
            'video_only': 'data/mustard/mustard_muted_videos',
        }

    def get_default_dataset_path(self) -> str:
        return 'data/mustard/mustard_dataset/mustard_dataset_test.json'

    def get_task_query(self) -> str:
        return 'Is the person being sarcastic or not?'

    def get_system_prompt(self) -> str:
        return (
            'Please analyze the inputs provided to determine whether the person is being sarcastic or not.\n'
            "If you think the input includes exaggerated description or includes strong emotion or its real meaning is not aligned with the original one, please answer 'Yes'. "
            "If you think the input is neutral or its true meaning is not different from its original one, please answer 'No'. "
            "Your answer should begin with either 'Yes' or 'No', followed by your reasoning. "
            'If you are not sure, please provide your best guess and do not say that you are not sure. '
            'You should only make Yes judgement when you are very sure that the person is sarcastic.'
        )

    def get_debate_prompts(self) -> Dict[str, str]:
        return {
            'video_init': (
                'Analyze the visual cues in the muted video to determine whether the '
                'utterance is sarcastic.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' "
                "or 'My Answer: No'."
            ),
            'audio_init': (
                'Analyze the audio to determine whether the utterance is sarcastic.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' "
                "or 'My Answer: No'."
            ),
            'text_init': (
                'Analyze the utterance text to determine whether it is sarcastic.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' "
                "or 'My Answer: No'."
            ),
            'video_refine': (
                'You previously analyzed the visual cues for this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from the other experts:\n'
                '- Audio Expert: {audio_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'Re-examine the visual evidence in light of their perspectives and decide '
                'whether this utterance is sarcastic.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_refine': (
                'You previously analyzed the audio for this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from the other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'Re-examine the audio evidence in light of their perspectives and decide '
                'whether this utterance is sarcastic.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_refine': (
                'You previously analyzed the text of this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from the other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Audio Expert: {audio_answer}\n\n'
                'Re-examine the text evidence in light of their perspectives and decide '
                'whether this utterance is sarcastic.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'judge': (
                'You are an impartial Judge. Three experts have debated whether this '
                'utterance is sarcastic or not.\n'
                'Here is the discussion:\n\n{debate_history}\n\n'
                'Based on all evidence from the video, audio, and text analyses, determine '
                'if this utterance is sarcastic or not.\n'
                "Your answer must start with 'Yes' or 'No', followed by your reasoning."
            ),
        }

    def get_query_aug_prompts(self) -> Dict[str, str]:
        return {
            'controller_init': (
                'You are a Sarcasm Detection Controller. Your task is to coordinate three modality experts '
                '(Video, Audio, Text) to determine if the person is being sarcastic or not.\n\n'
                'For Round 1, generate THREE specific questions - one for each expert:\n'
                '1. A question for the VIDEO expert to analyze visual expressions and gestures\n'
                '2. A question for the AUDIO expert to analyze vocal tone and speech patterns\n'
                '3. A question for the TEXT expert to analyze the utterance text\n\n'
                'The questions should guide experts to first analyze, then determine if the person is being sarcastic.\n\n'
                'Format your response exactly as:\n'
                'VIDEO_QUESTION: [your question]\n'
                'AUDIO_QUESTION: [your question]\n'
                'TEXT_QUESTION: [your question]'
            ),
            'controller_followup': (
                'You are a Sarcasm Detection Controller coordinating three modality experts.\n\n'
                'Here are the responses from Round {prev_round}:\n'
                '- Video Expert: {video_response}\n'
                '- Audio Expert: {audio_response}\n'
                '- Text Expert: {text_response}\n\n'
                'Based on these responses, generate follow-up questions for Round {round_num} to dig deeper.\n'
                'Focus on areas where the evidence is unclear or where experts might disagree.\n\n'
                'Format your response exactly as:\n'
                'VIDEO_QUESTION: [your question]\n'
                'AUDIO_QUESTION: [your question]\n'
                'TEXT_QUESTION: [your question]'
            ),
            'controller_decision': (
                'You are the final decision maker integrating analyses from three '
                'modality experts (video, audio, text) over {num_rounds} rounds of '
                'questioning. The experts follow strict scoring rubrics that make '
                'them conservative individually — a single modality rarely has '
                'enough evidence alone to commit Yes, so they often hedge.\n\n'
                'Conversation history:\n{conversation_history}\n\n'
                'AGGREGATION RULE:\n'
                '- Answer Yes if ANY single modality clearly identified a sarcasm '
                'signal, OR if two or more modalities leaned toward sarcasm (even '
                'without high individual confidence).\n'
                '- Answer No only if all three modalities clearly indicated no '
                'sarcasm signal.\n'
                '- Do NOT require unanimous high confidence from the experts — '
                'their conservatism is by design; your job is to synthesize their '
                'combined evidence decisively.\n'
                "Your answer must start with 'Yes' or 'No', followed by your reasoning."
            ),
            'video_agent': (
                'Question from the controller: {question}\n\n'
                'Analyze the visual cues in the muted video and answer the question.'
            ),
            'audio_agent': (
                'Question from the controller: {question}\n\n'
                'Analyze the audio and answer the question.'
            ),
            'text_agent': (
                'Question from the controller: {question}\n\n'
                "Utterance text: '{text}'\n\n"
                'Analyze the text and answer the question.'
            ),
        }

    def get_ensemble_task_instructions(self) -> Dict[str, str]:
        return {
            'video': (
                'Task: Decide whether the utterance is sarcastic based on the visual '
                'cues in the muted video.\n'
                f'{_AV_COMMIT_INSTRUCTION}\n'
                "Begin your answer with 'Yes' or 'No', followed by a brief justification."
            ),
            'audio': (
                'Task: Decide whether the utterance is sarcastic based on the audio.\n'
                f'{_AV_COMMIT_INSTRUCTION}\n'
                "Begin your answer with 'Yes' or 'No', followed by a brief justification."
            ),
            'text': (
                'Task: Decide whether the utterance below is sarcastic.\n'
                f'{_TEXT_COMMIT_INSTRUCTION}\n'
                "Begin your answer with 'Yes' or 'No', followed by a brief justification.\n\n"
                "Text: '{text}'"
            ),
        }


# Dataset registry
DATASET_CONFIGS = {
    'urfunny': URFunnyConfig(),
    'mustard': MustardConfig(),
}


def get_dataset_config(dataset_name: str) -> DatasetConfig:
    """Get dataset configuration"""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f'Unknown dataset: {dataset_name}. '
            f'Available: {list(DATASET_CONFIGS.keys())}'
        )
    return DATASET_CONFIGS[dataset_name]
