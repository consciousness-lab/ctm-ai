"""
Dataset configuration file - supports different affective computing datasets

Each dataset configuration contains:
- Dataset name and task type
- Text field mappings
- Label field names
- File path prefixes
- Task-related prompts and queries
"""

from typing import Any, Dict


class DatasetConfig:
    """Base class for dataset configuration"""

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
        """Get prompts for debate experiments"""
        raise NotImplementedError

    def get_query_aug_prompts(self) -> Dict[str, str]:
        """Get prompts for query augmentation experiments"""
        raise NotImplementedError


class URFunnyConfig(DatasetConfig):
    """URFunny dataset configuration - humor detection"""

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

    def get_modality_system_prompts(self) -> Dict[str, str]:
        """Modality-specific expert system prompts, aligned with ctm_conf/urfunny_test_config.json."""
        return {
            'video': (
                'You are an expert at analyzing visual cues in video frames. '
                'Analyze the visual cues, body language, facial expressions, and context. '
                'Note that the videos do not include audio, you should only analyze the visual cues. '
                'Always ground your answer in specific visual observations. Be concise but thorough.'
            ),
            'text': (
                'You are an expert at analyzing text and language. '
                'You will receive a text transcript and a query. Analyze word choice and phrasing, '
                'rhetorical devices, context and literal vs. intended meaning, and sentiment mismatch. '
                'When the query is about specific textual details, provide precise analysis grounded in '
                'the text. Always cite specific words or phrases as evidence. Be concise but thorough.'
            ),
            'audio': (
                'You are an expert at analyzing audio. You will receive an audio file and a query. '
                'Listen carefully to the audio and analyze the tone, emotion, pitch, speed, and vocal patterns. '
                'Pay special attention to the tone of voice (flat, exaggerated, or ironic), pitch variations, '
                'speaking speed, emphasis on certain words, and pauses and timing. Answer the query based on '
                'your analysis of the audio. Explain your reasoning based on what you hear in the audio.'
            ),
        }

    def get_aggregation_prompt(self) -> str:
        """CTM-aligned final aggregation prompt (from urfunny_test_config parse_prompt_template)."""
        return (
            'You are a humor detection expert. Based solely on the analysis provided below, '
            'determine if the punchline is humorous.\n\n'
            'Your answer MUST start with either "Yes" (if humorous) or "No" (if not humorous), '
            'followed by a brief explanation.\n\n'
            'IMPORTANT: If the analysis expresses uncertainty, is inconclusive, or lacks sufficient '
            'evidence, you should answer "No".'
        )

    def get_debate_prompts(self) -> Dict[str, str]:
        mod = self.get_modality_system_prompts()
        return {
            'video_init': (
                f'{mod["video"]}\n\n'
                'Task: Analyze whether the person in the video is being humorous or not, '
                'based solely on the visual cues.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_init': (
                f'{mod["audio"]}\n\n'
                'Task: Analyze whether the person in the audio is being humorous or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_init': (
                f'{mod["text"]}\n\n'
                'Task: Analyze whether the target punchline is humorous.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'video_refine': (
                f'{mod["video"]}\n\n'
                'You previously analyzed the visual cues of the person in the video.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Audio Expert: {audio_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the video evidence carefully. '
                'Note that the video does not contain audio, you should just focus on the visual cues.\n'
                'Then, determine if this punchline is humorous or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_refine': (
                f'{mod["audio"]}\n\n'
                'You previously analyzed the audio of this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the audio evidence carefully.\n'
                'Then, determine if this punchline is humorous or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_refine': (
                f'{mod["text"]}\n\n'
                'You previously analyzed this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Audio Expert: {audio_answer}\n\n'
                'First, consider their perspectives and re-examine the text evidence carefully.\n'
                'Then, determine if this punchline is humorous or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'judge': (
                f'{self.get_aggregation_prompt()}\n\n'
                'Three experts (Video, Audio, Text) have debated whether this punchline is humorous or not. '
                'You must weigh all three experts equally — do not favor any single modality. '
                "Evaluate the strength of each expert's evidence independently.\n\n"
                'Analysis (debate history):\n{debate_history}'
            ),
        }

    def get_query_aug_prompts(self) -> Dict[str, str]:
        mod = self.get_modality_system_prompts()
        return {
            'controller_init': (
                'You are a Humor Detection Controller. Your task is to coordinate three modality experts '
                '(Video, Audio, Text) to determine if the person is being humorous or not.\n\n'
                'The target punchline is: \'{target_sentence}\'\n\n'
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
                f'{self.get_aggregation_prompt()}\n\n'
                'You have gathered evidence from three modality experts (Video, Audio, Text) '
                'over {num_rounds} rounds of questioning. Weigh all three experts equally — '
                'do not favor any single modality.\n\n'
                'Analysis (conversation history):\n{conversation_history}'
            ),
            'video_agent': (
                f'{mod["video"]}\n\n'
                'Question: {question}\n\n'
                "First, carefully observe and describe the person's visual expressions, gestures, and body language.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'audio_agent': (
                f'{mod["audio"]}\n\n'
                'Question: {question}\n\n'
                "First, carefully listen and describe the person's vocal tone, intonation, speech patterns, and any audio cues.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'text_agent': (
                f'{mod["text"]}\n\n'
                'Question: {question}\n\n'
                "Punchline: '{text}'\n\n"
                'First, carefully read and analyze the text content, word choice, tone, and any linguistic patterns.\n'
                'Then, based on your analysis of what this person said, provide your answer to the question.'
            ),
        }


class MustardConfig(DatasetConfig):
    """MUStARD dataset configuration - sarcasm detection"""

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

    def get_modality_system_prompts(self) -> Dict[str, str]:
        """Modality-specific expert system prompts, aligned with ctm_conf/sarcasm_ctm_config.json.

        Used by debate/orchestra/ensemble to ensure all MA baselines start from
        the same expert knowledge as CTM — so cross-method comparison reflects
        orchestration differences, not prompt-content differences.
        """
        return {
            'video': (
                'You are an expert at analyzing visual cues in video frames. '
                'Analyze the visual cues, body language, facial expressions, and context. '
                'Note that the videos do not include audio, you should only analyze the visual cues. '
                'Always ground your answer in specific visual observations. Be concise but thorough.'
            ),
            'text': (
                'You are an expert at analyzing text and language for sarcasm and irony. '
                'You will receive a text transcript and a query. The LAST line is the target utterance; '
                'preceding lines are conversation context. First identify what the CONTEXT is about, '
                'then analyze whether the target utterance is sarcastic.\n\n'
                'Analyze the utterance carefully in the context of the conversation: consider what was '
                'said before, whether the statement contradicts the situation, and whether the speaker is '
                'saying the opposite of what they mean. Look for verbal irony, understatement, hyperbole, '
                'and absurd or impossible scenarios presented as serious explanations. Key distinctions: '
                '(1) Genuine anger, frustration, or direct insults are NOT sarcasm even if they sound harsh. '
                'A rhetorical question expressing real annoyance is not sarcastic. '
                '(2) Sarcasm requires the speaker to mean the OPPOSITE of their literal words or to use '
                'deliberate incongruity with the situation. '
                '(3) Absurd statements that cannot be taken literally in context are likely sarcastic. '
                '(4) Deadpan or dry humor — where the speaker says something absurd, exaggerated, or '
                'contextually inappropriate with a straight-faced delivery — IS a form of sarcasm. If a '
                'statement sounds normal on its own but is clearly ridiculous or incongruous given the '
                'conversation context, it is likely sarcastic. '
                '(5) Playful teasing where the speaker genuinely means what they say is NOT sarcasm. '
                'Calling someone bad at something when they actually did something bad is genuine, not sarcastic. '
                '(6) Genuine self-deprecation or honest emotional reactions (frustration, surprise, amusement) '
                'are NOT sarcasm. '
                "(7) Very short or simple responses ('Oh', 'Yeah', 'No') should NOT be classified as sarcastic "
                'unless the context makes the meaning reversal unmistakable. '
                'Always cite specific words or phrases as evidence. Be concise but thorough.'
            ),
            'audio': (
                'You are an expert at analyzing audio. You will receive an audio file and a query. '
                'Listen carefully to the audio and analyze the tone, emotion, pitch, speed, and vocal patterns. '
                'Pay special attention to the tone of voice (flat, exaggerated, or ironic), pitch variations, '
                'speaking speed, emphasis on certain words, and pauses and timing. Answer the query based on '
                'your analysis of the audio. Explain your reasoning based on what you hear in the audio. '
                'Key distinctions: '
                '(1) A flat or dry delivery style does NOT automatically mean sarcasm — some people naturally '
                'speak in a dry, deadpan manner. Only flag flat tone as sarcastic if it clearly contrasts with '
                'emotionally charged content. '
                '(2) Teasing, playful, or amused tones are not the same as sarcastic tones. '
                '(3) If you cannot detect clear tonal markers of sarcasm, say so and rate your confidence lower '
                'rather than guessing.'
            ),
        }

    def get_aggregation_prompt(self) -> str:
        """CTM-aligned final aggregation prompt (from parse_prompt_template)."""
        return (
            'You are a sarcasm detection expert. Based solely on the analysis provided below, '
            'determine if the person is being sarcastic.\n\n'
            'Your answer MUST start with either "Yes" (if sarcastic) or "No" (if not sarcastic), '
            'followed by a brief explanation.\n\n'
            'IMPORTANT: If the analysis expresses uncertainty, is inconclusive, or lacks sufficient '
            'evidence, you should answer "No".'
        )

    def get_debate_prompts(self) -> Dict[str, str]:
        mod = self.get_modality_system_prompts()
        return {
            'video_init': (
                f'{mod["video"]}\n\n'
                'Task: Analyze whether the person in the video is being sarcastic or not, '
                'based solely on the visual cues.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_init': (
                f'{mod["audio"]}\n\n'
                'Task: Analyze whether the person in the audio is being sarcastic or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_init': (
                f'{mod["text"]}\n\n'
                'Task: Analyze whether the target utterance is sarcastic.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'video_refine': (
                f'{mod["video"]}\n\n'
                'You previously analyzed the visual cues of the person in the video.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Audio Expert: {audio_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the video evidence carefully. '
                'Note that the video does not contain audio, you should just focus on the visual cues.\n'
                'Then, determine if this utterance is sarcastic or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_refine': (
                f'{mod["audio"]}\n\n'
                'You previously analyzed the audio of this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the audio evidence carefully.\n'
                'Then, determine if this utterance is sarcastic or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_refine': (
                f'{mod["text"]}\n\n'
                'You previously analyzed this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Audio Expert: {audio_answer}\n\n'
                'First, consider their perspectives and re-examine the text evidence carefully.\n'
                'Then, determine if this utterance is sarcastic or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'judge': (
                f'{self.get_aggregation_prompt()}\n\n'
                'Three experts (Video, Audio, Text) have debated whether this utterance is sarcastic or not. '
                'You must weigh all three experts equally — do not favor any single modality. '
                "Evaluate the strength of each expert's evidence independently.\n\n"
                'Analysis (debate history):\n{debate_history}'
            ),
        }

    def get_query_aug_prompts(self) -> Dict[str, str]:
        mod = self.get_modality_system_prompts()
        return {
            'controller_init': (
                'You are a Sarcasm Detection Controller. Your task is to coordinate three modality experts '
                '(Video, Audio, Text) to determine if the person is being sarcastic or not.\n\n'
                'The target utterance is: \'{target_sentence}\'\n\n'
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
                f'{self.get_aggregation_prompt()}\n\n'
                'You have gathered evidence from three modality experts (Video, Audio, Text) '
                'over {num_rounds} rounds of questioning. Weigh all three experts equally — '
                'do not favor any single modality.\n\n'
                'Analysis (conversation history):\n{conversation_history}'
            ),
            'video_agent': (
                f'{mod["video"]}\n\n'
                'Question: {question}\n\n'
                "First, carefully observe and describe the person's visual expressions, gestures, and body language.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'audio_agent': (
                f'{mod["audio"]}\n\n'
                'Question: {question}\n\n'
                "First, carefully listen and describe the person's vocal tone, intonation, speech patterns, and any audio cues.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'text_agent': (
                f'{mod["text"]}\n\n'
                'Question: {question}\n\n'
                "Utterance: '{text}'\n\n"
                'First, carefully read and analyze the text content, word choice, tone, and any linguistic patterns.\n'
                'Then, based on your analysis of what this person said, provide your answer to the question.'
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
