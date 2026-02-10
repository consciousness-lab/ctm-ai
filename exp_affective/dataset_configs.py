"""
Dataset configuration file - supports different affective computing datasets

Each dataset configuration contains:
- Dataset name and task type
- Text field mappings
- Label field names
- File path prefixes
- Task-related prompts and queries
"""

from typing import Dict, Any


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
            "If you think the inputs includes exaggerated description or it is expressing sarcastic meaning, please answer 'Yes'."
            "If you think the inputs is neutral or just common meaning, please answer 'No'."
            "Your answer should begin with either 'Yes' or 'No', followed by your reasoning."
            'If you are not sure, please provide your best guess and do not say that you are not sure.'
        )

    def get_debate_prompts(self) -> Dict[str, str]:
        return {
            'video_init': (
                'You are a Video Analysis Expert. You will analyze whether the person in the video is being humorous or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_init': (
                'You are an Audio Analysis Expert. You will analyze whether the person in the audio is being humorous or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_init': (
                'You are a Text Analysis Expert. You will be given a punchline that was said by a person, and analyze whether the person is being humorous or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'video_refine': (
                'You are a Video Analysis Expert. You previously analyzed the video of this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Audio Expert: {audio_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the video evidence carefully.\n'
                'Then, determine if this punchline is humorous or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_refine': (
                'You are an Audio Analysis Expert. You previously analyzed the audio of this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the audio evidence carefully.\n'
                'Then, determine if this punchline is humorous or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_refine': (
                'You are a Text Analysis Expert. You previously analyzed this punchline.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Audio Expert: {audio_answer}\n\n'
                'First, consider their perspectives and re-examine the text evidence of the punchline carefully.\n'
                'Then, determine if this punchline is humorous or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'judge': (
                'You are an impartial Judge. Three experts have debated whether this punchline is humorous or not.\n'
                'Here is the discussion:\n\n{debate_history}\n\n'
                'Based on all evidence from the video, audio, and text analyses, determine if this punchline is humorous or not.\n'
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
                'You are a Humor Detection Controller. You have gathered evidence from three modality experts '
                'over {num_rounds} rounds of questioning.\n\n'
                'Here is the complete conversation history:\n{conversation_history}\n\n'
                'Based on all the evidence gathered from video, audio, and text analyses, '
                'determine if this person is being humorous or not.\n'
                "Your answer must start with 'Yes' or 'No', followed by your reasoning."
            ),
            'video_agent': (
                'You are a Video Analysis Expert. You will analyze video frames showing a person.\n'
                'Question: {question}\n\n'
                "First, carefully observe and describe the person's visual expressions, gestures, and body language.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'audio_agent': (
                'You are an Audio Analysis Expert. You will analyze audio of a person speaking.\n'
                'Question: {question}\n\n'
                "First, carefully listen and describe the person's vocal tone, intonation, speech patterns, and any audio cues.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'text_agent': (
                'You are a Text Analysis Expert. You will be given a punchline that was said by a person.\n'
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
        return sample['utterance']

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
        return 'Is the person sarcasm or not?'

    def get_system_prompt(self) -> str:
        return (
            'Please analyze the inputs provided to determine whether the person being sarcastic or not.\n'
            "If you think the inputs includes exaggerated description or includes strong emotion or its real meaning is not aligned with the original one, please answer 'Yes'."
            "If you think the inputs is neutral or its true meaning is not different from its original one, please answer 'No'."
            "Your answer should begin with either 'Yes' or 'No', followed by your reasoning."
            'If you are not sure, please provide your best guess and do not say that you are not sure.'
            'You should only make Yes judgement when you are very sure that the person is sarcastic.'
        )

    def get_debate_prompts(self) -> Dict[str, str]:
        return {
            'video_init': (
                'You are a Video Analysis Expert. You will analyze whether the person in the video is being sarcastic or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_init': (
                'You are an Audio Analysis Expert. You will analyze whether the person in the audio is being sarcastic or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_init': (
                'You are a Text Analysis Expert. You will be given an utterance that was said by a person, and analyze whether the person is being sarcastic or not.\n'
                "First, provide your analysis. Then, end your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'video_refine': (
                'You are a Video Analysis Expert. You previously analyzed the video of this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Audio Expert: {audio_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the video evidence carefully.\n'
                'Then, determine if this utterance is sarcastic or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'audio_refine': (
                'You are an Audio Analysis Expert. You previously analyzed the audio of this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Text Expert: {text_answer}\n\n'
                'First, consider their perspectives and re-examine the audio evidence carefully.\n'
                'Then, determine if this utterance is sarcastic or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'text_refine': (
                'You are a Text Analysis Expert. You previously analyzed this utterance.\n'
                'Your previous answer: {own_answer}\n\n'
                'Here are the analyses from other experts:\n'
                '- Video Expert: {video_answer}\n'
                '- Audio Expert: {audio_answer}\n\n'
                'First, consider their perspectives and re-examine the text evidence of the utterance carefully.\n'
                'Then, determine if this utterance is sarcastic or not.\n'
                "End your response with 'My Answer: Yes' or 'My Answer: No'."
            ),
            'judge': (
                'You are an impartial Judge. Three experts have debated whether this utterance is sarcastic or not.\n'
                'Here is the discussion:\n\n{debate_history}\n\n'
                'Based on all evidence from the video, audio, and text analyses, determine if this utterance is sarcastic or not.\n'
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
                'You are a Sarcasm Detection Controller. You have gathered evidence from three modality experts '
                'over {num_rounds} rounds of questioning.\n\n'
                'Here is the complete conversation history:\n{conversation_history}\n\n'
                'Based on all the evidence gathered from video, audio, and text analyses, '
                'determine if this person is being sarcastic or not.\n'
                "Your answer must start with 'Yes' or 'No', followed by your reasoning."
            ),
            'video_agent': (
                'You are a Video Analysis Expert. You will analyze video frames showing a person.\n'
                'Question: {question}\n\n'
                "First, carefully observe and describe the person's visual expressions, gestures, and body language.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'audio_agent': (
                'You are an Audio Analysis Expert. You will analyze audio of a person speaking.\n'
                'Question: {question}\n\n'
                "First, carefully listen and describe the person's vocal tone, intonation, speech patterns, and any audio cues.\n"
                'Then, based on your analysis, provide your answer to the question.'
            ),
            'text_agent': (
                'You are a Text Analysis Expert. You will be given an utterance that was said by a person.\n'
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
