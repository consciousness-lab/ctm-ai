import json
import os
from typing import Dict, Union

import torch
from torch.utils.data import Dataset


class URFunnyDataset(Dataset[Dict[str, Union[str, torch.Tensor]]]):
    def __init__(self, dataset_path: str, video_frames_root: str, audio_root: str):
        with open(dataset_path, 'r', encoding='utf-8') as file:
            self.dataset = json.load(file)
        self.video_frames_root = video_frames_root
        self.audio_root = audio_root
        self.keys = list(self.dataset.keys())

    def __len__(self) -> int:
        return len(self.keys)

    def __getitem__(self, idx: int) -> dict[str, Union[str, torch.Tensor]]:
        key = self.keys[idx]
        sample = self.dataset[key]

        video_frames_path = os.path.join(self.video_frames_root, f'{key}_frames')

        audio_path = os.path.join(self.audio_root, f'{key}_audio.mp4')

        text_list = sample['context_sentences']
        punchline = sample['punchline_sentence']
        text_list.append(punchline)
        full_context = '\n'.join(text_list)

        label = sample['label']

        return {
            'video_frames_path': video_frames_path,
            'audio_path': audio_path,
            'context': full_context,
            'punchline': punchline,
            'label': label,
            'filename': key,
        }
