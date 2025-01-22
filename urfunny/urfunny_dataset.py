import json
import os

from torch.utils.data import Dataset


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)


class URFunnyDataset(Dataset):
    def __init__(self, dataset_path, video_frames_root, audio_root):
        self.dataset = load_data(dataset_path)
        self.video_frames_root = video_frames_root
        self.audio_root = audio_root
        self.keys = list(self.dataset.keys())

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
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
