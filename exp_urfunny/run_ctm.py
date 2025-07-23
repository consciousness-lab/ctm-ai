import logging
import os

from torch.utils.data import DataLoader
from urfunny_dataset import URFunnyDataset

from ctm_ai.ctms import CTM

if __name__ == '__main__':
    logging.basicConfig(
        filename='ctm_urfunny.log',
        filemode='a',
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO,
    )

    dataset_path = 'dataset_sample.json'
    video_frames_root = 'test_inputs/urfunny_frames'
    audio_root = 'test_inputs/urfunny_audios'

    dataset = URFunnyDataset(dataset_path, video_frames_root, audio_root)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    for batch in dataloader:
        for i in range(len(batch['label'])):
            query = 'Is the persion being hurmous or not? '
            file_paths = []
            for root, dirs, files in os.walk(batch['video_frames_path'][i]):
                for file in files:
                    file_paths.append(str(os.path.join(root, file)))
            file_paths.sort()
            ctm = CTM('urfunny_test')
            answer = ctm(
                query=query,
                text=batch['punchline'][i],
                video_frames_path=file_paths,
                audio_path=batch['audio_path'][i],
            )
            print(answer)
