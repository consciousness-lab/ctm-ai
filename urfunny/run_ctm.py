import logging
import os

from torch.utils.data import DataLoader
from urfunny_dataset import URFunnyDataset

from ctm_ai.ctms import CTM

SYS_PROMPT = (
    'Please analyze the multimodal inputs provided to determine the punchline provided humor or not.'
    "If you think the these inputs include exaggerated description or it is expressing sarcastic meaning, please answer 'Yes'."
    "If you think the these inputs are neutral or just common meaning, please answer 'No'."
    'You should also provide your reason for your answer. '
)


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
            print(batch['video_frames_path'][i])
            logging.info(
                f"======================{batch['filename'][i]} starting ======================"
            )
            logging.info(f"Query: {batch['punchline'][i]}")
            logging.info(f"Ground truth: {batch['label'][i]}")
            query = f"{SYS_PROMPT}\n\n punchline: {batch['punchline'][i]}"
            file_paths = []
            for root, dirs, files in os.walk(batch['video_frames_path'][i]):
                for file in files:
                    file_paths.append(str(os.path.join(root, file)))
            file_paths.sort()
            ctm = CTM('urfunny_test')
            answer = ctm(
                query=query,
                text=batch['context'][i],
                video_frames_path=file_paths,
                audio_path=batch['audio_path'][i],
            )
            print(answer)
