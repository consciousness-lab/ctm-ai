import json
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

    dataset_path = 'data_raw/urfunny_dataset_test.json'
    video_frames_root = 'test_inputs/urfunny_frames'
    audio_root = 'test_inputs/urfunny_audios'

    dataset = URFunnyDataset(dataset_path, video_frames_root, audio_root)

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    output_file = 'ctm_urfunny.jsonl'

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
            print('------------------------------------------')
            print(answer)
            print('------------------------------------------')

            if isinstance(answer, tuple) and len(answer) == 2:
                answer_text, confidence_score = answer
                if hasattr(confidence_score, 'item'):
                    confidence_score = confidence_score.item()
                serializable_answer = [answer_text, float(confidence_score)]
            else:
                serializable_answer = [str(answer)]

            result = {
                str(batch['filename'][i]): {
                    'answer': serializable_answer,
                    'label': str(batch['label'][i]),
                }
            }
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
