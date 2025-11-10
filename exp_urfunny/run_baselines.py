import logging
import os

from torch.utils.data import DataLoader
from urfunny_dataset import URFunnyDataset

from exp_baselines import ConsciousTuringMachineBaseline, GeminiMultimodalLLM

SYS_PROMPT = (
    'Please analyze the inputs provided to determine the punchline provided humor or not.'
    "If you think the these inputs include exaggerated description or it is expressing sarcastic meaning, please answer 'Yes'."
    "If you think the these inputs are neutral or just common meaning, please answer 'No'."
    'You should also provide your reason for your answer. '
)


def run_baseline(name: str) -> None:
    logging.basicConfig(
        filename=f'baseline_{name}_urfunny.log',
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
            logging.info(
                f'======================{batch["filename"][i]} starting ======================'
            )
            logging.info(f'Query: {batch["punchline"][i]}')
            logging.info(f'Ground truth: {batch["label"][i]}')
            query = f'{SYS_PROMPT}\n\n punchline: {batch["punchline"][i]}'
            if name == 'gemini':
                gemini_llm = GeminiMultimodalLLM(
                    file_name=batch['filename'][i],
                    image_frames_folder=batch['video_frames_path'][i],
                    audio_file_path=batch['audio_path'][i],
                    context=batch['context'][i],
                    query=query,
                    model_name='gemini-1.5-pro',
                )

                generated_text = gemini_llm.generate_response()
                logging.info(f'Response text: {generated_text}')
                if generated_text:
                    print('generated response：')
                    print(generated_text)
            if name == 'ctm_without_tree':
                file_paths = []
                for root, dirs, files in os.walk(batch['video_frames_path'][i]):
                    for file in files:
                        file_paths.append(str(os.path.join(root, file)))
                file_paths.sort()
                ctm = ConsciousTuringMachineBaseline('urfunny_test')
                answer = ctm(
                    query=query,
                    text=batch['context'][i],
                    video_frames_path=file_paths,
                    audio_path=batch['audio_path'][i],
                )
                print(answer)


if __name__ == '__main__':
    x = 'ctm_without_tree'
    run_baseline(x)
