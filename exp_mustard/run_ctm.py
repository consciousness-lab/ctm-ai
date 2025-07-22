import json
import os
import sys

from ctm_ai.ctms.ctm import ConsciousTuringMachine

sys.path.append('..')


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file='ctm.jsonl'):
    dataset = load_data('mustard_dataset/mustard_dataset_test.json')
    ctm = ConsciousTuringMachine('sarcasm_ctm')
    target_sentence = dataset[test_file]['utterance']
    query = 'Is the person saying sarcasm or not?'
    text_list = dataset[test_file]['context']
    text_list.append(target_sentence)
    fullContext = ''
    for i in range(len(text_list)):
        currentUtterance = f'{text_list[i]} \n'
        fullContext += currentUtterance
    audio_path = f'mustard_audios/{test_file}_audio.mp4'
    video_frames_path = f'mustard_frames/{test_file}_frames'
    file_paths = [
        os.path.join(video_frames_path, file_name)
        for file_name in os.listdir(video_frames_path)
        if os.path.isfile(os.path.join(video_frames_path, file_name))
    ]
    answer = ctm(
        query=query,
        text=fullContext,
        video_frames_path=file_paths,
        audio_path=audio_path,
    )

    print('------------------------------------------')
    print(answer)
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [answer],
            'label': dataset[test_file]['sarcasm'],
        }
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    dataset_path = 'mustard_dataset/mustard_dataset_test.json'
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f'Total Test Cases: {len(test_list)}')
    test_list = test_list[:1]

    for test_file in test_list:
        run_instance(test_file)
