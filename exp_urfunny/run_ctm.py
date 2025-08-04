import json
import os
import sys

from ctm_ai.ctms.ctm import ConsciousTuringMachine

sys.path.append('..')


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file='ctm_urfunny.jsonl'):
    dataset = load_data('data_raw/urfunny_dataset_test.json')
    ctm = ConsciousTuringMachine('urfunny_test')
    target_sentence = dataset[test_file]['punchline_sentence']
    query = 'Is the persion being hurmous or not?'
    context_sentences = 'context setences: '
    for i in range(len(dataset[test_file]['context_sentences'])):
        context_sentences += dataset[test_file]['context_sentences'][i]
    context_sentences += '\npunchline sentence: ' + target_sentence
    audio_path = f'test_inputs/urfunny_audios/{test_file}_audio.mp4'
    video_frames_path = f'test_inputs/urfunny_frames/{test_file}_frames'
    file_paths = [
        os.path.join(video_frames_path, file_name)
        for file_name in os.listdir(video_frames_path)
        if os.path.isfile(os.path.join(video_frames_path, file_name))
    ]
    answer = ctm(
        query=query,
        text=target_sentence,
        video_frames_path=file_paths,
        audio_path=audio_path,
    )

    print('------------------------------------------')
    print(target_sentence)
    print(answer)
    print('------------------------------------------')

    result = {
        test_file: {
            'answer': [answer],
            'label': dataset[test_file]['label'],
        }
    }

    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    dataset_path = 'data_raw/urfunny_dataset_test.json'
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f'Total Test Cases: {len(test_list)}')
    run_instance('630')

    # for test_file in test_list:
    #     run_instance(test_file)
