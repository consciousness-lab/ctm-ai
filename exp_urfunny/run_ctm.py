import concurrent.futures
import json
import os
import sys
from typing import Any, Dict, List

from ctm_ai.ctms.ctm import ConsciousTuringMachine

sys.path.append('..')


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def run_single_instance(
    test_file: str, dataset: Dict[str, Any], output_file: str = 'ctm_urfunny.jsonl'
) -> Dict[str, Any]:
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
    print(f'Test file: {test_file}')
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

    return result


def run_parallel_instances(
    test_files: List[str],
    dataset: Dict[str, Any],
    output_file: str = 'ctm_urfunny_parallel.jsonl',
    max_workers: int = 4,
) -> List[Dict[str, Any]]:
    results = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test_file = {
            executor.submit(
                run_single_instance, test_file, dataset, output_file
            ): test_file
            for test_file in test_files
        }

        for future in concurrent.futures.as_completed(future_to_test_file):
            test_file = future_to_test_file[future]
            try:
                result = future.result()
                results.append(result)
                print(f'Completed: {test_file}')
            except Exception as exc:
                print(f'Test file {test_file} generated an exception: {exc}')

    return results


def run_instance(test_file, output_file='ctm_urfunny.jsonl'):
    dataset = load_data('data_raw/urfunny_dataset_test.json')
    return run_single_instance(test_file, dataset, output_file)


if __name__ == '__main__':
    dataset_path = 'data_raw/urfunny_dataset_test.json'
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f'Total Test Cases: {len(test_list)}')

    # test_files_to_run = [
    #     "1008",
    #     "1009",
    #     "1141",
    #     "1142",
    #     "1167",
    #     "1171",
    #     "1183",
    #     "1184",
    #     "1243",
    #     "1247",
    #     "1298",
    #     "1328",
    #     "1327",
    #     "1326",
    # ]
    test_files_to_run = [
        '1171',
        '1183',
        '1184',
        '1243',
        '1247',
        '1298',
        '1328',
        '1327',
        '1326',
    ]
    print(f'Running {len(test_files_to_run)} test files in parallel...')

    # results = run_parallel_instances(
    #     test_files=test_files_to_run,
    #     dataset=dataset,
    #     output_file="ctm_urfunny_tune.jsonl",
    #     max_workers=3,
    # )

    # for test_file in test_files_to_run:
    results = run_single_instance(
        test_file='1171',
        dataset=dataset,
        output_file='ctm_urfunny_tune.jsonl',
    )

    print(f'Completed {len(results)} test cases successfully!')
