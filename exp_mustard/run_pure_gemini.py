import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from exp_baselines import GeminiMultimodalLLM

sys.path.append('..')

SYS_PROMPT = (
    'Please analyze the inputs provided to determine the punchline provided sarcasm or not.'
    "If you think these inputs includes exaggerated description or its real meaning is not aligned with the original one, please answer 'Yes'."
    "If you think these inputs is neutral or its true meaning is not different from its original one, please answer 'No'."
    "Your answer should be either 'Yes' or 'No', with nothing else."
)

file_lock = Lock()


def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, dataset, output_file='baseline_gemini_parallel.jsonl'):
    try:
        print(f'[{test_file}] Starting processing...')

        target_sentence = dataset[test_file]['utterance']
        query = 'Is the person sarcasm or not?'
        text_list = dataset[test_file]['context']
        text_list.append(target_sentence)
        fullContext = ''
        for i in range(len(text_list)):
            currentUtterance = f'{text_list[i]} \n'
            fullContext += currentUtterance

        audio_path = f'mustard_audios/{test_file}_audio.mp4'
        video_frames_path = f'mustard_frames/{test_file}_frames'

        if not os.path.exists(audio_path):
            print(f'[{test_file}] Audio not exist: {audio_path}')
            return f'Audio file not found for {test_file}'

        if not os.path.exists(video_frames_path):
            print(f'[{test_file}] Video frames not exist: {video_frames_path}')
            return f'Video frames not found for {test_file}'

        print(f'[{test_file}] Target sentence: {target_sentence[:100]}...')
        print(f'[{test_file}] Audio found: {audio_path}')
        print(f'[{test_file}] Video frames found: {video_frames_path}')

        gemini_llm = GeminiMultimodalLLM(
            file_name=test_file,
            image_frames_folder=video_frames_path,
            audio_file_path=audio_path,
            context=target_sentence,
            query=query,
            model_name='gemini-2.0-flash-lite',
        )

        print(f'[{test_file}] Calling Gemini...')
        start_time = time.time()
        answer = gemini_llm.generate_response()
        end_time = time.time()

        print(f'[{test_file}] Gemini call completed in {end_time - start_time:.2f}s')
        print(f'[{test_file}] Answer: {answer}')

        result = {
            test_file: {
                'answer': [answer],
                'label': dataset[test_file]['sarcasm'],
            }
        }

        with file_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f'[{test_file}] Result saved to {output_file}')
        return f'Successfully processed {test_file}'

    except Exception as e:
        error_msg = f'Error processing {test_file}: {str(e)}'
        print(f'[{test_file}] {error_msg}')
        import traceback

        traceback.print_exc()
        return error_msg


def run_parallel(max_workers=4, output_file='baseline_gemini_only_language.jsonl'):
    dataset_path = 'mustard_dataset/mustard_dataset_test.json'
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f'Total test samples: {len(test_list)}')
    print(f'Using {max_workers} workers')
    print(f'Output file: {output_file}')
    print('=' * 50)

    with open(output_file, 'w', encoding='utf-8'):
        pass

    start_time = time.time()
    completed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_test = {
            executor.submit(run_instance, test_file, dataset, output_file): test_file
            for test_file in test_list
        }

        for future in as_completed(future_to_test):
            test_file = future_to_test[future]
            completed_count += 1

            try:
                result = future.result()
                print(f'Progress: {completed_count}/{len(test_list)} - {result}')
            except Exception as exc:
                print(f'Error processing {test_file}: {exc}')

    end_time = time.time()
    total_time = end_time - start_time
    print('=' * 50)
    print(f'Total processing time: {total_time:.2f} seconds')
    print(f'Average time per sample: {total_time / len(test_list):.2f} seconds')


if __name__ == '__main__':
    max_workers = 1

    run_parallel(max_workers=max_workers)
