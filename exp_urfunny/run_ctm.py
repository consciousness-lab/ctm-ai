import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

from ctm_ai.ctms.ctm import ConsciousTuringMachine

sys.path.append("..")


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


file_lock = Lock()


def run_instance(
    test_file,
    dataset,
    output_file="ctm_0208.jsonl",
):
    try:
        print(f"[{test_file}] Starting processing...")

        ctm = ConsciousTuringMachine("urfunny_test")
        target_sentence = dataset[test_file]["punchline_sentence"]
        query = "Is the person being humorous or not?"

        print(f"[{test_file}] Target sentence: {target_sentence[:100]}...")
        print(f"[{test_file}] Query: {query}")

        audio_path = f"urfunny_audios/{test_file}_audio.mp4"
        video_path = f"urfunny_muted_videos/{test_file}.mp4"

        if not os.path.exists(audio_path):
            print(f"[{test_file}] Audio not exist: {audio_path}")
            audio_path = None
        else:
            print(f"[{test_file}] Audio found: {audio_path}")

        if not os.path.exists(video_path):
            print(f"[{test_file}] Video not exist: {video_path}")
            video_path = None
        else:
            print(f"[{test_file}] Video found: {video_path}")

        print(f"[{test_file}] Calling CTM...")
        start_time = time.time()
        answer, weight_score, parsed_answer = ctm(
            query=query,
            text=target_sentence,
            video_path=video_path,
            audio_path=audio_path,
        )
        end_time = time.time()

        print(f"[{test_file}] CTM call completed in {end_time - start_time:.2f}s")
        print(f"[{test_file}] Answer: {answer}")
        print(f"[{test_file}] Parsed answer: {parsed_answer}")
        result = {
            test_file: {
                "answer": [answer],
                "parsed_answer": [parsed_answer],
                "weight_score": weight_score,
                "label": dataset[test_file]["label"],
            }
        }

        with file_lock:
            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        print(f"[{test_file}] Result saved to {output_file}")
        return f"Successfully processed {test_file}"

    except Exception as e:
        error_msg = f"Error processing {test_file}: {str(e)}"
        print(f"[{test_file}] {error_msg}")
        import traceback

        traceback.print_exc()
        return error_msg


def run_parallel(max_workers=4, output_file="ctm_0208.jsonl"):
    dataset_path = "data_raw/urfunny_dataset_test.json"
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f"Total test samples: {len(test_list)}")
    print(f"Using {max_workers} workers")
    print(f"Output file: {output_file}")
    print("=" * 50)

    with open(output_file, "w", encoding="utf-8"):
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
                print(f"Progress: {completed_count}/{len(test_list)} - {result}")
            except Exception as exc:
                print(f"Error processing {test_file}: {exc}")

    end_time = time.time()
    total_time = end_time - start_time
    print("=" * 50)
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average time per sample: {total_time / len(test_list):.2f} seconds")


if __name__ == "__main__":
    max_workers = 16

    run_parallel(max_workers=max_workers)
