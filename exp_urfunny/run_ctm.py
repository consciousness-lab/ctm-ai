import json
import sys

from ctm_ai.ctms.ctm import ConsciousTuringMachine

sys.path.append("..")


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file="ctm_urfunny.jsonl"):
    dataset = load_data("data_raw/urfunny_dataset_test.json")
    ctm = ConsciousTuringMachine("urfunny_test")
    target_sentence = dataset[test_file]["punchline_sentence"]
    query = "Is the persion being humorous or not?"
    audio_path = f"urfunny_audios/{test_file}_audio.mp4"
    video_path = f"urfunny_muted_videos/{test_file}.mp4"
    answer = ctm(
        query=query,
        text=target_sentence,
        video_path=video_path,
        audio_path=audio_path,
    )

    print("------------------------------------------")
    print(target_sentence)
    print(answer)
    print("------------------------------------------")

    result = {
        test_file: {
            "answer": [answer],
            "label": dataset[test_file]["label"],
        }
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset_path = "data_raw/urfunny_dataset_test.json"
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f"Total Test Cases: {len(test_list)}")
    run_instance("630")

    # for test_file in test_list:
    #     run_instance(test_file)
