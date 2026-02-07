import json
import sys

from ctm_ai.ctms.ctm import ConsciousTuringMachine

sys.path.append("..")


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file="ctm.jsonl"):
    dataset = load_data("mustard_dataset/mustard_dataset_test.json")

    ctm = ConsciousTuringMachine("sarcasm_ctm")
    target_sentence = dataset[test_file]["utterance"]
    query = "Is the person sarcasm or not?"
    audio_path = f"mustard_audios/{test_file}_audio.mp4"
    video_path = f"mustard_muted_videos/{test_file}.mp4"
    answer = ctm(
        query=query,
        text=target_sentence,
        video_path=video_path,
        audio_path=audio_path,
    )

    print("------------------------------------------")
    print(answer)
    print("------------------------------------------")

    result = {
        test_file: {
            "answer": [answer],
            "label": dataset[test_file]["sarcasm"],
        }
    }

    with open(output_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    dataset_path = "mustard_dataset/mustard_dataset_test.json"
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f"Total Test Cases: {len(test_list)}")

    # for test_file in test_list:
    run_instance("2_7")
