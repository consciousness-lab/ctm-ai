import json
import os
import sys
import time

from exp_baselines import ConsciousnessTuringMachineBaseline

sys.path.append("..")

SYS_PROMPT = (
    "Please analyze the inputs provided to determine the punchline provided below humor or not."
    "If you think the these inputs include exaggerated description or it is expressing sarcastic meaning, please answer 'Yes'."
    "If you think the these inputs are neutral or just common meaning, please answer 'No'."
    "You should also provide your reason for your answer. "
)


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file="baseline2.jsonl"):
    dataset = load_data("urfunny_dataset_test.json")
    ctm = ConsciousnessTuringMachineBaseline("urfunny_test")
    ctm.config.filename_log = f"{test_file}.log"
    target_sentence = dataset[test_file]["punchline_sentence"]
    query = f"{SYS_PROMPT}\n\n punchline:'{target_sentence}' "
    text_list = dataset[test_file]["context_sentences"]
    text_list.append(target_sentence)
    fullContext = ""
    for i in range(len(text_list)):
        currentUtterance = f"{text_list[i]} \n"
        fullContext += currentUtterance
    audio_path = f"urfunny_audios/{test_file}_audio.mp4"
    video_frames_path = f"urfunny_frames/{test_file}_frames"
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
    print("------------------------------------------")
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
    dataset_path = "urfunny_dataset_test.json"
    dataset = load_data(dataset_path)

    test_list = list(dataset.keys())
    print(f"Total Test Cases: {len(test_list)}")

    start_index = test_list.index("3432") + 1 if "3432" in test_list else 0
    for test_file in test_list[start_index:]:
        run_instance(test_file)
        time.sleep(5)
