import json
import os
import sys

from ctm_ai.ctms.ctm import ConsciousnessTuringMachine

sys.path.append("..")

SYS_PROMPT = (
    "Please analyze the inputs provided to determine the person is sarcasm or not."
    "You should also provide your reason for your answer."
)
LANGUAGE_DESCRIPTION = (
    "Additional instructions to judge the text is sarcasm or not:"
    "If you think the text includes exaggerated description or includes strong emotion or its real meaning is not aligned with the original one, it usually means sarcasm."
    "If you think the text is neutral or its true meaning is not different from its original one, it usually means not sarcasm."
    "Please make sure that your answer is based on the text itself, not on the context or your personal knowledge."
    "You should only make sarcasm judgement when you are very sure that the text is sarcastic."
)


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file="ctm.jsonl"):
    dataset = load_data("mustard_dataset/mustard_dataset_test.json")
    ctm = ConsciousnessTuringMachine("sarcasm_ctm")
    target_sentence = dataset[test_file]["utterance"]
    query = SYS_PROMPT
    text_list = dataset[test_file]["context"]
    text_list.append(target_sentence)
    fullContext = f"{LANGUAGE_DESCRIPTION}\n\n The full context is: \n"
    for i in range(len(text_list)):
        currentUtterance = f"{text_list[i]} \n"
        fullContext += currentUtterance
    audio_path = f"mustard_audios/{test_file}_audio.mp4"
    video_frames_path = f"mustard_frames/{test_file}_frames"
    file_paths = [
        os.path.join(video_frames_path, file_name)
        for file_name in os.listdir(video_frames_path)
        if os.path.isfile(os.path.join(video_frames_path, file_name))
    ]
    answer = ctm(
        query=query,
        text=fullContext,
        # video_frames_path=file_paths,
        # audio_path=audio_path,
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
    test_list = test_list[:100]
    false_list = []

    for test_file in test_list:
        if not dataset[test_file]["sarcasm"]:
            false_list.append(test_file)

    print(f"Total False Cases: {len(false_list)}")
    for test_file in false_list:
        run_instance(test_file)
