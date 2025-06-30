import json
import sys
import time

from exp_baselines import GeminiMultimodalLLM

sys.path.append("..")

SYS_PROMPT = (
    "Please analyze the inputs provided to determine the punchline provided sarcasm or not."
    "Your answer should start with 'Yes' or 'No'."
    "If you think these inputs includes exaggerated description or its real meaning is not aligned with the original one, please answer 'Yes'."
    "If you think these inputs is neutral or its true meaning is not different from its original one, please answer 'No'."
    "You should also provide your reason for your answer."
)


def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def run_instance(test_file, output_file="baseline1.jsonl"):
    dataset = load_data("mustard_dataset/mustard_dataset_test.json")
    target_sentence = dataset[test_file]["utterance"]
    query = f"{SYS_PROMPT}\n\n punchline:'{target_sentence}' "
    text_list = dataset[test_file]["context"]
    text_list.append(target_sentence)
    fullContext = ""
    for i in range(len(text_list)):
        currentUtterance = f"{text_list[i]} \n"
        fullContext += currentUtterance
    audio_path = f"mustard_audios/{test_file}_audio.mp4"
    video_frames_path = f"mustard_frames/{test_file}_frames"
    gemini_llm = GeminiMultimodalLLM(
        file_name=test_file,
        image_frames_folder=video_frames_path,
        audio_file_path=audio_path,
        context=fullContext,
        query=query,
        model_name="gemini-1.5-flash-8b",
    )
    answer = gemini_llm.generate_response()
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

    # start_index = test_list.index("10159") + 1 if "10159" in test_list else 0
    for test_file in test_list:
        run_instance(test_file)
        time.sleep(5)
