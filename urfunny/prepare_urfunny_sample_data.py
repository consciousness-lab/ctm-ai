import json
import random


def generate_urfunny_data(
    baseline_res_path: str,
    origin_dataset_path: str,
    sample_dataset_path: str,
    sample_num: int,
) -> None:
    predicted_data_list = []
    with open(baseline_res_path, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip():
                predicted_data_list.append(json.loads(line))
    with open(origin_dataset_path, 'r', encoding='utf-8') as json_file:
        origin_dataset = json.load(json_file)
    predicted_wrong_data_list = []
    for data in predicted_data_list:
        if data['logits'][1] > data['logits'][0] and data['target'] == 0:
            predicted_wrong_data_list.append(data)
        elif data['logits'][1] < data['logits'][0] and data['target'] == 1:
            predicted_wrong_data_list.append(data)
        else:
            continue
    sample_list = []
    count_0 = 0
    count_1 = 0
    random.shuffle(predicted_wrong_data_list)
    for wrong_data in predicted_wrong_data_list:
        if wrong_data['target'] == 0 and count_0 < sample_num // 2:
            count_0 += 1
            sample_list.append(wrong_data['image_id'])
        if wrong_data['target'] == 1 and count_1 < (sample_num - sample_num // 2):
            count_1 += 1
            sample_list.append(wrong_data['image_id'])
    urfunny_sample_dataset = {}
    for key in origin_dataset:
        if key in sample_list:
            urfunny_sample_dataset[key] = origin_dataset[key]
    with open(sample_dataset_path, 'w', encoding='utf-8') as file:
        json.dump(urfunny_sample_dataset, file, ensure_ascii=False, indent=4)
    print(f'Successful in writing: {sample_dataset_path}')


if __name__ == '__main__':
    generate_urfunny_data(
        'urfunny_baseline_logits.jsonl',
        'data_raw/urfunny_dataset_test.json',
        'dataset_sample.json',
        60,
    )
