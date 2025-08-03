import argparse
import json

from constants.category_mapping import (
    MULTI_TURN_FUNC_DOC_FILE_MAPPING,
    TEST_FILE_MAPPING,
)
from constants.eval_config import (
    MULTI_TURN_FUNC_DOC_PATH,
    PROJECT_ROOT,
    PROMPT_PATH,
    RESULT_PATH,
    TEST_IDS_TO_GENERATE_PATH,
)
from openai_handler import OpenAIHandler
from utils import is_multi_turn, load_file, parse_test_category_argument, sort_key

from ctm_ai.ctms import BFCLCTM

RETRY_LIMIT = 3
# 60s for the timer to complete. But often we find that even with 60 there is a conflict. So 65 is a safe no.
RETRY_DELAY = 65  # Delay in seconds


def get_args():
    parser = argparse.ArgumentParser()
    # Refer to model_choice for supported models.
    parser.add_argument('--model', type=str, default='gpt-4o', nargs='+')
    # Refer to test_categories for supported categories.
    parser.add_argument('--test-category', type=str, default='all', nargs='+')

    parser.add_argument('--result-dir', default=None, type=str)
    parser.add_argument('--run-ids', action='store_true', default=False)
    args = parser.parse_args()

    return args


def get_involved_test_entries(test_category_args, run_ids):
    all_test_file_paths, all_test_categories, all_test_entries_involved = [], [], []
    if run_ids:
        with open(TEST_IDS_TO_GENERATE_PATH) as f:
            test_ids_to_generate = json.load(f)
        for category, test_ids in test_ids_to_generate.items():
            if len(test_ids) == 0:
                continue
            test_file_path = TEST_FILE_MAPPING[category]
            all_test_entries_involved.extend(
                [
                    entry
                    for entry in load_file(PROMPT_PATH / test_file_path)
                    if entry['id'] in test_ids
                ]
            )
            all_test_categories.append(category)
            all_test_file_paths.append(test_file_path)

    else:
        all_test_file_paths, all_test_categories = parse_test_category_argument(
            test_category_args
        )
        # Make a copy here since we are removing list elemenets inside the for loop
        for test_category, file_to_open in zip(
            all_test_categories[:], all_test_file_paths[:]
        ):
            all_test_entries_involved.extend(load_file(PROMPT_PATH / file_to_open))

    return (
        all_test_file_paths,
        all_test_categories,
        all_test_entries_involved,
    )


def collect_test_cases(
    args,
    model_name,
    all_test_categories,
    all_test_file_paths,
    all_test_entries_involved,
):
    model_name_dir = model_name.replace('/', '_')
    model_result_dir = args.result_dir / model_name_dir

    existing_result = []
    for test_category, file_to_open in zip(all_test_categories, all_test_file_paths):
        result_file_path = model_result_dir / file_to_open.replace(
            '.json', '_result.json'
        )
        if result_file_path.exists():
            # Not allowing overwrite, we will load the existing results
            if not args.allow_overwrite:
                existing_result.extend(load_file(result_file_path))
            # Allow overwrite and not running specific test ids, we will delete the existing result file before generating new results
            elif not args.run_ids:
                result_file_path.unlink()
            # Allow overwrite and running specific test ids, we will do nothing here
            else:
                pass

        existing_ids = [entry['id'] for entry in existing_result]

    test_cases_to_generate = [
        test_case
        for test_case in all_test_entries_involved
        if test_case['id'] not in existing_ids
    ]
    test_cases_to_generate = process_multi_turn_test_case(test_cases_to_generate)

    return sorted(test_cases_to_generate, key=sort_key)


def process_multi_turn_test_case(test_cases):
    """
    Multi-turn test cases don't have the function doc in the prompt. We need to add them here.
    """
    for entry in test_cases:
        if not is_multi_turn(entry['id']):
            continue
        involved_classes = entry['involved_classes']
        entry['function'] = []
        for func_collection in involved_classes:
            # func_doc is a list of dict
            func_doc = load_file(
                MULTI_TURN_FUNC_DOC_PATH
                / MULTI_TURN_FUNC_DOC_FILE_MAPPING[func_collection]
            )
            entry['function'].extend(func_doc)

        # Handle Miss Func category; we need to remove the holdout function doc
        if 'missed_function' in entry:
            for turn_index, missed_func_names in entry['missed_function'].items():
                entry['missed_function'][turn_index] = []
                for missed_func_name in missed_func_names:
                    for i, func_doc in enumerate(entry['function']):
                        if func_doc['name'] == missed_func_name:
                            # Add the missed function doc to the missed_function list
                            entry['missed_function'][turn_index].append(func_doc)
                            # Remove it from the function list
                            entry['function'].pop(i)
                            break

    return test_cases


def main(args):
    if type(args.test_category) is not list:
        args.test_category = [args.test_category]

    (
        all_test_file_paths,
        all_test_categories,
        all_test_entries_involved,
    ) = get_involved_test_entries(args.test_category, args.run_ids)

    # Handle model argument - it can be a list or string
    if isinstance(args.model, list):
        model_name = args.model[0]  # Use the first model if multiple provided
    else:
        model_name = args.model

    print(f'Generating results for {model_name}')
    if args.run_ids:
        print('Running specific test cases. Ignoring `--test-category` argument.')
    else:
        print(f'Running full test cases for categories: {all_test_categories}.')

    if args.result_dir is not None:
        args.result_dir = PROJECT_ROOT / args.result_dir
    else:
        args.result_dir = RESULT_PATH

    test_cases_total = collect_test_cases(
        args,
        model_name,
        all_test_categories,
        all_test_file_paths,
        all_test_entries_involved,
    )
    for test_case in test_cases_total:
        openai_handler = OpenAIHandler(model_name)
        inference_data = openai_handler.process_test_entry(test_case)
        query = inference_data['message'][0]['content']
        ctm = BFCLCTM('bfcl_test', inference_data=inference_data)
        answer = ctm(
            query=query,
        )
        print(answer)


if __name__ == '__main__':
    args = get_args()
    main(args)
