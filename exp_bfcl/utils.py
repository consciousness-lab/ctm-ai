import json

from constants.category_mapping import TEST_COLLECTION_MAPPING, TEST_FILE_MAPPING


def load_file(file_path, sort_by_id=False):
    result = []
    with open(file_path) as f:
        file = f.readlines()
        for line in file:
            result.append(json.loads(line))

    if sort_by_id:
        result.sort(key=sort_key)
    return result


def parse_test_category_argument(test_category_args):
    test_name_total = set()
    test_filename_total = set()

    for test_category in test_category_args:
        if test_category in TEST_COLLECTION_MAPPING:
            for test_name in TEST_COLLECTION_MAPPING[test_category]:
                test_name_total.add(test_name)
                test_filename_total.add(TEST_FILE_MAPPING[test_name])
        elif test_category in TEST_FILE_MAPPING:
            test_name_total.add(test_category)
            test_filename_total.add(TEST_FILE_MAPPING[test_category])
        else:
            # Invalid test category name
            raise Exception(f'Invalid test category name provided: {test_category}')

    return sorted(list(test_filename_total)), sorted(list(test_name_total))


def is_multi_turn(test_category):
    return 'multi_turn' in test_category


def sort_key(entry):
    """
    Index comes in two forms: TestCategory_Index or TestCategory_Index-FuncDocSubIndex-PromptSubIndex; both 0-indexed.

    TestCategory_Index: For example, `simple_20` means the 21st entry in the `simple` test category.

    TestCategory_Index-FuncDocSubIndex-PromptSubIndex is used when there are multiple prompts for a single function doc; this only happens in the live dataset.
    FuncDocSubIndex increments for each unique function doc.
    PromptSubIndex is per function doc. It resets to 0 for each function doc.
        For example, `live_simple_19-3-15` means the 20th entry in the `live_simple` test category.
        This entry has the 4th unique function doc and the 16th prompt for that function doc (there are at least 15 other prompts for this same function doc in this category).

    In either case, the universal index is enough to sort the entries.
    """
    parts = entry['id'].rsplit('_', 1)
    test_category, index = parts[0], parts[1]
    # This handles the case where the index is in the form TestCategory_Index-FuncDocSubIndex-PromptSubIndex
    if '-' in index:
        index = index.split('-')[0]
    return (test_category, int(index))


def _get_language_specific_hint(test_category):
    if test_category == 'java':
        return ' Note that the provided function is in Java 8 SDK syntax.'
    elif test_category == 'javascript':
        return ' Note that the provided function is in JavaScript syntax.'
    else:
        return ' Note that the provided function is in Python 3 syntax.'


def func_doc_language_specific_pre_processing(function, test_category):
    if len(function) == 0:
        return function

    assert type(function) == list
    for item in function:
        # Add language specific hints to the function description
        func_description = item['description']
        item['description'] = item['description'] + _get_language_specific_hint(
            test_category
        )
        # Process the parameters
        properties = item['parameters']['properties']
        if test_category == 'java':
            for key, value in properties.items():
                if value['type'] == 'any':
                    properties[key]['description'] += (
                        ' This parameter can be of any type of Java object in string representation.'
                    )
                else:
                    value['description'] += (
                        f' This is Java {value["type"]} type parameter in string representation.'
                    )
                if value['type'] == 'ArrayList' or value['type'] == 'Array':
                    value['description'] += (
                        f' The list elements are of type {value["items"]["type"]}; they are not in string representation.'
                    )
                    del value['items']

                value['type'] = 'string'

        elif test_category == 'javascript':
            for key, value in properties.items():
                if value['type'] == 'any':
                    properties[key]['description'] += (
                        ' This parameter can be of any type of JavaScript object in string representation.'
                    )
                else:
                    value['description'] += (
                        f' This is JavaScript {value["type"]} type parameter in string representation.'
                    )
                if value['type'] == 'array':
                    value['description'] += (
                        f' The list elements are of type {value["items"]["type"]}; they are not in string representation.'
                    )
                    del value['items']

                if value['type'] == 'dict':
                    if 'properties' in value:  # not every dict has properties
                        value['description'] += (
                            f' The dictionary entries have the following schema; they are not in string representation. {json.dumps(value["properties"])}'
                        )
                        del value['properties']

                value['type'] = 'string'

    return function
