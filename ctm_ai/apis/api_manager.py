import json
import os
import time

import requests
from termcolor import colored
from tqdm import tqdm

from .api_base import base_env
from .api_server import change_name, get_rapidapi_response, standardize


def get_white_list(tool_root_dir):
    white_list_dir = os.path.join(tool_root_dir)
    white_list = {}
    for cate in tqdm(os.listdir(white_list_dir)):
        if not os.path.isdir(os.path.join(white_list_dir, cate)):
            continue
        for file in os.listdir(os.path.join(white_list_dir, cate)):
            if not file.endswith('.json'):
                continue
            standard_tool_name = file.split('.')[0]
            with open(os.path.join(white_list_dir, cate, file)) as reader:
                js_data = json.load(reader)
            origin_tool_name = js_data['tool_name']
            white_list[standardize(origin_tool_name)] = {
                'description': js_data['tool_description'],
                'standard_tool_name': standard_tool_name,
            }
    return white_list


def contain(candidate_list, white_list):
    output = []
    for cand in candidate_list:
        if cand not in white_list.keys():
            return False
        output.append(white_list[cand])
    return output


# rapidapi env wrapper
class rapidapi_wrapper(base_env):
    def __init__(self, query_json, tool_descriptions, args, process_id=0):
        super(rapidapi_wrapper).__init__()

        self.tool_root_dir = args.tool_root_dir
        self.toolbench_key = args.toolbench_key
        self.rapidapi_key = args.rapidapi_key
        self.use_rapidapi_key = args.use_rapidapi_key
        self.api_customization = args.api_customization
        self.service_url = 'http://8.130.32.149:8080/rapidapi'
        self.max_observation_length = args.max_observation_length
        self.observ_compress_method = args.observ_compress_method
        self.process_id = process_id

        self.tool_names = []
        self.cate_names = []

        self.input_description = query_json['query']
        self.functions = []
        self.api_name_reflect = {}
        self.standard_tool_name_reflect_all_info = {}
        self.openai_function_names = []
        self.openai_name_reflect_all_info = {}

        data_dict = self.fetch_api_json(query_json)
        self.tool_descriptions = self.build_tool_description(data_dict)

        for k, api_json in enumerate(data_dict['api_list']):
            standard_tool_name = tool_descriptions[k][0]
            openai_function_json, cate_name, pure_api_name, openai_function_name = (
                self.api_json_to_openai_json(api_json, standard_tool_name)
            )
            self.functions.append(openai_function_json)
            self.openai_function_names.append(openai_function_name)

            self.api_name_reflect[openai_function_json['name']] = pure_api_name
            self.tool_names.append(standard_tool_name)
            self.cate_names.append(cate_name)
            tool_description = tool_descriptions[k][1]
            api_description = openai_function_json['description']
            self.openai_name_reflect_all_info[openai_function_name] = [
                openai_function_json,
                f'tool description:\n{standard_tool_name}: {tool_description}\n\napi description:\n{pure_api_name}: {api_description}',
            ]

        finish_func = {
            'name': 'Finish',
            'description': 'If you believe that you have obtained a result that can answer the task, please call this function to provide the final answer. Alternatively, if you recognize that you are unable to proceed with the task in the current state, call this function to restart. Remember: you must ALWAYS call this function at the end of your attempt, and the only part that will be shown to the user is the final answer, so it should contain sufficient information.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'return_type': {
                        'type': 'string',
                        'enum': ['give_answer', 'give_up_and_restart'],
                    },
                    'final_answer': {
                        'type': 'string',
                        'description': 'The final answer you want to give the user. You should have this field if "return_type"=="give_answer"',
                    },
                },
                'required': ['return_type'],
            },
        }

        self.functions.append(finish_func)
        self.CALL_MAX_TIME = 3
        self.task_description = """You should use functions to help handle the real time user querys. Remember:
1.ALWAYS call \"Finish\" function at the end of the task. And the final answer should contain enough information to show to the user,If you can't handle the task, or you find that function calls always fail(the function is not valid now), use function Finish->give_up_and_restart.
2.Do not use origin tool names, use only subfunctions' names.
You have access of the following tools:\n"""

        unduplicated_reflection = {}
        for standardize_tool_name, tool_des in tool_descriptions:
            unduplicated_reflection[standardize_tool_name] = tool_des

        for k, (standardize_tool_name, tool_des) in enumerate(
            unduplicated_reflection.items()
        ):
            striped = tool_des[:512].replace('\n', '').strip()
            if striped == '':
                striped = 'None'
            self.task_description += f'{k + 1}.{standardize_tool_name}: {striped}\n'

        self.success = 0

    def build_tool_description(self, data_dict):
        white_list = get_white_list(self.tool_root_dir)
        origin_tool_names = [
            standardize(cont['tool_name']) for cont in data_dict['api_list']
        ]
        tool_des = contain(origin_tool_names, white_list)
        if not tool_des:
            return []
        tool_descriptions = [
            [cont['standard_tool_name'], cont['description']] for cont in tool_des
        ]
        return tool_descriptions

    def fetch_api_json(self, query_json):
        data_dict = {'api_list': []}
        for item in query_json['api_list']:
            cate_name = item['category_name']
            tool_name = standardize(item['tool_name'])
            api_name = change_name(standardize(item['api_name']))
            tool_json = json.load(
                open(
                    os.path.join(self.tool_root_dir, cate_name, tool_name + '.json'),
                    'r',
                )
            )
            append_flag = False
            api_dict_names = []
            for api_dict in tool_json['api_list']:
                api_dict_names.append(api_dict['name'])
                pure_api_name = change_name(standardize(api_dict['name']))
                if pure_api_name != api_name:
                    continue
                api_json = {}
                api_json['category_name'] = cate_name
                api_json['api_name'] = api_dict['name']
                api_json['api_description'] = api_dict['description']
                api_json['required_parameters'] = api_dict['required_parameters']
                api_json['optional_parameters'] = api_dict['optional_parameters']
                api_json['tool_name'] = tool_json['tool_name']
                data_dict['api_list'].append(api_json)
                append_flag = True
                break
            if not append_flag:
                print(api_name, api_dict_names)
        return data_dict

    def api_json_to_openai_json(self, api_json, standard_tool_name):
        description_max_length = 256
        templete = {
            'name': '',
            'description': '',
            'parameters': {
                'type': 'object',
                'properties': {},
                'required': [],
                'optional': [],
            },
        }

        map_type = {'NUMBER': 'integer', 'STRING': 'string', 'BOOLEAN': 'boolean'}

        pure_api_name = change_name(standardize(api_json['api_name']))
        templete['name'] = pure_api_name + f'_for_{standard_tool_name}'
        templete['name'] = templete['name'][-64:]
        openai_function_name = templete['name']

        templete['description'] = (
            f'This is the subfunction for tool "{standard_tool_name}", you can use this tool.'
        )

        if api_json['api_description'].strip() != '':
            tuncated_description = (
                api_json['api_description']
                .strip()
                .replace(api_json['api_name'], templete['name'])[
                    :description_max_length
                ]
            )
            templete['description'] = (
                templete['description']
                + f'The description of this function is: "{tuncated_description}"'
            )
        if (
            'required_parameters' in api_json.keys()
            and len(api_json['required_parameters']) > 0
        ):
            for para in api_json['required_parameters']:
                name = standardize(para['name'])
                name = change_name(name)
                if para['type'] in map_type:
                    param_type = map_type[para['type']]
                else:
                    param_type = 'string'
                prompt = {
                    'type': param_type,
                    'description': para['description'][:description_max_length],
                }

                default_value = para['default']
                if len(str(default_value)) != 0:
                    prompt = {
                        'type': param_type,
                        'description': para['description'][:description_max_length],
                        'example_value': default_value,
                    }
                else:
                    prompt = {
                        'type': param_type,
                        'description': para['description'][:description_max_length],
                    }

                templete['parameters']['properties'][name] = prompt
                templete['parameters']['required'].append(name)
            for para in api_json['optional_parameters']:
                name = standardize(para['name'])
                name = change_name(name)
                if para['type'] in map_type:
                    param_type = map_type[para['type']]
                else:
                    param_type = 'string'

                default_value = para['default']
                if len(str(default_value)) != 0:
                    prompt = {
                        'type': param_type,
                        'description': para['description'][:description_max_length],
                        'example_value': default_value,
                    }
                else:
                    prompt = {
                        'type': param_type,
                        'description': para['description'][:description_max_length],
                    }

                templete['parameters']['properties'][name] = prompt
                templete['parameters']['optional'].append(name)

        return templete, api_json['category_name'], pure_api_name, openai_function_name

    def check_success(self):
        return self.success

    def to_json(self):
        return {}

    def restart(self):
        pass

    def get_score(self):
        return 0.0

    def step(self, action, input_str):
        obs, code = self._step(action_name=action, action_input=input_str)
        if len(obs) > self.max_observation_length:
            obs = obs[: self.max_observation_length] + '...'
        return obs, code

    def _step(self, action_name='', action_input=''):
        """Need to return an observation string and status code:
        0 means normal response
        1 means there is no corresponding api name
        2 means there is an error in the input
        3 represents the end of the generation and the final answer appears
        4 means that the model decides to pruning by itself
        5 represents api call timeout
        6 for 404
        7 means not subscribed
        8 represents unauthorized
        9 represents too many requests
        10 stands for rate limit
        11 message contains "error" field
        12 error sending request
        """
        if action_name == 'Finish':
            try:
                json_data = json.loads(action_input, strict=False)
            except json.JSONDecodeError:
                json_data = {}
                if '"return_type": "' in action_input:
                    if '"return_type": "give_answer"' in action_input:
                        return_type = 'give_answer'
                    elif '"return_type": "give_up_and_restart"' in action_input:
                        return_type = 'give_up_and_restart'
                    else:
                        return_type = action_input[
                            action_input.find('"return_type": "')
                            + len('"return_type": "') : action_input.find('",')
                        ]
                    json_data['return_type'] = return_type
                if '"final_answer": "' in action_input:
                    final_answer = action_input[
                        action_input.find('"final_answer": "')
                        + len('"final_answer": "') :
                    ]
                    json_data['final_answer'] = final_answer
            if 'return_type' not in json_data.keys():
                return '{error:"must have "return_type""}', 2
            if json_data['return_type'] == 'give_up_and_restart':
                return '{"response":"chose to give up and restart"}', 4
            elif json_data['return_type'] == 'give_answer':
                if 'final_answer' not in json_data.keys():
                    return '{error:"must have "final_answer""}', 2

                self.success = 1  # succesfully return final_answer
                return '{"response":"successfully giving the final answer."}', 3
            else:
                return '{error:""return_type" is not a valid choice"}', 2
        else:
            for k, function in enumerate(self.functions):
                if function['name'].endswith(action_name):
                    pure_api_name = self.api_name_reflect[function['name']]
                    payload = {
                        'category': self.cate_names[k],
                        'tool_name': self.tool_names[k],
                        'api_name': pure_api_name,
                        'tool_input': action_input,
                        'strip': self.observ_compress_method,
                        'toolbench_key': self.toolbench_key,
                    }
                    if self.process_id == 0:
                        print(
                            colored(
                                f'query to {self.cate_names[k]}-->{self.tool_names[k]}-->{action_name}',
                                color='yellow',
                            )
                        )
                    if self.use_rapidapi_key or self.api_customization:
                        payload['rapidapi_key'] = self.rapidapi_key
                        response = get_rapidapi_response(
                            payload, api_customization=self.api_customization
                        )
                    else:
                        time.sleep(2)  # rate limit: 30 per minute
                        headers = {'toolbench_key': self.toolbench_key}
                        response = requests.post(
                            self.service_url, json=payload, headers=headers, timeout=15
                        )
                        if response.status_code != 200:
                            return (
                                json.dumps(
                                    {
                                        'error': f'request invalid, data error. status_code={response.status_code}',
                                        'response': '',
                                    }
                                ),
                                12,
                            )
                        try:
                            response = response.json()
                        except json.JSONDecodeError:
                            print(response)
                            return (
                                json.dumps(
                                    {
                                        'error': 'request invalid, data error',
                                        'response': '',
                                    }
                                ),
                                12,
                            )
                    # 1 Hallucinating function names
                    # 4 means that the model decides to pruning by itself
                    # 5 represents api call timeout
                    # 6 for 404
                    # 7 means not subscribed
                    # 8 represents unauthorized
                    # 9 represents too many requests
                    # 10 stands for rate limit
                    # 11 message contains "error" field
                    # 12 error sending request
                    if response['error'] == 'API not working error...':
                        status_code = 6
                    elif response['error'] == 'Unauthorized error...':
                        status_code = 7
                    elif response['error'] == 'Unsubscribed error...':
                        status_code = 8
                    elif response['error'] == 'Too many requests error...':
                        status_code = 9
                    elif response['error'] == 'Rate limit per minute error...':
                        print('Reach api calling limit per minute, sleeping...')
                        time.sleep(10)
                        status_code = 10
                    elif response['error'] == 'Message error...':
                        status_code = 11
                    else:
                        status_code = 0
                    return json.dumps(response), status_code
                    # except Exception as e:
                    #     return json.dumps({"error": f"Timeout error...{e}", "response": ""}), 5
            return (
                json.dumps(
                    {'error': f'No such function name: {action_name}', 'response': ''}
                ),
                1,
            )
