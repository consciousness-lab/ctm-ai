import os
import json
from typing import Any, Callable, Dict, List, Optional, Tuple, Type
import numpy as np
from litellm import completion
from numpy.typing import NDArray
from .processor_base import BaseProcessor
from litellm import completion
from ..scorers import ToolScorer
from ..chunks import Chunk
from .utils import parse_json_response

TOOL_SYSTEM_PROMPT = """
You are a tool agent designed to help users by utilizing available tools to answer their queries or complete tasks.
"""
TOOL_ANSWER_PROMPT = """
Regarding to the task: {query}, the answer of the function call is: {new_message}. You should utilize the information in the history and the answer of the function call to answer the query. Provide specific information if you can, do not just say you successfully called it.
There might have some answers to other queries and extra information, if you think it is useful, you should utilize them to provide more comprehensive answer to the query. If you think you should use the information of another apis or tools, you should ask like "what is the results of calling the api of `API_NAME` for more answers instead of asking for the response format of another api endpoint.
Please respond in JSON format with the following structure:
{{
    "response": "Your detailed response to the query",
    "additional_question": "If you are not sure about the answer, you should generate a question that potentially can be answered by other tools."
}}

Your additional_question should be potentially answerable by other tools like search engine and about specific information that you are not sure about.
Your additional_question should be just about what kind of information you need to get from other tools like search engine, nothing else about the task or original query should be included. For example, what is the weather in the city, what is the stock price of the company, etc. The question needs to be short and clean.
"""

TEXT_ANSWER_PROMPT = """
Regarding to the task: {query}, the answer of the model is: {new_message}. Based on the answer, do you have other questions? If you have other questions, you should generate a question that potentially can be answered by other tools. 
You should generate your response based on the extra information and previous answers, and the answer of the current model. answer as speccific as you can.
Please respond in JSON format with the following structure:
{{
    "response": "Your detailed response to the query",
    "additional_question": "If you are not sure about the answer, you should generate a question that potentially can be answered by other tools."
}}

Your additional_question should be potentially answerable by other tools like search engine and about specific information that you are not sure about.
Your additional_question should be just about what kind of information you need to get from other tools like search engine, nothing else about the task or original query should be included. For example, what is the weather in the city, what is the stock price of the company, etc. The question needs to be short and clean.
"""


def clean_tools_for_vertex_ai(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Clean tool definitions to remove fields not supported by Vertex AI
    """
    cleaned_tools = []
    for tool in tools:
        if tool.get('type') == 'function' and 'function' in tool:
            function_def = tool['function'].copy()

            # Clean parameters
            if 'parameters' in function_def:
                params = function_def['parameters'].copy()

                # Remove 'optional' field
                if 'optional' in params:
                    del params['optional']

                # Clean properties to remove 'example_value'
                if 'properties' in params:
                    cleaned_properties = {}
                    for prop_name, prop_def in params['properties'].items():
                        cleaned_prop = prop_def.copy()
                        if 'example_value' in cleaned_prop:
                            del cleaned_prop['example_value']
                        cleaned_properties[prop_name] = cleaned_prop
                    params['properties'] = cleaned_properties

                function_def['parameters'] = params

            cleaned_tool = {'type': 'function', 'function': function_def}
            cleaned_tools.append(cleaned_tool)
        else:
            cleaned_tools.append(tool)

    return cleaned_tools


def register_tool_processors(openai_function_names: List[str]):
    for openai_function_name in openai_function_names:
        processor_name = openai_function_name
        BaseProcessor._processor_registry[processor_name] = ToolProcessor


@BaseProcessor.register_processor('tool_processor')
class ToolProcessor(BaseProcessor):
    REQUIRED_KEYS = ['OPENAI_API_KEY']

    def _build_executor_content(
        self,
        query: str,
        api_manager: Any = None,
        function_name: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        content = f'Task Description: {query}\n'

        content += f"""
You should utilize the information in the context history and the tool `{function_name}` to solve the task.
In the context history, there might have some answers to the task, or some information you can use to call the tool `{function_name}`, you should utilize them to better solve and answer the task. 

DECISION:
- First decide whether to call the tool `{function_name}`.
- If the tool helps even partially or it is one of the steps/tools to solve the task, CALL IT.
- If the tool does not help at all, or you think the context history already provides enough information to answer the task, answer directly, provide comprehensive answer to the task.

OUTPUT PROTOCOL (MUST follow strictly):
- If you CALL the tool:
  - Return ONLY a function call via tool_calls.
  - Set assistant.content to null (no natural-language text).
  - Do NOT include any text explanation.
- If you DO NOT call the tool:
  - Return ONLY a natural-language answer in assistant.content.
  - Do NOT include tool_calls.
  - Include all the information you think is useful to answer the task in the extra information and previous answers.
"""

        if len(self.fuse_history) > 0:
            content += '\nThere are extra information from other tools:\n'
            for i, item in enumerate(self.fuse_history, 1):
                content += f'{i}. {item["answer"]}\n'

        if len(self.winner_answer) > 0:
            content += '\nThere are some previous answers to the same query, think further based on this answer:\n'
            for i, item in enumerate(self.winner_answer, 1):
                content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'
        return content

    def ask_executor(
        self,
        query: str,
        api_manager: Any = None,
        function_name: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, str]:
        messages = [
            {'role': 'system', 'content': TOOL_SYSTEM_PROMPT},
            {'role': 'user', 'content': query},
        ]

        # Check if api_manager is None
        if api_manager is None:
            raise ValueError(
                f'api_manager is None for function {function_name}. This should not happen in tool processors.'
            )

        # Clean tools for Vertex AI compatibility
        raw_tools = api_manager.funcs_to_all_info[function_name]
        cleaned_tools = clean_tools_for_vertex_ai(raw_tools)

        response = completion(
            model=self.model,
            messages=messages,
            tools=cleaned_tools,
            tool_choice='auto',
        )
        response_message = response.choices[0].message
        retry_times = 3

        # Initialize default values
        gist = ''
        additional_question = ''

        if response_message.content is not None and response_message.tool_calls is None:
            execuotr_answer = response_message.content
            text_prompt = TEXT_ANSWER_PROMPT.format(
                query=query, new_message=execuotr_answer
            )
            if len(self.fuse_history) > 0:
                text_prompt += '\nThere are extra information from other tools:\n'
                for i, item in enumerate(self.fuse_history, 1):
                    text_prompt += f'{i}. {item["answer"]}\n'

            if len(self.winner_answer) > 0:
                text_prompt += '\nThere are some previous answers to the same query, think further based on this answer:\n'
                for i, item in enumerate(self.winner_answer, 1):
                    text_prompt += f'{i}. {item["processor_name"]}: {item["answer"]}\n'
            response = completion(
                model=self.model,
                messages=[{'role': 'user', 'content': text_prompt}],
                *args,
                **kwargs,
            )
            gist, additional_question = parse_json_response(
                response.choices[0].message.content, 'Can you provide more information?'
            )
        elif (
            response_message.tool_calls is not None and response_message.content is None
        ):
            tool_call = response_message.tool_calls[0]
            func_name = getattr(tool_call.function, 'name', None) or function_name
            func_args = getattr(tool_call.function, 'arguments', '{}')

            if isinstance(func_args, dict):
                function_args = json.dumps(func_args, ensure_ascii=False)
            else:
                function_args = str(func_args) if func_args is not None else '{}'
            for i in range(retry_times):
                try:
                    tool_answer, status_code = api_manager.step(
                        action=func_name, input_str=function_args
                    )
                    if status_code in (0, 3):
                        break
                except Exception as e:
                    if i < retry_times - 1:
                        continue
                    else:
                        tool_answer = {
                            'error': f'tool execution failed: {type(e).__name__}: {e}',
                            'response': '',
                        }
            tool_prompt = TOOL_ANSWER_PROMPT.format(
                query=query, new_message=tool_answer
            )
            if len(self.fuse_history) > 0:
                tool_prompt += '\nThere are extra information from other tools:\n'
                for i, item in enumerate(self.fuse_history, 1):
                    tool_prompt += f'{i}. {item["answer"]}\n'

            if len(self.winner_answer) > 0:
                tool_prompt += '\nThere are some previous answers to the same query, think further based on this answer:\n'
                for i, item in enumerate(self.winner_answer, 1):
                    tool_prompt += f'{i}. {item["processor_name"]}: {item["answer"]}\n'
            response = completion(
                model=self.model,
                messages=[{'role': 'user', 'content': tool_prompt}],
                *args,
                **kwargs,
            )
            gist, additional_question = parse_json_response(
                response.choices[0].message.content, 'Can you provide more information?'
            )
        else:
            # Handle case where both content and tool_calls are None, or both have values
            # This is an unexpected case, provide a default response
            gist = 'No valid response received from the model'
            additional_question = 'Can you provide more information?'

        return {
            'response': gist,
            'additional_question': additional_question,
        }

    def ask(
        self,
        query: str,
        api_manager: Any = None,
        is_fuse: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        content = self._build_executor_content(
            query=query,
            api_manager=api_manager,
            function_name=self.name,
        )
        executor_output = self.ask_executor(
            query=content,
            api_manager=api_manager,
            function_name=self.name,
        )
        if is_fuse:
            self.add_fuse_history(query, executor_output['response'])
        self.add_all_context_history(
            query,
            executor_output['response'],
            executor_output['additional_question'],
        )
        scorer = ToolScorer(*args, **kwargs)
        scorer_output = scorer.ask(query=query, messages=executor_output)
        additional_question = executor_output['additional_question'] or ''
        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )
        return chunk
