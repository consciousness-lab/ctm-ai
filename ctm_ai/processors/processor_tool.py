import json
from typing import Any, Dict, List

from litellm import completion

from ..chunks import Chunk
from .processor_base import BaseProcessor
from .utils import JSON_FORMAT_SCORE, parse_json_response_with_scores

TOOL_SYSTEM_PROMPT = """
You are a tool agent designed to help users by utilizing available tools to answer their queries or complete tasks.
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

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """ToolProcessor doesn't use this method as it has custom tool execution flow."""
        # This method is required by BaseProcessor but not used in custom ask()
        return None

    def ask_executor(
        self,
        query: str,
        api_manager: Any = None,
        function_name: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Execute tool and get structured response with self-evaluation scores."""
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
            **self._completion_kwargs,
            messages=messages,
            tools=cleaned_tools,
            tool_choice='auto',
        )
        response_message = response.choices[0].message
        retry_times = 3

        # Process response based on whether tool was called or not
        if response_message.content is not None and response_message.tool_calls is None:
            # Case 1: Model provided direct answer without calling tool
            executor_answer = response_message.content
            structured_output = self._build_structured_output_from_text(
                query, executor_answer, *args, **kwargs
            )
        elif (
            response_message.tool_calls is not None and response_message.content is None
        ):
            # Case 2: Model called the tool
            tool_call = response_message.tool_calls[0]
            func_name = getattr(tool_call.function, 'name', None) or function_name
            func_args = getattr(tool_call.function, 'arguments', '{}')

            if isinstance(func_args, dict):
                function_args = json.dumps(func_args, ensure_ascii=False)
            else:
                function_args = str(func_args) if func_args is not None else '{}'

            # Execute the tool with retries
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

            structured_output = self._build_structured_output_from_tool(
                query, tool_answer, *args, **kwargs
            )
        else:
            # Handle unexpected case
            structured_output = {
                'response': 'No valid response received from the model',
                'additional_question': 'Can you provide more information?',
                'relevance': 0.5,
                'confidence': 0.5,
                'surprise': 0.5,
            }

        return structured_output

    def _build_structured_output_from_text(
        self, query: str, text_answer: str, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Build structured output with scores when model provides direct text answer."""
        context_info = self._get_context_info()

        prompt = f"""Regarding the task: {query}

The model's direct answer:
{text_answer}
{context_info}

Based on this answer, please:
1. Provide your final response to the task
2. Generate an additional question if you need information from other tools (e.g., "what is the weather in the city?", "what is the stock price?"). If no additional information is needed, leave it empty.
3. Self-evaluate your response with relevance, confidence, and surprise scores.

{JSON_FORMAT_SCORE}"""

        return self._ask_for_structured_output(prompt, *args, **kwargs)

    def _build_structured_output_from_tool(
        self, query: str, tool_answer: Any, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Build structured output with scores after tool execution."""
        context_info = self._get_context_info()

        prompt = f"""Regarding the task: {query}

The tool execution result:
{tool_answer}
{context_info}

Based on the tool result, please:
1. Provide a comprehensive response to the task (be specific, don't just say you called the tool successfully)
2. Generate an additional question if you need information from other tools. If no additional information is needed, leave it empty.
3. Self-evaluate your response with relevance, confidence, and surprise scores.

{JSON_FORMAT_SCORE}"""

        return self._ask_for_structured_output(prompt, *args, **kwargs)

    def _get_context_info(self) -> str:
        """Get formatted context information from history."""
        context_info = ''
        if len(self.fuse_history) > 0:
            context_info += '\nExtra information from other tools:\n'
            for i, item in enumerate(self.fuse_history, 1):
                context_info += f'{i}. {item["answer"]}\n'

        if len(self.winner_answer) > 0:
            context_info += '\nPrevious answers to consider:\n'
            for i, item in enumerate(self.winner_answer, 1):
                context_info += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        return context_info

    def _ask_for_structured_output(
        self, prompt: str, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get structured output with self-evaluation scores using litellm."""
        call_kwargs = {
            **self._completion_kwargs,
            'messages': [{'role': 'user', 'content': prompt}],
            'max_tokens': self.max_tokens,
            'n': self.return_num,
            'temperature': self.temperature,
            **kwargs,
        }
        response = completion(**call_kwargs)

        content = response.choices[0].message.content

        # Parse the JSON response with scores
        parsed = parse_json_response_with_scores(
            content, default_additional_question='Can you provide more information?'
        )

        return parsed

    def ask(
        self,
        query: str,
        api_manager: Any = None,
        is_fuse: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        """Process tool-based queries with self-evaluation scoring."""
        content = self._build_executor_content(
            query=query,
            api_manager=api_manager,
            function_name=self.name,
        )

        executor_output = self.ask_executor(
            query=content,
            api_manager=api_manager,
            function_name=self.name,
            *args,
            **kwargs,
        )

        if is_fuse:
            self.add_fuse_history(query, executor_output['response'])

        self.add_all_context_history(
            query,
            executor_output['response'],
            executor_output.get('additional_question', ''),
        )

        # Extract scores using the base processor method
        scorer_output = self._extract_scores_from_executor_output(executor_output)
        additional_question = executor_output.get('additional_question', '')

        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )
        return chunk
