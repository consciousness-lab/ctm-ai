import time
from typing import Any, Dict, List

import litellm
from litellm import completion
from tenacity import retry, stop_after_attempt, wait_random_exponential

from ..messengers import Message
import os


def convert_messages_to_litellm_format(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert CTM messages to LiteLLM format."""
    result = []
    for m in messages:
        msg_text = m.content if m.content is not None else m.query
        if msg_text is None:
            continue
        result.append({'role': m.role, 'content': msg_text})
    return result


@retry(wait=wait_random_exponential(min=1, max=40), stop=stop_after_attempt(3))
def litellm_completion_request(
    messages: List[Message],
    functions=None,
    model: str = 'gpt-4o',
    max_tokens: int = 1024,
    temperature: float = 0.0,
    n: int = 1,
    **kwargs,
):
    """Make a completion request using LiteLLM with retry logic."""
    litellm_messages = convert_messages_to_litellm_format(messages)

    completion_kwargs = {
        'model': model,
        'messages': litellm_messages,
        'max_tokens': max_tokens,
        'temperature': temperature,
        'n': n,
        **kwargs,
    }

    if functions:
        if isinstance(functions, dict):
            completion_kwargs['functions'] = [functions]
        else:
            completion_kwargs['functions'] = functions

    response = completion(**completion_kwargs)
    return response


def convert_message_to_litellm_format(message: Message) -> Dict[str, str]:
    """Convert Message to LiteLLM format."""
    # Use gist if content is None, otherwise use content
    content = message.content if message.content is not None else message.gist
    if content is None:
        raise ValueError('Message content and gist cannot both be None')

    return {'role': message.role, 'content': content}


def call_llm(
    messages: List[Message],
    functions=None,
    function_call=None,
    model: str = 'gpt-4o',
    max_tokens: int = 1024,
    temperature: float = 0.0,
    n: int = 1,
    process_id: int = 0,
    try_times: int = 3,
    **kwargs,
) -> tuple[Dict[str, Any], int, int]:
    """
    Call LLM using LiteLLM with retry logic and error handling.
    Returns: (message_dict, error_code, total_tokens)
    """
    for attempt in range(try_times):
        time.sleep(15)
        try:
            response = litellm_completion_request(
                messages,
                functions=functions,
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                n=n,
                **kwargs,
            )

            message = response.choices[0].message
            total_tokens = response.usage.total_tokens

            if process_id == 0:
                print(f'[process({process_id})] total tokens: {total_tokens}')

            message_dict = {
                'role': message.role,
                'content': message.content,
            }

            # Handle function calls if present
            if hasattr(message, 'function_call') and message.function_call:
                message_dict['function_call'] = {
                    'name': message.function_call.name,
                    'arguments': message.function_call.arguments,
                }

                # Clean up function name if it has dots
                if '.' in message_dict['function_call']['name']:
                    message_dict['function_call']['name'] = message_dict[
                        'function_call'
                    ]['name'].split('.')[-1]

            return message_dict, 0, total_tokens

        except Exception as e:
            print(
                f'[process({process_id})] Attempt {attempt + 1}/{try_times} failed with error: {repr(e)}'
            )

    return (
        {
            'role': 'assistant',
            'content': f'[ERROR] Failed after {try_times} attempts.',
        },
        -1,
        0,
    )


def ask_llm_standard(
    messages: List[Message],
    model: str = 'gpt-4o',
    max_tokens: int = 300,
    temperature: float = 0.0,
    n: int = 5,
    **kwargs,
) -> List[str]:
    """
    Standard LLM asking method for basic text completion using LiteLLM.
    Returns a list of generated responses.
    """
    litellm_messages = convert_messages_to_litellm_format(messages)

    response = completion(
        model=model,
        messages=litellm_messages,
        max_tokens=max_tokens,
        temperature=temperature,
        n=n,
        **kwargs,
    )

    return [response.choices[i].message.content for i in range(len(response.choices))]


def configure_litellm(
    model_name: str = 'gpt-4o',
    success_callbacks: List[str] = None,
    failure_callbacks: List[str] = None,
) -> None:
    """Configure LiteLLM settings and callbacks."""
    # Set default model if not specified
    if not hasattr(litellm, 'model'):
        litellm.model = model_name

    # Configure logging and callbacks if provided
    if success_callbacks:
        litellm.success_callback = success_callbacks
    if failure_callbacks:
        litellm.failure_callback = failure_callbacks
