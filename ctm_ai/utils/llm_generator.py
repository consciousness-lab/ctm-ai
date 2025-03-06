import json
import os
from typing import Any, Dict, List

import litellm


class LLMGenerator:
    def __init__(
        self,
        generator_name: str = 'gpt-4o',
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ):
        self.generator_name = generator_name
        self.max_tokens = max_tokens
        self.temperature = temperature
        self._setup_api_keys()

    def _setup_api_keys(self) -> None:
        if 'gpt' in self.generator_name:
            litellm.api_key = os.getenv('OPENAI_API_KEY')
        elif 'deepseek' in self.generator_name:
            litellm.api_key = os.getenv('DEEPSEEK_API_KEY')
        elif 'claude' in self.generator_name:
            litellm.api_key = os.getenv('ANTHROPIC_API_KEY')
        elif 'gemini' in self.generator_name:
            litellm.api_key = os.getenv('GOOGLE_API_KEY')

    def generate_response(
        self,
        prompt: str,
        system_prompt: str = 'You are an AI assistant. Always return JSON format.',
    ) -> Dict[str, Any]:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': prompt},
        ]

        if 'gpt' in self.generator_name:
            response = litellm.completion(
                model=self.generator_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={'type': 'json_object'},
            )
            return {
                'content': response.choices[0].message['content'].strip(),
                'usage': response['usage'],
            }

        elif 'claude' in self.generator_name:
            response = litellm.completion(
                model=self.generator_name,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={'type': 'json_object'},
            )
            return {
                'content': response.choices[0].message['content'].strip(),
                'usage': response['usage'],
            }

        elif 'gemini' in self.generator_name:
            try:
                response = litellm.completion(
                    model=self.generator_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    response_format={'type': 'json_object'},
                )
                return {
                    'content': response.choices[0].message['content'].strip(),
                    'usage': response['usage'],
                }
            except Exception:
                return self._handle_non_json_model(messages)

        else:
            return self._handle_non_json_model(messages)

    def _handle_non_json_model(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        raw_response = litellm.completion(
            model=self.generator_name,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        try:
            json_content = json.loads(raw_response.choices[0].message['content'])
            return {'content': json.dumps(json_content), 'usage': raw_response['usage']}
        except json.JSONDecodeError:
            # If not valid JSON, use GPT-4o to fix it
            fixed_response = litellm.completion(
                model='gpt-4o',
                messages=[
                    {
                        'role': 'system',
                        'content': 'You are a JSON parsing assistant. The input may contain incorrect JSON-like text. Your task is to fix it and return a valid JSON.',
                    },
                    {
                        'role': 'user',
                        'content': f"Please correct the following JSON data: {raw_response.choices[0].message['content']}",
                    },
                ],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                response_format={'type': 'json_object'},
            )

            return {
                'content': fixed_response.choices[0].message['content'].strip(),
                'usage': {
                    'prompt_tokens': raw_response['usage']['prompt_tokens']
                    + fixed_response['usage']['prompt_tokens'],
                    'completion_tokens': raw_response['usage']['completion_tokens']
                    + fixed_response['usage']['completion_tokens'],
                    'total_tokens': raw_response['usage']['total_tokens']
                    + fixed_response['usage']['total_tokens'],
                },
            }

    def generate_batch_responses(
        self,
        prompts: List[str],
        system_prompt: str = 'You are an AI assistant. Always return JSON format.',
    ) -> List[Dict[str, Any]]:
        return [self.generate_response(prompt, system_prompt) for prompt in prompts]


if __name__ == '__main__':
    # Set environment variables for API keys first
    # os.environ["OPENAI_API_KEY"] = "your-openai-key"
    # os.environ["ANTHROPIC_API_KEY"] = "your-anthropic-key"
    # os.environ["GOOGLE_API_KEY"] = "your-google-key"
    # os.environ["DEEPSEEK_API_KEY"] = "your-deepseek-key"

    # Initialize with different models and parameters
    gpt_generator = LLMGenerator('gpt-4o', max_tokens=2000, temperature=0.5)
    claude_generator = LLMGenerator(
        'claude-3-opus-20240229', max_tokens=1500, temperature=0.7
    )
    gemini_generator = LLMGenerator('gemini-1.5-pro', max_tokens=1000, temperature=0.3)
    deepseek_generator = LLMGenerator('deepseek-chat', max_tokens=800, temperature=0.6)

    # Test generation
    prompt = (
        'Generate a JSON object with information about 3 planets in our solar system.'
    )

    # gpt_response = gpt_generator.generate_response(prompt)
    # print("GPT Response:", gpt_response["content"])

    # claude_response = claude_generator.generate_response(prompt)
    # print("Claude Response:", claude_response["content"])
