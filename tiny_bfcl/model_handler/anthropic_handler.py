"""
Anthropic model handler for Tiny BFCL
Handles interactions with Anthropic API for Claude models
"""

import time
from typing import Any, Dict, List, Tuple

import anthropic
from anthropic import Anthropic

from .base_handler import BaseHandler


class AnthropicHandler(BaseHandler):
    """Anthropic model handler"""

    def __init__(self, model_config: Dict[str, Any]):
        super().__init__(model_config)
        self.client = Anthropic()

    def _pre_query_processing_FC(
        self, inference_data: Dict[str, Any], test_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pre-processing for FC mode"""
        inference_data['message'] = []
        return inference_data

    def _compile_tools(
        self, inference_data: Dict[str, Any], test_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compile tools for Anthropic format"""
        functions = test_entry['function']
        tools = []

        for func in functions:
            tool = {
                'name': func['name'],
                'description': func.get('description', ''),
                'input_schema': func.get('parameters', {}),
            }
            tools.append(tool)

        inference_data['tools'] = tools
        return inference_data

    def _query_FC(self, inference_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Query for FC mode"""
        message = inference_data['message']
        tools = inference_data['tools']

        inference_data['inference_input_log'] = {
            'message': repr(message),
            'tools': tools,
        }

        kwargs = {
            'messages': message,
            'model': self.model_name,
            'temperature': self.temperature,
        }

        if len(tools) > 0:
            kwargs['tools'] = tools

        start_time = time.time()
        response = self.client.messages.create(**kwargs)
        end_time = time.time()

        return response, end_time - start_time

    def _parse_query_response_FC(self, api_response) -> Dict[str, Any]:
        """Parse FC mode response"""
        model_responses = []
        tool_call_ids = []

        if api_response.content and len(api_response.content) > 0:
            for content in api_response.content:
                if hasattr(content, 'type') and content.type == 'tool_use':
                    model_responses.append({content.name: content.input})
                    tool_call_ids.append(content.id)

        if not model_responses:
            model_responses = (
                api_response.content[0].text if api_response.content else ''
            )

        return {
            'model_responses': model_responses,
            'model_responses_message_for_chat_history': api_response,
            'tool_call_ids': tool_call_ids,
            'reasoning_content': '',
            'input_token': api_response.usage.input_tokens,
            'output_token': api_response.usage.output_tokens,
        }

    def add_first_turn_message_FC(
        self, inference_data: Dict[str, Any], first_turn_message: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add first turn message"""
        inference_data['message'].extend(first_turn_message)
        return inference_data

    def inference_single_turn_FC(
        self, test_entry: Dict[str, Any], include_input_log: bool = False
    ):
        """Single turn inference for FC mode"""
        inference_data = {}
        inference_data = self._pre_query_processing_FC(inference_data, test_entry)
        inference_data = self._compile_tools(inference_data, test_entry)
        inference_data = self.add_first_turn_message_FC(
            inference_data, test_entry['question'][0]
        )

        api_response, query_latency = self._query_FC(inference_data)
        model_response_data = self._parse_query_response_FC(api_response)

        # Process metadata
        metadata = {}
        if include_input_log:
            metadata['inference_log'] = [
                {
                    'role': 'inference_input',
                    'content': inference_data.get('inference_input_log', ''),
                }
            ]
        metadata['input_token_count'] = model_response_data['input_token']
        metadata['output_token_count'] = model_response_data['output_token']
        metadata['latency'] = query_latency

        if (
            'reasoning_content' in model_response_data
            and model_response_data['reasoning_content'] != ''
        ):
            metadata['reasoning_content'] = model_response_data['reasoning_content']

        return model_response_data['model_responses'], metadata

    # Prompting mode methods (simplified implementation)
    def _pre_query_processing_prompting(
        self, test_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Pre-processing for prompting mode"""
        return {'message': []}

    def _query_prompting(self, inference_data: Dict[str, Any]) -> Tuple[Any, float]:
        """Query for prompting mode"""
        inference_data['inference_input_log'] = {
            'message': repr(inference_data['message'])
        }

        kwargs = {
            'messages': inference_data['message'],
            'model': self.model_name,
            'temperature': self.temperature,
        }

        start_time = time.time()
        response = self.client.messages.create(**kwargs)
        end_time = time.time()

        return response, end_time - start_time

    def _parse_query_response_prompting(self, api_response) -> Dict[str, Any]:
        """Parse prompting mode response"""
        content = api_response.content[0].text if api_response.content else ''
        return {
            'model_responses': content,
            'model_responses_message_for_chat_history': api_response,
            'reasoning_content': '',
            'input_token': api_response.usage.input_tokens,
            'output_token': api_response.usage.output_tokens,
        }

    def add_first_turn_message_prompting(
        self, inference_data: Dict[str, Any], first_turn_message: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Add first turn message"""
        inference_data['message'].extend(first_turn_message)
        return inference_data

    def inference_single_turn_prompting(
        self, test_entry: Dict[str, Any], include_input_log: bool = False
    ):
        """Single turn inference for prompting mode"""
        inference_data = self._pre_query_processing_prompting(test_entry)
        inference_data = self.add_first_turn_message_prompting(
            inference_data, test_entry['question'][0]
        )

        api_response, query_latency = self._query_prompting(inference_data)
        model_response_data = self._parse_query_response_prompting(api_response)

        # Process metadata
        metadata = {}
        if include_input_log:
            metadata['inference_log'] = [
                {
                    'role': 'inference_input',
                    'content': inference_data.get('inference_input_log', ''),
                }
            ]
        metadata['input_token_count'] = model_response_data['input_token']
        metadata['output_token_count'] = model_response_data['output_token']
        metadata['latency'] = query_latency

        if (
            'reasoning_content' in model_response_data
            and model_response_data['reasoning_content'] != ''
        ):
            metadata['reasoning_content'] = model_response_data['reasoning_content']

        return model_response_data['model_responses'], metadata
