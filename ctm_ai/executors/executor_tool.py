import json
from typing import Any, Dict, List, Optional, Union

from ..executors.executor_base import BaseExecutor
from ..messengers.message import Message
from ..tools import (
    call_builtin_tool,
    generate_tool_question,
    get_builtin_tool_definitions,
)
from ..utils.error_handler import message_exponential_backoff


@BaseExecutor.register_executor('tool_executor')
class ToolExecutor(BaseExecutor):
    def __init__(
        self,
        name: str,
        use_builtin_tools: bool = True,
        io_function=None,
        openai_function_names: Optional[Union[str, List[str]]] = None,
        custom_functions: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize ToolExecutor

        Args:
            name: Executor name
            use_builtin_tools: Whether to use builtin tools (default: True)
            io_function: External tool interface (optional)
            openai_function_names: External tool function name list (optional)
            custom_functions: Custom function list (optional)
        """
        super().__init__(name, *args, **kwargs)

        # Initialize tool configuration
        self.io_function = io_function
        self.configured_functions = []

        # Add builtin tools
        if use_builtin_tools:
            self.configured_functions.extend(get_builtin_tool_definitions())

        # Add external tools
        if io_function and openai_function_names:
            if isinstance(openai_function_names, str):
                openai_function_names = [openai_function_names]

            for func_name in openai_function_names:
                try:
                    external_function = io_function.openai_name_reflect_all_info[
                        func_name
                    ][0]
                    self.configured_functions.append(external_function)
                except Exception as e:
                    print(f'Warning: Unable to load external tool {func_name}: {e}')

        # Add custom functions
        if custom_functions:
            self.configured_functions.extend(custom_functions)

        # Save configuration info
        self.use_builtin_tools = use_builtin_tools
        self.openai_function_names = openai_function_names or []

    def init_model(self, *args, **kwargs):
        """Initialize the model using the base class functionality."""
        super().init_model(*args, **kwargs)

    @message_exponential_backoff()
    def ask(
        self,
        messages: List[Message],
        max_token: int = 300,
        return_num: int = 5,
        model: str = None,
        *args: Any,
        **kwargs: Any,
    ) -> Message:
        """
        Simple ask method for unified tool calling

        Args:
            messages: Message list
            max_token: Maximum tokens for response
            return_num: Number of response candidates
            model: Model name to use

        Returns:
            Message containing response or function result with additional question
        """
        # Call LLM for tool selection and invocation
        model = model or self.model_name
        new_message, error_code, total_tokens = self.call_llm(
            messages,
            functions=self.configured_functions,
            model=model,
            max_tokens=max_token,
            n=return_num,
            **kwargs,
        )

        assert new_message['role'] == 'assistant'

        # Handle normal content response (no tool used)
        if 'content' in new_message.keys() and new_message['content'] is not None:
            additional_question = self._generate_general_question(
                messages, new_message['content']
            )
            return Message(
                role='assistant',
                content=new_message['content'],
                gist=new_message['content'],
                additional_question=additional_question,
            )

        # Handle tool call response
        if 'function_call' in new_message.keys():
            function_call_data = new_message['function_call']
            assert isinstance(function_call_data, dict)

            function_name = function_call_data['name']
            function_arguments = function_call_data['arguments']

            # Parse arguments
            try:
                if isinstance(function_arguments, str):
                    arguments = json.loads(function_arguments)
                else:
                    arguments = function_arguments
            except json.JSONDecodeError as e:
                error_msg = f'Parameter parsing failed: {str(e)}'
                return Message(
                    role='function',
                    content=error_msg,
                    gist=error_msg,
                    additional_question='Please check your request format and try again.',
                )

            # Handle builtin tool calls
            builtin_tool_names = [
                tool['name'] for tool in get_builtin_tool_definitions()
            ]
            if function_name in builtin_tool_names:
                return self._handle_builtin_tool_call(
                    messages, function_name, arguments
                )

            # Handle external tool calls
            elif self.io_function:
                return self._handle_external_tool_call(
                    messages, self.io_function, function_name, function_arguments
                )

            # Unknown tool
            available_tool_names = [func['name'] for func in self.configured_functions]
            error_msg = f'Unknown tool: {function_name}'
            return Message(
                role='function',
                content=error_msg,
                gist=error_msg,
                additional_question=f"Available tools include: {', '.join(available_tool_names)}. Please select a valid tool.",
            )

        # Error case
        error_msg = '[ERROR] Model did not return valid response'
        return Message(
            role='assistant',
            content=error_msg,
            gist=error_msg,
            additional_question="Please rephrase your request and I'll try to understand and help you better.",
        )

    def add_custom_function(self, function_definition: Dict[str, Any]):
        """Dynamically add custom function"""
        self.configured_functions.append(function_definition)

    def add_external_tool(self, io_function, openai_function_name: str):
        """Dynamically add external tool"""
        try:
            external_function = io_function.openai_name_reflect_all_info[
                openai_function_name
            ][0]
            self.configured_functions.append(external_function)
            if openai_function_name not in self.openai_function_names:
                self.openai_function_names.append(openai_function_name)
            if not self.io_function:
                self.io_function = io_function
        except Exception as e:
            print(f'Unable to add external tool {openai_function_name}: {e}')

    def remove_function(self, function_name: str):
        """Remove specified function"""
        self.configured_functions = [
            func for func in self.configured_functions if func['name'] != function_name
        ]
        if function_name in self.openai_function_names:
            self.openai_function_names.remove(function_name)

    def _handle_builtin_tool_call(
        self, messages: List[Message], function_name: str, arguments: Dict[str, Any]
    ) -> Message:
        """Handle builtin tool calls"""
        try:
            # Call builtin tool
            observation = call_builtin_tool(function_name, arguments)

            # Get original query for question generation
            original_query = ''
            for msg in messages:
                if msg.content and msg.role == 'user':
                    original_query = msg.content
                    break

            # Generate tool-specific follow-up question
            additional_question = generate_tool_question(
                function_name, original_query, observation
            )

            return Message(
                role='function',
                content=observation,
                gist=observation,
                additional_question=additional_question,
            )

        except Exception as e:
            error_msg = f'Builtin tool call failed: {str(e)}'
            return Message(
                role='function',
                content=error_msg,
                gist=error_msg,
                additional_question=f'Tool {function_name} encountered an issue. Please check your input and try again.',
            )

    def _handle_external_tool_call(
        self,
        messages: List[Message],
        io_function,
        function_name: str,
        function_arguments: str,
    ) -> Message:
        """Handle external tool calls"""
        try:
            observation, status = io_function.step(function_name, function_arguments)

            # Get original query
            original_query = ''
            for msg in messages:
                if msg.content and msg.role == 'user':
                    original_query = msg.content
                    break

            # Generate general question for external tools
            additional_question = f'Based on the {function_name} tool result, what other related questions can I help you with?'

            return Message(
                role='function',
                content=observation,
                gist=observation,
                additional_question=additional_question,
            )

        except Exception as e:
            error_msg = f'External tool call failed: {str(e)}'
            return Message(
                role='function',
                content=error_msg,
                gist=error_msg,
                additional_question=f'Tool {function_name} encountered an issue. Please check your input and try again.',
            )

    def _generate_general_question(self, messages: List[Message], response: str) -> str:
        """Generate follow-up question for general responses"""
        # Get original query
        original_query = ''
        for msg in messages:
            if msg.content and msg.role == 'user':
                original_query = msg.content
                break

        # Generate general follow-up questions
        general_questions = [
            f"Regarding '{original_query}', what specific aspects would you like to know more about?",
            'Would you like me to use search or calculation tools to get more information?',
            'Which part would you like me to explain in more detail?',
        ]

        return general_questions[0]  # Simply return the first question

    def get_available_functions(self) -> List[Dict[str, Any]]:
        """Get all currently configured available functions"""
        return self.configured_functions.copy()

    def list_function_names(self) -> List[str]:
        """List all available function names"""
        return [func['name'] for func in self.configured_functions]

    def get_configuration(self) -> Dict[str, Any]:
        """Get current configuration information"""
        return {
            'use_builtin_tools': self.use_builtin_tools,
            'openai_function_names': self.openai_function_names,
            'total_functions': len(self.configured_functions),
            'function_names': self.list_function_names(),
        }
