import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import litellm
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class TwitterMCPAgent:
    """Simple Twitter MCP Agent"""

    def __init__(self, model: str = 'gpt-4o-mini'):
        self.model = model
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict] = []

        self.system_prompt = """You are a Twitter/X data analysis assistant.
Use tools to retrieve data, then provide concise summaries of key information."""

    async def __aenter__(self):
        """Connect to MCP"""
        server_params = StdioServerParameters(
            command='npx',
            args=[
                'mcp-remote',
                'https://mcp.rapidapi.com',
                '--header',
                'x-api-host: spotify-downloader9.p.rapidapi.com',
                '--header',
                'x-api-key: e8164c5895msh0c84e9103a0cd26p19ff47jsn4b0706614d59',
            ],
        )

        self._streams_context = stdio_client(server_params)
        streams = await self._streams_context.__aenter__()

        self._session_context = ClientSession(*streams)
        self.session = await self._session_context.__aenter__()

        await self.session.initialize()

        tools_response = await self.session.list_tools()
        self.tools = tools_response.tools

        return self

    async def __aexit__(self, *args):
        """Disconnect from MCP"""
        if self.session:
            await self._session_context.__aexit__(*args)
            await self._streams_context.__aexit__(*args)

    def _get_tools_for_llm(self) -> List[Dict]:
        """Convert tool format for LLM"""
        return [
            {
                'type': 'function',
                'function': {
                    'name': tool.name,
                    'description': tool.description,
                    'parameters': getattr(
                        tool, 'inputSchema', {'type': 'object', 'properties': {}}
                    ),
                },
            }
            for tool in self.tools
        ]

    async def _call_tool(self, name: str, args: Dict) -> str:
        """Call MCP tool"""
        result = await self.session.call_tool(name, args)
        if result.content:
            return '\n'.join(
                item.text if hasattr(item, 'text') else str(item)
                for item in result.content
            )
        return ''

    async def run(self, prompt: str) -> str:
        """
        Execute prompt and return result

        Args:
            prompt: User's question/instruction

        Returns:
            LLM's response
        """
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ]
        tools = self._get_tools_for_llm()

        # Call LLM
        response = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, tool_choice='auto'
        )

        assistant_message = response.choices[0].message

        # Handle tool calls
        while assistant_message.tool_calls:
            messages.append(
                {
                    'role': 'assistant',
                    'content': assistant_message.content,
                    'tool_calls': [
                        {
                            'id': tc.id,
                            'type': 'function',
                            'function': {
                                'name': tc.function.name,
                                'arguments': tc.function.arguments,
                            },
                        }
                        for tc in assistant_message.tool_calls
                    ],
                }
            )

            # Execute tools
            for tc in assistant_message.tool_calls:
                result = await self._call_tool(
                    tc.function.name, json.loads(tc.function.arguments)
                )
                messages.append(
                    {'role': 'tool', 'tool_call_id': tc.id, 'content': result[:10000]}
                )

            # Call LLM again
            response = await litellm.acompletion(
                model=self.model, messages=messages, tools=tools, tool_choice='auto'
            )
            assistant_message = response.choices[0].message

        return assistant_message.content or ''


# ==================== Direct Function Call ====================


async def query(prompt: str, model: str = 'gpt-4o-mini') -> str:
    """
    One-line call

    Example:
        result = await query("Search for MrBeast's latest tweets")
    """
    async with TwitterMCPAgent(model=model) as agent:
        return await agent.run(prompt)


def query_sync(prompt: str, model: str = 'gpt-4o-mini') -> str:
    """
    Synchronous version

    Example:
        result = query_sync("Search for MrBeast's latest tweets")
    """
    return asyncio.run(query(prompt, model))


# ==================== Direct Execution ====================

if __name__ == '__main__':
    # Set your prompt
    PROMPT = 'songs made by John Williams'

    # Set model (optional)
    MODEL = os.environ.get('LLM_MODEL', 'gpt-4o-mini')

    print(f'Prompt: {PROMPT}')
    print(f'Model: {MODEL}')
    print('-' * 50)

    result = query_sync(PROMPT, MODEL)
    print(result)
