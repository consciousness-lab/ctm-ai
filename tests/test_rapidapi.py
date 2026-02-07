"""
LiteLLM + MCP 直接调用版本
给一个 prompt，返回结果，完事
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional

import litellm
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class TwitterMCPAgent:
    """简单的 Twitter MCP Agent"""

    def __init__(self, model: str = 'gpt-4o-mini'):
        self.model = model
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict] = []

        self.system_prompt = """你是一个 Twitter/X 数据分析助手。
使用工具获取数据，然后用中文简洁地总结关键信息。"""

    async def __aenter__(self):
        """连接 MCP"""
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
        """断开连接"""
        if self.session:
            await self._session_context.__aexit__(*args)
            await self._streams_context.__aexit__(*args)

    def _get_tools_for_llm(self) -> List[Dict]:
        """转换工具格式"""
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
        """调用 MCP 工具"""
        result = await self.session.call_tool(name, args)
        if result.content:
            return '\n'.join(
                item.text if hasattr(item, 'text') else str(item)
                for item in result.content
            )
        return ''

    async def run(self, prompt: str) -> str:
        """
        执行 prompt，返回结果

        Args:
            prompt: 用户的问题/指令

        Returns:
            LLM 的回复
        """
        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ]
        tools = self._get_tools_for_llm()

        # 调用 LLM
        response = await litellm.acompletion(
            model=self.model, messages=messages, tools=tools, tool_choice='auto'
        )

        assistant_message = response.choices[0].message

        # 处理工具调用
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

            # 执行工具
            for tc in assistant_message.tool_calls:
                result = await self._call_tool(
                    tc.function.name, json.loads(tc.function.arguments)
                )
                messages.append(
                    {'role': 'tool', 'tool_call_id': tc.id, 'content': result[:10000]}
                )

            # 再次调用 LLM
            response = await litellm.acompletion(
                model=self.model, messages=messages, tools=tools, tool_choice='auto'
            )
            assistant_message = response.choices[0].message

        return assistant_message.content or ''


# ==================== 直接调用函数 ====================


async def query(prompt: str, model: str = 'gpt-4o-mini') -> str:
    """
    一行代码调用

    Example:
        result = await query("搜索 MrBeast 的最新推文")
    """
    async with TwitterMCPAgent(model=model) as agent:
        return await agent.run(prompt)


def query_sync(prompt: str, model: str = 'gpt-4o-mini') -> str:
    """
    同步版本

    Example:
        result = query_sync("搜索 MrBeast 的最新推文")
    """
    return asyncio.run(query(prompt, model))


# ==================== 直接运行 ====================

if __name__ == '__main__':
    # 设置你的 prompt
    PROMPT = 'songs made by John Williams'

    # 设置模型 (可选)
    MODEL = os.environ.get('LLM_MODEL', 'gpt-4o-mini')

    print(f'Prompt: {PROMPT}')
    print(f'Model: {MODEL}')
    print('-' * 50)

    result = query_sync(PROMPT, MODEL)
    print(result)
