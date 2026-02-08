import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from litellm import completion
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ...chunks import Chunk
from ..processor_base import BaseProcessor

# Prompt for generating GeoDB search query
GEODB_QUERY_PROMPT = """Based on the following information, generate a GeoDB Cities search query (in English, max 50 characters) that would help find relevant geographic data about cities, regions, countries, or islands.

Original Query: {query}
Additional Information: {history_info}

Generate ONLY the search query text, nothing else. Your query should focus on geographic locations, city names, country names, or region names.
Examples: "Tokyo Japan", "cities in France", "population New York"
"""

# Prompt for generating additional question
GEODB_ADDITIONAL_PROMPT = """You should utilize the information in the context history and the current response from GeoDB Cities to generate an additional question about the query. Your additional question should be potentially answerable by other modality models and about specific information that you are not sure about. Your additional question should be just about what kind of information you need to get from other modality models or other tools, nothing else about the task or original query should be included. The question needs to be short and clean.

Original Query: {query}
GeoDB Response: {response}
Additional Information: {history_info}
"""


class GeoDBMCPAgent:
    """GeoDB Cities MCP Agent for geographic data retrieval"""

    def __init__(self, rapidapi_key: str, model: str = 'gpt-4o-mini'):
        self.model = model
        self.rapidapi_key = rapidapi_key
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict] = []
        self.system_prompt = """You are a geographic data retrieval assistant using GeoDB Cities API.
Use the tools to fetch city, region, country, and island data. Summarize the key information concisely in English.
You can search places by name, filter by country, get population data, elevation, timezone, and distances between places."""

    async def __aenter__(self):
        server_params = StdioServerParameters(
            command='npx',
            args=[
                'mcp-remote',
                'https://mcp.rapidapi.com',
                '--header',
                'x-api-host: wft-geo-db.p.rapidapi.com',
                '--header',
                f'x-api-key: {self.rapidapi_key}',
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
        if self.session:
            await self._session_context.__aexit__(*args)
            await self._streams_context.__aexit__(*args)

    def _get_tools_for_llm(self) -> List[Dict]:
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
        result = await self.session.call_tool(name, args)
        if result.content:
            return '\n'.join(
                item.text if hasattr(item, 'text') else str(item)
                for item in result.content
            )
        return ''

    async def run(self, prompt: str) -> str:
        from litellm import acompletion

        messages = [
            {'role': 'system', 'content': self.system_prompt},
            {'role': 'user', 'content': prompt},
        ]
        tools = self._get_tools_for_llm()

        response = await acompletion(
            model=self.model, messages=messages, tools=tools, tool_choice='auto'
        )

        assistant_message = response.choices[0].message

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

            for tc in assistant_message.tool_calls:
                result = await self._call_tool(
                    tc.function.name, json.loads(tc.function.arguments)
                )
                messages.append(
                    {'role': 'tool', 'tool_call_id': tc.id, 'content': result[:10000]}
                )

            response = await acompletion(
                model=self.model, messages=messages, tools=tools, tool_choice='auto'
            )
            assistant_message = response.choices[0].message

        return assistant_message.content or ''


@BaseProcessor.register_processor('geodb_processor')
class GeoDBProcessor(BaseProcessor):
    REQUIRED_KEYS = ['RAPIDAPI_KEY']

    def __init__(
        self, name: str, group_name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(name, group_name, *args, **kwargs)
        self.rapidapi_key = os.environ.get('RAPIDAPI_KEY', '')
        self.geodb_model = kwargs.get('geodb_model', 'gpt-4o-mini')

    def _build_history_info(self) -> str:
        """Build history information string from fuse_history and winner_answer."""
        content = ''

        if len(self.fuse_history) > 0:
            content += '\nExtra information from other processors:\n'
            for i, item in enumerate(self.fuse_history, 1):
                content += (
                    f'{i}. {item.get("processor_name", "unknown")}: {item["answer"]}\n'
                )

        if len(self.winner_answer) > 0:
            content += '\nPrevious answers to the query:\n'
            for i, item in enumerate(self.winner_answer, 1):
                content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        return content

    def _generate_geodb_query(self, query: str) -> str:
        """LLM Call 1: Generate a concise GeoDB search query."""
        history_info = self._build_history_info()

        prompt = GEODB_QUERY_PROMPT.format(
            query=query,
            history_info=history_info if history_info else 'None',
        )

        response = completion(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=100,
            temperature=0.3,
        )

        geodb_query = response.choices[0].message.content.strip()
        geodb_query = geodb_query.strip('"\'')
        return geodb_query[:50]

    async def _call_geodb_mcp_async(self, geodb_query: str) -> str:
        """MCP Call: Use GeoDBMCPAgent to search geographic data."""
        try:
            async with GeoDBMCPAgent(
                rapidapi_key=self.rapidapi_key, model=self.geodb_model
            ) as agent:
                return await agent.run(geodb_query)
        except Exception as e:
            return f'Error calling GeoDB MCP: {str(e)}'

    def _call_geodb_mcp(self, geodb_query: str) -> str:
        """Sync wrapper for async MCP call."""
        try:
            return asyncio.run(self._call_geodb_mcp_async(geodb_query))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._call_geodb_mcp_async(geodb_query))
            finally:
                loop.close()

    def _generate_additional_question(self, query: str, response: str) -> str:
        """LLM Call 2: Generate additional question based on GeoDB response."""
        history_info = self._build_history_info()

        prompt = GEODB_ADDITIONAL_PROMPT.format(
            query=query,
            response=response[:2000],
            history_info=history_info if history_info else 'None',
        )

        llm_response = completion(
            model=self.model,
            messages=[{'role': 'user', 'content': prompt}],
            max_tokens=200,
            temperature=0.3,
        )

        return llm_response.choices[0].message.content.strip()

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        is_fuse: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        """
        GeoDB Processor workflow:
        1. LLM Call 1: query + history → GeoDB search query
        2. MCP Call: search GeoDB Cities
        3. LLM Call 2: response + history → additional question
        """
        clean_query = query

        # Step 1: LLM Call - Generate GeoDB search query
        geodb_query = self._generate_geodb_query(query)
        print(f'[GeoDBProcessor] Generated search query: {geodb_query}')

        # Step 2: MCP Call - Search GeoDB
        geodb_response = self._call_geodb_mcp(geodb_query)

        if geodb_response.startswith('Error calling GeoDB MCP:'):
            print(f'[GeoDBProcessor] {geodb_response}')
            return None

        # Step 3: LLM Call - Generate additional question
        additional_question = self._generate_additional_question(
            clean_query, geodb_response
        )

        executor_output = {
            'response': geodb_response,
            'additional_question': additional_question,
        }

        if is_fuse:
            self.add_fuse_history(clean_query, executor_output['response'])

        self.add_all_context_history(
            clean_query,
            executor_output['response'],
            executor_output['additional_question'],
        )

        # Extract scores from executor output
        scorer_output = self._extract_scores_from_executor_output(executor_output)

        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )

        return chunk
