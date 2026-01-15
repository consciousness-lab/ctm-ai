import asyncio
import json
import os
from typing import Any, Dict, List, Optional

from litellm import completion
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ..chunks import Chunk
from ..scorers import BaseScorer
from .processor_base import BaseProcessor

# Prompt for generating Exercise search query
EXERCISE_QUERY_PROMPT = """Based on the following information, generate an exercise search query (in English, max 50 characters) that would help find relevant exercises from ExerciseDB.

Original Query: {query}
Additional Information: {history_info}

Generate ONLY the search query text, nothing else. Your query should focus on body parts, muscle groups, equipment, or exercise names.
Examples: "chest exercises", "biceps dumbbell", "back bodyweight", "leg press", "abs workout"
"""

# Prompt for generating additional question
EXERCISE_ADDITIONAL_PROMPT = """You should utilize the information in the context history and the current response from ExerciseDB API to generate an additional question about the query. Your additional question should be potentially answerable by other modality models and about specific information that you are not sure about. Your additional question should be just about what kind of information you need to get from other modality models or other tools, nothing else about the task or original query should be included. The question needs to be short and clean.

Original Query: {query}
Exercise Response: {response}
Additional Information: {history_info}
"""


class ExerciseMCPAgent:
    """ExerciseDB MCP Agent for finding exercises by body part, muscle, or equipment"""

    def __init__(self, rapidapi_key: str, model: str = "gpt-4o-mini"):
        self.model = model
        self.rapidapi_key = rapidapi_key
        self.session: Optional[ClientSession] = None
        self.tools: List[Dict] = []
        self.system_prompt = """You are a fitness exercise assistant using ExerciseDB API.
Use the tools to find exercises categorized by body part, target muscle group, and equipment.
The API provides detailed exercise information including names, instructions, and form animations.
Summarize the key exercise information concisely in English, including:
- Exercise name
- Target muscle group
- Required equipment
- Brief description of the movement
When listing multiple exercises, organize them clearly by category."""

    async def __aenter__(self):
        server_params = StdioServerParameters(
            command="npx",
            args=[
                "mcp-remote",
                "https://mcp.rapidapi.com",
                "--header",
                "x-api-host: exercisedb.p.rapidapi.com",
                "--header",
                f"x-api-key: {self.rapidapi_key}",
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
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": getattr(
                        tool, "inputSchema", {"type": "object", "properties": {}}
                    ),
                },
            }
            for tool in self.tools
        ]

    async def _call_tool(self, name: str, args: Dict) -> str:
        result = await self.session.call_tool(name, args)
        if result.content:
            return "\n".join(
                item.text if hasattr(item, "text") else str(item)
                for item in result.content
            )
        return ""

    async def run(self, prompt: str) -> str:
        from litellm import acompletion

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]
        tools = self._get_tools_for_llm()

        response = await acompletion(
            model=self.model, messages=messages, tools=tools, tool_choice="auto"
        )

        assistant_message = response.choices[0].message

        while assistant_message.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
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
                    {"role": "tool", "tool_call_id": tc.id, "content": result[:10000]}
                )

            response = await acompletion(
                model=self.model, messages=messages, tools=tools, tool_choice="auto"
            )
            assistant_message = response.choices[0].message

        return assistant_message.content or ""


@BaseProcessor.register_processor("exercise_processor")
class ExerciseProcessor(BaseProcessor):
    REQUIRED_KEYS = ["RAPIDAPI_KEY"]

    def __init__(
        self, name: str, group_name: Optional[str] = None, *args: Any, **kwargs: Any
    ) -> None:
        super().__init__(name, group_name, *args, **kwargs)
        self.rapidapi_key = os.environ.get("RAPIDAPI_KEY", "")
        self.exercise_model = kwargs.get("exercise_model", "gpt-4o-mini")

    def _build_history_info(self) -> str:
        """Build history information string from fuse_history and winner_answer."""
        content = ""

        if len(self.fuse_history) > 0:
            content += "\nExtra information from other processors:\n"
            for i, item in enumerate(self.fuse_history, 1):
                content += (
                    f'{i}. {item.get("processor_name", "unknown")}: {item["answer"]}\n'
                )

        if len(self.winner_answer) > 0:
            content += "\nPrevious answers to the query:\n"
            for i, item in enumerate(self.winner_answer, 1):
                content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        return content

    def _generate_exercise_query(self, query: str) -> str:
        """LLM Call 1: Generate a concise exercise search query."""
        history_info = self._build_history_info()

        prompt = EXERCISE_QUERY_PROMPT.format(
            query=query,
            history_info=history_info if history_info else "None",
        )

        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
        )

        exercise_query = response.choices[0].message.content.strip()
        exercise_query = exercise_query.strip("\"'")
        return exercise_query[:50]

    async def _call_exercise_mcp_async(self, exercise_query: str) -> str:
        """MCP Call: Use ExerciseMCPAgent to search for exercises."""
        try:
            async with ExerciseMCPAgent(
                rapidapi_key=self.rapidapi_key, model=self.exercise_model
            ) as agent:
                return await agent.run(exercise_query)
        except Exception as e:
            return f"Error calling Exercise MCP: {str(e)}"

    def _call_exercise_mcp(self, exercise_query: str) -> str:
        """Sync wrapper for async MCP call."""
        try:
            return asyncio.run(self._call_exercise_mcp_async(exercise_query))
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(
                    self._call_exercise_mcp_async(exercise_query)
                )
            finally:
                loop.close()

    def _generate_additional_question(self, query: str, response: str) -> str:
        """LLM Call 2: Generate additional question based on Exercise response."""
        history_info = self._build_history_info()

        prompt = EXERCISE_ADDITIONAL_PROMPT.format(
            query=query,
            response=response[:2000],
            history_info=history_info if history_info else "None",
        )

        llm_response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
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
        Exercise Processor workflow:
        1. LLM Call 1: query + history → Exercise search query
        2. MCP Call: search for exercises
        3. LLM Call 2: response + history → additional question
        """
        clean_query = query

        # Step 1: LLM Call - Generate exercise search query
        exercise_query = self._generate_exercise_query(query)
        print(f"[ExerciseProcessor] Generated search query: {exercise_query}")

        # Step 2: MCP Call - Get exercise data
        exercise_response = self._call_exercise_mcp(exercise_query)

        if exercise_response.startswith("Error calling Exercise MCP:"):
            print(f"[ExerciseProcessor] {exercise_response}")
            return None

        # Step 3: LLM Call - Generate additional question
        additional_question = self._generate_additional_question(
            clean_query, exercise_response
        )

        executor_output = {
            "response": exercise_response,
            "additional_question": additional_question,
        }

        if is_fuse:
            self.add_fuse_history(clean_query, executor_output["response"])

        self.add_all_context_history(
            clean_query,
            executor_output["response"],
            executor_output["additional_question"],
        )

        scorer = BaseScorer(*args, **kwargs)
        scorer_output = scorer.ask(query=clean_query, messages=executor_output)

        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )

        return chunk
