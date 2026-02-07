from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types
from litellm import completion

from ..chunks import Chunk
from .processor_base import BaseProcessor
from .utils import parse_json_response_with_scores


@BaseProcessor.register_processor("search_processor")
class SearchProcessor(BaseProcessor):
    REQUIRED_KEYS = ["GEMINI_API_KEY"]

    def _build_executor_content(
        self,
        query: str,
        text: Optional[str] = None,
        video_frames_path: Optional[List[str]] = None,
        is_fuse: bool = False,
        **kwargs: Any,
    ) -> str:
        content = f"Query: {query}\n"

        if not is_fuse:
            if len(self.fuse_history) > 0:
                content += "\nThere are extra information from other processors:\n"
                for i, item in enumerate(self.fuse_history, 1):
                    content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

            if len(self.winner_answer) > 0:
                content += "\nThere are some previous answers to the same query, think further based on this answer:\n"
                for i, item in enumerate(self.winner_answer, 1):
                    content += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        return content

    def build_executor_messages(
        self,
        query: str,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        """SearchProcessor doesn't use this method as it has custom search flow."""
        # This method is required by BaseProcessor but not used in custom ask()
        return None

    def ask(
        self,
        query: str,
        text: Optional[str] = None,
        is_fuse: bool = False,
        *args: Any,
        **kwargs: Any,
    ) -> Chunk:
        """Custom ask method that integrates Google Search with self-evaluation scoring."""
        clean_query = query

        # Build context-aware query for search
        query_with_context = self._build_executor_content(
            query=query,
            text=text,
            is_fuse=is_fuse,
        )

        # Step 1: Use Google Search to get initial response
        client = genai.Client()
        grounding_tool = types.Tool(google_search=types.GoogleSearch())

        system_instruction = (
            f"You are an expert search agent. Based on all the information provided, "
            f"search for and provide a comprehensive answer to the query: {query}"
        )

        config = types.GenerateContentConfig(
            tools=[grounding_tool], system_instruction=system_instruction
        )

        search_response = client.models.generate_content(
            model="gemini-2.0-flash-lite",
            contents=query_with_context,
            config=config,
        )

        search_result = search_response.text

        # Step 2: Generate structured output with answer, additional_question, and scores
        # Build the prompt that includes search result and asks for structured JSON output
        structured_prompt = self._build_structured_prompt(clean_query, search_result)

        # Use the standard self-evaluation scoring format
        executor_output = self._ask_for_structured_output(
            structured_prompt, *args, **kwargs
        )

        # Ensure we have the response from search result
        if not executor_output.get("response"):
            executor_output["response"] = search_result

        # Add to history
        self.add_all_context_history(
            clean_query,
            executor_output["response"],
            executor_output.get("additional_question", ""),
        )

        # Extract scores using the base processor method
        scorer_output = self._extract_scores_from_executor_output(executor_output)
        additional_question = executor_output.get("additional_question", "")

        # Create chunk
        chunk = self.merge_outputs_into_chunk(
            name=self.name,
            scorer_output=scorer_output,
            executor_output=executor_output,
            additional_question=additional_question,
        )
        return chunk

    def _build_structured_prompt(self, query: str, search_result: str) -> str:
        """Build prompt that asks for structured output with self-evaluation scores."""
        context_info = ""
        if len(self.fuse_history) > 0:
            context_info += "\nExtra information from other processors:\n"
            for i, item in enumerate(self.fuse_history, 1):
                context_info += f'{i}. {item["answer"]}\n'

        if len(self.winner_answer) > 0:
            context_info += (
                "\nPrevious answers to consider (may not be fully correct):\n"
            )
            for i, item in enumerate(self.winner_answer, 1):
                context_info += f'{i}. {item["processor_name"]}: {item["answer"]}\n'

        from .utils import JSON_FORMAT_SCORE

        prompt = f"""Query: {query}

Search Result:
{search_result}
{context_info}

Based on the search result above, please:
1. Provide a comprehensive response to the query
2. Generate an additional question if you need more information from other modality models (e.g., "what is the facial expression?", "what is shown in the image?"). If no additional information is needed, leave it empty.
3. Self-evaluate your response with relevance, confidence, and surprise scores.

{JSON_FORMAT_SCORE}"""

        return prompt

    def _ask_for_structured_output(
        self, prompt: str, *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        """Get structured output with self-evaluation scores using litellm."""
        response = completion(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=self.max_tokens,
            n=self.return_num,
            temperature=self.temperature,
            *args,
            **kwargs,
        )

        content = response.choices[0].message.content

        # Parse the JSON response with scores
        parsed = parse_json_response_with_scores(
            content, default_additional_question=""
        )

        return parsed
