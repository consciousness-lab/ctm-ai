from typing import Any, Dict, List

from .processor_base import BaseProcessor

PROMPT = """
You are an autonomous intelligent agent tasked with navigating a web browser. 
You will be given web-based tasks to complete using specific actions that you can issue. 
Review the current state of the page and all provided information to decide the single best next action to accomplish your goal.

Here's the information you have:
- The user's objective: This is the task you're trying to complete.
- The current web page's dom html: This is a simplified representation of the webpage, providing key information.
- The previous action: The last action you performed, which helps you track progress.
- The available action space: The possible types of actions you can perform.
- Additional info: Any other useful contextual data, outputs from other processors, and history thinking process, the answer from other processors.

User's objective: {objective}
DOM HTML: {html}
Previous action: {action_history}
Action space: {action_space}
Additional info: {other_info}

Output rules (read carefully):
1. You must issue **exactly ONE valid next action** that is appropriate for the current observation.  
2. You must **reason internally** but only output the final JSON result — do not show your reasoning.  
3. The output must be **exactly one fenced code block (triple backticks)** that contains **exactly one valid JSON object** and nothing else. 
4. When specifying the target element in actions, you MUST use the "bid" attribute value (e.g., bid="1188") from the accessibility tree. The bid attribute is the unique identifier assigned by BrowserGym to each element. **NEVER use class names, IDs, or other attributes**. Always use the bid value (e.g., click("1188") not click("ui-menu-item-wrapper")).
5. Based on the current observation and action history, if you think you performed all the necessary and there is no answer, like searched many times, or reach the end of the page, or no more elements to click, your action should be: "send_msg_to_user(\"No relevant information found\")", or "send_msg_to_user(\"[Summary of the previous]\")".
6. The JSON object must contain the following three fields:

{{
  "response": "Answer to the user's objective based on current information. If you found the answer, state it directly. If not found yet, say what information is missing or not available on the current page. **Do NOT describe what you are doing or will do** (e.g., avoid 'I am examining...' or 'I will move to...'). Just answer based on all the information you are provided. If no answer is available yet, respond with 'Answer to the query: Not yet.'",
  "action": "The single next action to perform as a plain string. **IMPORTANT**: Use the bid attribute value (e.g., \"1188\", \"1189\") from the HTML elements, NOT class names or IDs. Examples: click(\"1188\"), type(\"1197\",\"iphone 16\"), select(\"1200\",\"OptionA\"), send_msg_to_user(\"Done\")",
  "additional_question": "If you are unsure, ask a specific question that another processor (screenshot, html, or axtree) could answer to resolve uncertainty. For example: 'In the axtree, what is the role of element with bid=1188?'"
}}
Example 1(if you can answer the query):
{{
  "response": "Answer to the query: I see 5 'iPhone 16 Pro' listings in the current view: Apple Store, BestBuy, Target, Amazon, Walmart.",
  "action": "click(\"1188\")",
  "additional_question": "In axtree, does element with bid=1188 exist and is it enabled?"
}}
Example 2(if you cannot yet answer the query):
{{
  "response": "Answer to the query: Not yet. There are no reviews about 'underwater photos' on the current page.",
  "action": "click(\"1189\")",
  "additional_question": "In the screenshot, what label appears near element with bid=1189?"
}}
Example 3(if you can answer the query):
{{
  "response": "Answer to the query: Found 8 items that match the filters. Top 3 by price: A($999), B($1049), C($1099).",
  "action": "send_msg_to_user(\"Done\")",
  "additional_question": ""
}}

Note: In the examples above, \"1188\" and \"1189\" are bid attribute values from the HTML. Always use the bid value (e.g., bid=\"1188\") when referring to elements in actions, never use class names like \"ui-menu-item-wrapper\" or IDs like \"ui-id-249\".

Additional guidelines:
-The response field should ONLY contain the answer to the user's objective based on current information. **Do NOT describe your thought process or future actions** (avoid phrases like "I am examining...", "I will move to...", "I have read..."). Just state what you found or what is missing.
-The action field should ONLY contain a single valid action string — no code fences, no explanation.
-The additional_question should ONLY ask for missing perceptual information (e.g., from the HTML, axtree, or screenshot), nothing else about the task or original query should be included.
-If you think the content in the additional info(the answer of other processors) is useful, you should include it in the response.
-Output nothing else before or after the code block.
"""


@BaseProcessor.register_processor("html_processor")
class HtmlProcessor(BaseProcessor):
    REQUIRED_KEYS = ["GEMINI_API_KEY"]

    def build_executor_messages(
        self,
        query,
        action_space,
        action_history,
        html,
        axtree,
        screenshot,
        other_info,
        is_fuse,
        *args: Any,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        message = [
            {
                "role": "system",
                "content": "You are a UI Assistant, your goal is to help the user perform tasks using a web browser.",
            },
            {
                "role": "user",
                "content": PROMPT.format(
                    objective=query,
                    html=html,
                    action_history=action_history,
                    other_info=other_info,
                    action_space=action_space,
                ),
            },
        ]
        return message
