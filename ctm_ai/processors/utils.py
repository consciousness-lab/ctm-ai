import json

JSON_FORMAT = """
You should utilize the information in the context history and modality-specific information to answer the query.
There might have some answers to other queries, you should utilize them to answer the query. You should not generate the same additional questions as the previous ones.
Please respond in JSON format with the following structure:
{
    "response": "Your detailed response to the query",
    "additional_question": "If you are not sure about the answer, you should generate a question that potentially can be answered by other modality models or other tools like search engine."
}

Your additional_question should be potentially answerable by other modality models or other tools like search engine and about specific information that you are not sure about.
Your additional_question should be just about what kind of information you need to get from other modality models or other tools like search engine, nothing else about the task or original query should be included. For example, what is the tone of the audio, what is the facial expression of the person, what is the caption of the image, etc. The question needs to be short and clean."""


def parse_json_response(
    content: str, default_additional_question: str = ""
) -> tuple[str, str]:
    """
    Parse JSON response to extract response and additional_question.

    Args:
        content: The raw response content from LLM
        default_additional_question: Default additional question if parsing fails

    Returns:
        tuple: (parsed_content, additional_question)
    """
    try:
        if "```json" in content and "```" in content:
            start_idx = content.find("```json") + 7
            end_idx = content.rfind("```")
            if start_idx > 6 and end_idx > start_idx:
                json_content = content[start_idx:end_idx].strip()
                parsed_response = json.loads(json_content)
            else:
                parsed_response = json.loads(content)
        else:
            parsed_response = json.loads(content)

        parsed_content = parsed_response.get("response", content)
        additional_question = parsed_response.get(
            "additional_question", default_additional_question
        )
        if additional_question == "":
            additional_question = default_additional_question

        return parsed_content, additional_question

    except (json.JSONDecodeError, TypeError):
        return content, default_additional_question
