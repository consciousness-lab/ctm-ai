from typing import Dict, List, Any
from utils import func_doc_language_specific_pre_processing
import copy
import re
import json

GORILLA_TO_OPENAPI = {
    "integer": "integer",
    "float": "number",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
    "object": "object",
    "dict": "object",
    "list": "array",
    "tuple": "array",
}


def _cast_to_openai_type(
    properties: Dict[str, Any], mapping: Dict[str, str]
) -> Dict[str, Any]:
    for key, value in properties.items():
        if "type" not in value:
            properties[key]["type"] = "string"
        else:
            var_type = value["type"]

            if mapping == GORILLA_TO_OPENAPI and var_type == "float":
                properties[key]["format"] = "float"
                properties[key]["description"] += " This is a float type value."

            if var_type in mapping:
                properties[key]["type"] = mapping[var_type]
            else:
                properties[key]["type"] = "string"

        if properties[key]["type"] == "array" or properties[key]["type"] == "object":
            if "properties" in properties[key]:
                properties[key]["properties"] = _cast_to_openai_type(
                    properties[key]["properties"], mapping
                )
            elif "items" in properties[key]:
                properties[key]["items"]["type"] = mapping[
                    properties[key]["items"]["type"]
                ]

                if (
                    properties[key]["items"]["type"] == "array"
                    and "items" in properties[key]["items"]
                ):
                    properties[key]["items"]["items"]["type"] = mapping[
                        properties[key]["items"]["items"]["type"]
                    ]

                elif (
                    properties[key]["items"]["type"] == "object"
                    and "properties" in properties[key]["items"]
                ):
                    properties[key]["items"]["properties"] = _cast_to_openai_type(
                        properties[key]["items"]["properties"], mapping
                    )

    return properties


class OpenAIHandler:
    def __init__(self, model_name: str, temperature: float = 0.0):
        self.model_name = model_name
        self.temperature = temperature

    def _pre_query_processing_FC(
        self, inference_data: Dict[str, Any], test_entry: Dict[str, Any]
    ) -> Dict[str, Any]:
        inference_data["message"] = []
        return inference_data

    def _convert_to_openai_tool(
        self,
        functions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        functions = copy.deepcopy(functions)
        oai_tool = []

        for item in functions:
            if "." in item["name"]:
                item["name"] = re.sub(r"\.", "_", item["name"])

            item["parameters"]["type"] = "object"

            if "properties" in item["parameters"]:
                item["parameters"]["properties"] = _cast_to_openai_type(
                    item["parameters"]["properties"], GORILLA_TO_OPENAPI
                )

            if "response" in item:
                item[
                    "description"
                ] += f" The response field has the following schema: {json.dumps(item['response'])}"
                del item["response"]

            oai_tool.append({"type": "function", "function": item})

        return oai_tool

    def _compile_tools(self, inference_data: dict, test_entry: dict) -> dict:
        functions: list = test_entry["function"]
        test_category: str = test_entry["id"].rsplit("_", 1)[0]

        functions = func_doc_language_specific_pre_processing(functions, test_category)
        tools = self._convert_to_openai_tool(functions)

        inference_data["tools"] = tools

        return inference_data

    def add_first_turn_message_FC(
        self, inference_data: dict, first_turn_message: list[dict]
    ) -> dict:
        inference_data["message"].extend(first_turn_message)
        return inference_data

    def process_test_entry(self, test_entry: Dict[str, Any]) -> Dict[str, Any]:
        inference_data = {}

        inference_data = self._pre_query_processing_FC(inference_data, test_entry)

        inference_data = self._compile_tools(inference_data, test_entry)

        inference_data = self.add_first_turn_message_FC(
            inference_data, test_entry["question"][0]
        )

        return inference_data
