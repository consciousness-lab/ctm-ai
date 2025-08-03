from typing import Dict, List, Any


class BFCLManager:
    def __init__(self, inference_data: Dict[str, Any]):

        self.inference_data = inference_data
        self.function_names = []
        self.funcs_to_all_info = {}
        self.funcs_to_description = {}
        self.messages = inference_data["message"]
        self._parse_tools()

    def _parse_tools(self):
        if "tools" not in self.inference_data:
            return

        for tool in self.inference_data["tools"]:
            if tool["type"] == "function":
                func_info = tool["function"]
                func_name = func_info["name"]

                self.function_names.append(func_name)

                self.funcs_to_all_info[func_name] = func_info

                self.funcs_to_description[func_name] = func_info.get("description", "")

    def get_function_names(self) -> List[str]:
        return self.function_names

    def get_function_info(self, func_name: str) -> Dict[str, Any]:
        return self.funcs_to_all_info.get(func_name, {})

    def get_function_description(self, func_name: str) -> str:
        return self.funcs_to_description.get(func_name, "")

    def get_all_function_info(self) -> Dict[str, Dict[str, Any]]:
        return self.funcs_to_all_info

    def get_all_function_descriptions(self) -> Dict[str, str]:
        return self.funcs_to_description

    def get_messages(self) -> List[Dict[str, Any]]:
        return self.inference_data.get("message", [])

    def get_tools(self) -> List[Dict[str, Any]]:
        return self.inference_data.get("tools", [])

    def get_message_content(self, index: int = 0) -> str:
        messages = self.get_messages()
        if 0 <= index < len(messages):
            return messages[index].get("content", "")
        return ""

    def get_first_message_content(self) -> str:
        return self.get_message_content(0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        messages = self.get_messages()
        if 0 <= index < len(messages):
            return messages[index]
        raise IndexError("Message index out of range")

    def __len__(self) -> int:
        return len(self.get_messages())
