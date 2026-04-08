import json
import re
from abc import ABC, abstractmethod
from typing import Any

from rllm.tools.tool_base import ToolCall


class ToolParser(ABC):
    @abstractmethod
    def parse(self, model_response: str) -> list[ToolCall]:
        """Extract tool calls from the model response."""
        raise NotImplementedError("Subclasses must implement this method")

    @abstractmethod
    def get_tool_prompt(self, tools_schema: str) -> str:
        """Get the tool prompt for the model."""
        raise NotImplementedError("Subclasses must implement this method")

    @classmethod
    def get_parser(cls, tokenizer) -> "ToolParser":
        if isinstance(tokenizer.name_or_path, str):
            model_name = tokenizer.name_or_path.lower()
            tokenizer_cls = tokenizer.__class__.__name__.lower()
            print(f"model_name: {model_name}, tokenizer_cls: {tokenizer_cls}")
            if any(x in model_name for x in ("deepseek", "deepscaler", "deepcoder")) and "llama" in tokenizer_cls:
                print(f"Using R1ToolParser for {tokenizer.name_or_path}")
                return R1ToolParser()
            elif "qwen" in model_name or "r2e" in model_name or "deepswe" in model_name or "qwen" in tokenizer_cls:
                print(f"Using QwenToolParser for {tokenizer.name_or_path}")
                return QwenToolParser()
        raise ValueError(f"No tool parser found for {tokenizer.name_or_path}")


class R1ToolParser(ToolParser):
    def __init__(self):
        self.tool_calls_begin = "<｜tool▁calls▁begin｜>"
        self.tool_calls_end = "<｜tool▁calls▁end｜>"
        self.tool_call_begin = "<｜tool▁call▁begin｜>"
        self.tool_call_end = "<｜tool▁call▁end｜>"
        self.tool_sep = "<｜tool▁sep｜>"
        self.tool_output_begin = "<｜tool▁response▁begin｜>"
        self.tool_output_end = "<｜tool_response_end｜>"

    def parse(self, model_response: str) -> list[ToolCall]:
        tool_calls_dicts = self.parse_r1_tool_calls(model_response)
        return [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]

    def parse_r1_tool_calls(self, text: str) -> list[dict]:
        tool_calls = []
        call_idx = 0
        while True:
            call_idx = text.find(self.tool_call_begin, call_idx)
            if call_idx == -1:
                break

            call_start = call_idx + len(self.tool_call_begin)
            call_end = text.find(self.tool_call_end, call_start)
            if call_end == -1:
                break

            call_content = text[call_start:call_end].strip()
            func_prefix = "function" + self.tool_sep
            func_start = call_content.find(func_prefix)
            if func_start == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            func_name_start = func_start + len(func_prefix)
            func_name_end = call_content.find("\n", func_name_start)
            if func_name_end == -1:
                function_name = call_content[func_name_start:].strip()
            else:
                function_name = call_content[func_name_start:func_name_end].strip()

            json_start = call_content.find("```json\n")
            if json_start == -1:
                json_start = call_content.find("```json")
                if json_start == -1:
                    call_idx = call_end + len(self.tool_call_end)
                    continue
                json_start += len("```json")
            else:
                json_start += len("```json\n")

            json_end = call_content.find("```", json_start)
            if json_end == -1:
                call_idx = call_end + len(self.tool_call_end)
                continue

            args_str = call_content[json_start:json_end].strip()
            try:
                args_json = json.loads(args_str)
            except json.JSONDecodeError:
                call_idx = call_end + len(self.tool_call_end)
                continue

            tool_calls.append({"name": function_name, "arguments": args_json})
            call_idx = call_end + len(self.tool_call_end)

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""
# Tools

You may call one or more functions to assist with the user query.
<tools>
{tools_schema}
</tools>

Output format for tool calls:

<｜tool▁calls▁begin｜>
<｜tool▁call▁begin｜>function<｜tool▁sep｜>function_name
```json
{{"param1": "value1", "param2": "value2"}}
```
<｜tool▁call▁end｜>
// Additional tool calls follow the same format
<｜tool▁calls▁end｜>
"""


class QwenToolParser(ToolParser):
    def __init__(self):
        self.tool_call_begin = "<tool_call>"
        self.tool_call_end = "</tool_call>"
        self.tool_output_begin = "<tool_response>"
        self.tool_output_end = "</tool_response>"

    def parse(self, model_response: str) -> list[ToolCall]:
        tool_calls_dicts = self.parse_qwen_tool_calls(model_response)
        return [ToolCall(name=tc["name"], arguments=tc["arguments"]) for tc in tool_calls_dicts]

    def _normalize_arguments(self, arguments: Any) -> Any:
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if stripped.startswith(("{", "[")):
                try:
                    arguments = json.loads(stripped)
                except json.JSONDecodeError:
                    pass

        if isinstance(arguments, dict):
            nested_arguments = arguments.get("arguments")
            if isinstance(nested_arguments, dict):
                flattened_arguments = {key: value for key, value in arguments.items() if key != "arguments"}
                for key, value in nested_arguments.items():
                    flattened_arguments.setdefault(key, value)
                arguments = flattened_arguments

        return arguments

    def _coerce_call_data(self, call_data: Any, raw_content: str) -> dict[str, Any] | None:
        if not isinstance(call_data, dict):
            print(f"Skipping non-dict tool call payload: {raw_content}")
            return None

        name = call_data.get("name")
        arguments = call_data.get("arguments", call_data.get("parameters"))
        if not name or arguments is None:
            print(f"Skipping malformed tool call without required fields: {raw_content}")
            return None

        arguments = self._normalize_arguments(arguments)
        if isinstance(arguments, str):
            stripped = arguments.strip()
            if name == "execute_bash":
                arguments = {"command": stripped}
            elif name == "think":
                arguments = {"thought": stripped}
            elif name == "finish":
                arguments = {"message": stripped}
        return {"name": name, "arguments": arguments}

    def _repair_json_payload(self, payload: str) -> str:
        repaired = payload.strip()
        while repaired.count("{") < repaired.count("}") and repaired.endswith("}"):
            repaired = repaired[:-1].rstrip()
        while repaired.count("[") < repaired.count("]") and repaired.endswith("]"):
            repaired = repaired[:-1].rstrip()

        missing_braces = repaired.count("{") - repaired.count("}")
        if missing_braces > 0:
            repaired += "}" * missing_braces

        missing_brackets = repaired.count("[") - repaired.count("]")
        if missing_brackets > 0:
            repaired += "]" * missing_brackets

        return repaired

    def _extract_quoted_field(self, payload: str, key: str) -> str | None:
        match = re.search(rf'"{re.escape(key)}"\s*:\s*"((?:\\.|[^"\\])*)"', payload, re.S)
        if not match:
            return None
        return bytes(match.group(1), "utf-8").decode("unicode_escape")

    def _extract_multiline_field(self, payload: str, key: str, next_keys: list[str]) -> str | None:
        start_marker = f'"{key}": "'
        start = payload.find(start_marker)
        if start == -1:
            return None
        start += len(start_marker)

        end_candidates = []
        for next_key in next_keys:
            marker = f'", "{next_key}": "'
            idx = payload.find(marker, start)
            if idx != -1:
                end_candidates.append(idx)

        for marker in ['"}}', '"}', '"\n</tool_call>']:
            idx = payload.find(marker, start)
            if idx != -1:
                end_candidates.append(idx)

        end = min(end_candidates) if end_candidates else len(payload)
        return payload[start:end].replace("\\n", "\n").replace('\\"', '"')

    def _extract_loose_scalar(self, payload: str, key: str) -> str | None:
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*(?P<quote>["\'])(?P<body>.*?)(?P=quote)(?:\s*[,}}]|\s*$)',
            payload,
            re.S,
        )
        if not match:
            return None
        return match.group("body").replace("\\n", "\n").replace('\\"', '"').replace("\\'", "'")

    def _extract_raw_block_field(self, payload: str, key: str) -> str | None:
        match = re.search(
            rf'"{re.escape(key)}"\s*:\s*\n(?P<body>.*?)(?:\n\s*\}}(?:\s*$|\n)|\n\s*"</tool_call>|$)',
            payload,
            re.S,
        )
        if not match:
            return None

        body = match.group("body")
        cleaned_lines = []
        for line in body.splitlines():
            stripped = line.strip()
            if not stripped:
                cleaned_lines.append("")
                continue
            if stripped in {"</parameter>", "</timeout>", "</tool_call>", "</function>"}:
                continue
            cleaned_lines.append(line.rstrip())

        cleaned = "\n".join(cleaned_lines).strip()
        return cleaned or None

    def _parse_xmlish_tool_call(self, payload: str) -> dict[str, Any] | None:
        name_match = re.search(r'"name"\s*:\s*"([^"]+)"', payload)
        if not name_match:
            return None

        tool_name = name_match.group(1)
        arguments: dict[str, Any] = {}

        command_match = re.search(r'<command>\s*(.*?)\s*</command>', payload, re.S)
        if command_match:
            arguments["command"] = command_match.group(1).strip()

        timeout_match = re.search(r'<timeout>\s*(.*?)\s*</timeout>', payload, re.S)
        if timeout_match:
            timeout_text = timeout_match.group(1).strip()
            try:
                arguments["timeout"] = float(timeout_text)
            except ValueError:
                arguments["timeout"] = timeout_text

        security_match = re.search(r'<security_risk>\s*(.*?)\s*</security_risk>', payload, re.S)
        if security_match:
            arguments["security_risk"] = security_match.group(1).strip()

        if tool_name == "think":
            thought = self._extract_multiline_field(payload, "thought", [])
            if not thought:
                thought_match = re.search(r'"arguments"\s*:\s*<thought>\s*(.*?)\s*</thought>', payload, re.S)
                if thought_match:
                    thought = thought_match.group(1)
            if not thought:
                loose_thought_match = re.search(
                    r'"arguments"\s*:\s*"thought"\s*:\s*(.*?)(?:\n\s*\}|\s*</tool_call>|$)',
                    payload,
                    re.S,
                )
                if loose_thought_match:
                    thought = loose_thought_match.group(1)
            if thought:
                arguments["thought"] = thought.strip()
            elif re.fullmatch(r'\{\s*"name"\s*:\s*"think"\s*>?\s*', payload.strip()):
                arguments["thought"] = ""

        if tool_name == "str_replace_editor":
            command = self._extract_quoted_field(payload, "command")
            path = self._extract_quoted_field(payload, "path")
            old_str = self._extract_multiline_field(payload, "old_str", ["new_str", "insert_line"])
            new_str = self._extract_multiline_field(payload, "new_str", ["old_str", "insert_line"])
            if command:
                arguments["command"] = command
            if path:
                arguments["path"] = path
            if old_str is not None:
                arguments["old_str"] = old_str
            if new_str is not None:
                arguments["new_str"] = new_str

        if tool_name == "execute_bash" and "command" not in arguments:
            command = self._extract_quoted_field(payload, "command")
            if command is None:
                command = self._extract_quoted_field(payload, "arguments")
            if command is None:
                command = self._extract_multiline_field(payload, "command", ["timeout", "security_risk", "thought"])
            if command is None:
                command = self._extract_multiline_field(payload, "arguments", ["timeout", "security_risk", "thought"])
            if command is None:
                command = self._extract_loose_scalar(payload, "command")
            if command is None:
                raw_command_match = re.search(r'"command"\s*:\s*\n(.*?)\n\s*"', payload, re.S)
                if raw_command_match:
                    command = raw_command_match.group(1)
            if command is None:
                command = self._extract_raw_block_field(payload, "command")
            if command is None:
                command = self._extract_raw_block_field(payload, "arguments")
            if command:
                arguments["command"] = command.strip()

        if tool_name == "finish":
            message = self._extract_quoted_field(payload, "message")
            if message:
                arguments["message"] = message

        if not arguments:
            return None

        return {"name": tool_name, "arguments": arguments}

    def parse_qwen_tool_calls(self, text: str) -> list[dict[str, Any]]:
        tool_calls: list[dict[str, Any]] = []
        if self.tool_call_begin not in text:
            return tool_calls

        while self.tool_call_begin in text:
            start_idx = text.find(self.tool_call_begin)
            start = start_idx + len(self.tool_call_begin)
            next_begin = text.find(self.tool_call_begin, start)
            end = text.find(self.tool_call_end, start)

            if end == -1 and next_begin == -1:
                break

            if next_begin != -1 and (end == -1 or next_begin < end):
                json_content = text[start:next_begin].strip()
                advance_to = next_begin
            else:
                json_content = text[start:end].strip()
                advance_to = end + len(self.tool_call_end)

            parsed_call = None

            try:
                parsed_call = self._coerce_call_data(json.loads(json_content), json_content)
            except json.JSONDecodeError:
                repaired_json = self._repair_json_payload(json_content)
                if repaired_json != json_content:
                    try:
                        parsed_call = self._coerce_call_data(json.loads(repaired_json), repaired_json)
                    except json.JSONDecodeError:
                        parsed_call = None

                if parsed_call is None:
                    parsed_call = self._parse_xmlish_tool_call(json_content)

            if parsed_call is None:
                print(f"Error parsing tool call: {json_content}")
                text = text[advance_to:]
                continue

            tool_calls.append(parsed_call)
            text = text[advance_to:]

        return tool_calls

    def get_tool_prompt(self, tools_schema: str) -> str:
        return f"""

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tools_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>
""".rstrip()
