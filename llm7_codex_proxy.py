import json
import os
import time
import traceback
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - exercised at runtime when dependency is absent.
    OpenAI = None


HOST = os.environ.get("PROXY_HOST", "127.0.0.1")
PORT = int(os.environ.get("PROXY_PORT", "5011"))
LLM7_BASE_URL = os.environ.get("LLM7_BASE_URL", "https://api.llm7.io/v1").rstrip("/")
DEFAULT_MODEL = "gpt-5.5"
LLM7_MODEL = os.environ.get("LLM7_MODEL", DEFAULT_MODEL)
LLM7_API_KEY = os.environ.get("LLM7_API_KEY", "unused")
AGENTIC_TOOL_PROMPT = os.environ.get("AGENTIC_TOOL_PROMPT", "1").lower() not in ("0", "false", "no")
LLM7_SAFE_MODE = os.environ.get("LLM7_SAFE_MODE", "1").lower() not in ("0", "false", "no")
LLM7_EXTRA_BODY_PASSTHROUGH = os.environ.get("LLM7_EXTRA_BODY_PASSTHROUGH", "0").lower() in ("1", "true", "yes")
OPENAI_CLIENT = None

CHAT_PASSTHROUGH_KEYS = (
    "audio",
    "frequency_penalty",
    "function_call",
    "functions",
    "logit_bias",
    "logprobs",
    "max_completion_tokens",
    "max_tokens",
    "metadata",
    "modalities",
    "n",
    "parallel_tool_calls",
    "prediction",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "store",
    "stream_options",
    "temperature",
    "tool_choice",
    "tools",
    "top_logprobs",
    "top_p",
    "user",
    "verbosity",
)

RESPONSES_TO_CHAT_KEYS = {
    "max_output_tokens": "max_tokens",
}

RESPONSES_DIRECT_CHAT_KEYS = (
    "frequency_penalty",
    "metadata",
    "parallel_tool_calls",
    "presence_penalty",
    "reasoning_effort",
    "response_format",
    "seed",
    "service_tier",
    "stop",
    "temperature",
    "tool_choice",
    "top_p",
    "truncation",
    "user",
)

LLM7_SAFE_CHAT_KEYS = (
    "frequency_penalty",
    "max_completion_tokens",
    "max_tokens",
    "parallel_tool_calls",
    "presence_penalty",
    "stop",
    "temperature",
    "tool_choice",
    "tools",
    "top_p",
    "user",
)

LLM7_SAFE_RESPONSES_CHAT_KEYS = (
    "frequency_penalty",
    "parallel_tool_calls",
    "presence_penalty",
    "stop",
    "temperature",
    "tool_choice",
    "top_p",
    "user",
)

CHAT_RESERVED_KEYS = set(CHAT_PASSTHROUGH_KEYS) | {"model", "messages", "stream"}
RESPONSES_RESERVED_KEYS = (
    set(RESPONSES_DIRECT_CHAT_KEYS)
    | set(RESPONSES_TO_CHAT_KEYS)
    | {"model", "input", "instructions", "tools", "tool_choice", "stream"}
)


def now_unix():
    return int(time.time())


def upstream_model(model):
    if not model:
        return LLM7_MODEL
    return model


def valid_model(model):
    return model in ("default", "fast", "pro", DEFAULT_MODEL) or str(model).startswith("gpt-5")


def openai_client():
    global OPENAI_CLIENT
    if OpenAI is None:
        raise RuntimeError("Missing dependency: install the OpenAI SDK with `pip install openai`.")
    if OPENAI_CLIENT is None:
        OPENAI_CLIENT = OpenAI(api_key=LLM7_API_KEY, base_url=LLM7_BASE_URL, timeout=300.0)
    return OPENAI_CLIENT


def to_plain_data(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(exclude_none=True)
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return json.loads(json.dumps(value, default=lambda obj: getattr(obj, "__dict__", str(obj))))


def json_response(handler, status, data):
    body = json.dumps(data).encode("utf-8")
    handler.send_response(status)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(body)))
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()
    handler.wfile.write(body)


def sse_headers(handler):
    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("Connection", "keep-alive")
    handler.send_header("Access-Control-Allow-Origin", "*")
    handler.end_headers()


def write_sse(handler, data, event=None):
    if event:
        handler.wfile.write(f"event: {event}\n".encode("utf-8"))
    handler.wfile.write(f"data: {json.dumps(data, separators=(',', ':'))}\n\n".encode("utf-8"))
    handler.wfile.flush()


def write_chat_error(handler, model, message):
    write_sse(
        handler,
        {
            "id": f"error-{uuid.uuid4()}",
            "object": "chat.completion.chunk",
            "created": now_unix(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": message},
                    "finish_reason": "error",
                }
            ],
        },
    )


def llm7_stream(payload):
    return openai_client().chat.completions.create(**payload)


def log_upstream_exception(exc):
    print("LLM7 request failed:")
    traceback.print_exception(type(exc), exc, exc.__traceback__)


def content_to_text(value):
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text") or item.get("input_text") or item.get("output_text")
                if text:
                    parts.append(str(text))
        return "\n".join(parts)
    return json.dumps(value, separators=(",", ":"))


def content_to_chat_content(value):
    if value is None or isinstance(value, str):
        return value or ""
    if not isinstance(value, list):
        return content_to_text(value)

    content = []
    text_parts = []
    for item in value:
        if isinstance(item, str):
            text_parts.append(item)
            continue
        if not isinstance(item, dict):
            text_parts.append(content_to_text(item))
            continue

        item_type = item.get("type")
        if item_type in ("input_text", "output_text", "text"):
            text = item.get("text")
            if text:
                content.append({"type": "text", "text": str(text)})
            continue
        if item_type in ("input_image", "image_url"):
            image_url = item.get("image_url") or item.get("url")
            if image_url:
                content.append({"type": "image_url", "image_url": image_url})
            continue
        text = item.get("text") or item.get("input_text") or item.get("output_text")
        if text:
            content.append({"type": "text", "text": str(text)})

    if content:
        if text_parts:
            content.insert(0, {"type": "text", "text": "\n".join(text_parts)})
        return content
    return "\n".join(text_parts)


def tool_name(tool):
    if not isinstance(tool, dict):
        return ""
    if tool.get("type") == "function":
        function = tool.get("function") or {}
        return tool.get("name") or function.get("name") or ""
    return tool.get("name") or tool.get("type") or ""


def tool_description(tool):
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function") or {}
    return tool.get("description") or function.get("description") or ""


def build_agentic_tool_prompt(tools):
    if not tools or not AGENTIC_TOOL_PROMPT:
        return None

    lines = [
        "You are running inside an agentic coding app. The app can access the user's project files and can run tools when you call them.",
        "Do not say you cannot access the project or tools just because you cannot access them directly in natural language.",
        "When file inspection, terminal commands, edits, or other actions are needed, call the available tool instead of refusing.",
        "Available tools:",
    ]
    for tool in tools:
        name = tool_name(tool)
        if not name:
            continue
        description = tool_description(tool)
        if description:
            lines.append(f"- {name}: {description}")
        else:
            lines.append(f"- {name}")
    return "\n".join(lines)


def responses_input_to_messages(request_body):
    messages = []
    tool_prompt = build_agentic_tool_prompt(request_body.get("tools") or [])
    if tool_prompt:
        messages.append({"role": "system", "content": tool_prompt})

    instructions = request_body.get("instructions")
    if isinstance(instructions, str) and instructions.strip():
        messages.append({"role": "system", "content": instructions})

    input_value = request_body.get("input")
    if isinstance(input_value, str):
        messages.append({"role": "user", "content": input_value})
    elif isinstance(input_value, list):
        for item in input_value:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": content_to_text(item)})
                continue

            item_type = item.get("type", "message")
            if item_type == "function_call_output":
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": item.get("call_id", ""),
                        "content": content_to_text(item.get("output")),
                    }
                )
                continue
            if item_type == "function_call":
                call_id = item.get("call_id") or item.get("id") or f"call_{uuid.uuid4().hex}"
                messages.append(
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": call_id,
                                "type": "function",
                                "function": {
                                    "name": item.get("name", "unknown_tool"),
                                    "arguments": item.get("arguments", "{}"),
                                },
                            }
                        ],
                    }
                )
                continue
            if item_type in ("reasoning", "computer_call", "web_search_call"):
                continue

            role = item.get("role", "user")
            if role == "developer":
                role = "system"
            if role not in ("system", "user", "assistant", "tool"):
                role = "user"
            messages.append({"role": role, "content": content_to_chat_content(item.get("content", item))})

    if not messages:
        messages.append({"role": "user", "content": ""})
    return messages


def responses_tools_to_chat_tools(request_body):
    tools = []
    for tool in request_body.get("tools") or []:
        if not isinstance(tool, dict):
            continue
        if tool.get("type") != "function":
            name = tool.get("name") or tool.get("type")
            if not name:
                continue
            tools.append(
                {
                    "type": "function",
                    "function": {
                        "name": name,
                        "description": tool.get("description", f"Codex tool: {name}"),
                        "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                    },
                }
            )
            continue
        if "function" in tool:
            tools.append(tool)
            continue
        name = tool.get("name")
        if not name:
            continue
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": tool.get("parameters", {"type": "object", "properties": {}}),
                },
            }
        )
    return tools


def responses_tool_choice_to_chat_tool_choice(tool_choice):
    if not isinstance(tool_choice, dict):
        return tool_choice
    if tool_choice.get("type") == "function" and "name" in tool_choice:
        return {"type": "function", "function": {"name": tool_choice["name"]}}
    return tool_choice


def extra_body_from(body, reserved_keys):
    extra_body = body.get("extra_body") or {}
    if not isinstance(extra_body, dict):
        extra_body = {}
    for key, value in body.items():
        if key not in reserved_keys and not key.startswith("_"):
            extra_body[key] = value
    return extra_body


def build_chat_payload(body):
    payload = {
        "model": upstream_model(body.get("model", DEFAULT_MODEL)),
        "messages": body.get("messages", []),
        "stream": True,
    }
    passthrough_keys = LLM7_SAFE_CHAT_KEYS if LLM7_SAFE_MODE else CHAT_PASSTHROUGH_KEYS
    for key in passthrough_keys:
        if key in body:
            payload[key] = body[key]
    if "tools" in payload and "tool_choice" not in payload:
        payload["tool_choice"] = "auto"
    extra_body = extra_body_from(body, CHAT_RESERVED_KEYS) if LLM7_EXTRA_BODY_PASSTHROUGH else {}
    if extra_body and not LLM7_SAFE_MODE:
        payload["extra_body"] = extra_body
    return payload


def build_responses_chat_payload(body):
    payload = {
        "model": upstream_model(body.get("model", DEFAULT_MODEL)),
        "messages": responses_input_to_messages(body),
        "stream": True,
    }
    tools = responses_tools_to_chat_tools(body)
    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = responses_tool_choice_to_chat_tool_choice(body.get("tool_choice", "auto"))
    for response_key, chat_key in RESPONSES_TO_CHAT_KEYS.items():
        if response_key in body:
            payload[chat_key] = body[response_key]
    passthrough_keys = LLM7_SAFE_RESPONSES_CHAT_KEYS if LLM7_SAFE_MODE else RESPONSES_DIRECT_CHAT_KEYS
    for key in passthrough_keys:
        if key in body and key not in ("tool_choice", "truncation"):
            payload[key] = body[key]
    extra_body = extra_body_from(body, RESPONSES_RESERVED_KEYS) if LLM7_EXTRA_BODY_PASSTHROUGH else {}
    if extra_body and not LLM7_SAFE_MODE:
        payload["extra_body"] = extra_body
    return payload


def response_event(handler, event, data):
    write_sse(handler, data, event=event)


def base_response(response_id, model, status="in_progress", output=None, error=None):
    response = {
        "id": response_id,
        "object": "response",
        "created_at": now_unix(),
        "status": status,
        "model": model,
        "output": output or [],
        "parallel_tool_calls": True,
        "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0},
    }
    if error:
        response["error"] = error
    return response


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("[%s] %s" % (self.log_date_time_string(), fmt % args))

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "authorization, content-type, accept")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.end_headers()

    def do_GET(self):
        if self.path == "/health":
            return json_response(
                self,
                200,
                {
                    "status": "healthy",
                    "service": "llm7-codex-python-proxy",
                    "upstream": LLM7_BASE_URL,
                    "default_model": LLM7_MODEL,
                },
            )

        if self.path == "/v1/models":
            return json_response(
                self,
                200,
                {
                    "object": "list",
                    "data": [
                        {"id": "default", "object": "model", "created": now_unix(), "owned_by": "llm7"},
                        {"id": "fast", "object": "model", "created": now_unix(), "owned_by": "llm7"},
                        {"id": "pro", "object": "model", "created": now_unix(), "owned_by": "llm7"},
                        {"id": DEFAULT_MODEL, "object": "model", "created": now_unix(), "owned_by": "llm7"},
                    ],
                },
            )

        return json_response(self, 404, {"error": {"message": "Not found"}})

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            return json_response(self, 400, {"error": {"message": f"Invalid JSON: {exc}"}})

        if self.path in ("/chat/completions", "/v1/chat/completions"):
            return self.handle_chat_completions(body)

        if self.path == "/v1/responses":
            return self.handle_responses(body)

        return json_response(self, 404, {"error": {"message": "Not found"}})

    def handle_chat_completions(self, body):
        model = body.get("model", DEFAULT_MODEL)
        if not valid_model(model):
            return json_response(
                self,
                404,
                {"error": {"message": "Use model gpt-5.5, default, fast, pro, or gpt-5* alias."}},
            )

        payload = build_chat_payload(body)

        sse_headers(self)
        try:
            for event in llm7_stream(payload):
                write_sse(self, to_plain_data(event))
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            self.close_connection = True
        except Exception as exc:
            log_upstream_exception(exc)
            write_chat_error(self, model, f"LLM7 request failed: {exc}")
            self.close_connection = True

    def handle_responses(self, body):
        model = body.get("model", DEFAULT_MODEL)
        if not valid_model(model):
            return json_response(
                self,
                404,
                {"error": {"message": "Use model gpt-5.5, default, fast, pro, or gpt-5* alias."}},
            )

        payload = build_responses_chat_payload(body)

        response_id = f"resp_{uuid.uuid4().hex}"
        message_id = f"msg_{uuid.uuid4().hex}"
        output_text = []
        output_items = []
        tool_calls = {}
        tool_items_added = set()
        message_item_added = False
        content_part_added = False

        sse_headers(self)
        response_event(
            self,
            "response.created",
            {
                "type": "response.created",
                "response": base_response(response_id, model),
            },
        )

        try:
            for raw_event in llm7_stream(payload):
                event = to_plain_data(raw_event)
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    content = delta.get("content")
                    if content:
                        if not message_item_added:
                            message_item_added = True
                            response_event(
                                self,
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": 0,
                                    "item": {
                                        "type": "message",
                                        "id": message_id,
                                        "status": "in_progress",
                                        "role": "assistant",
                                        "content": [],
                                    },
                                },
                            )
                        if not content_part_added:
                            content_part_added = True
                            response_event(
                                self,
                                "response.content_part.added",
                                {
                                    "type": "response.content_part.added",
                                    "item_id": message_id,
                                    "output_index": 0,
                                    "content_index": 0,
                                    "part": {"type": "output_text", "text": ""},
                                },
                            )
                        output_text.append(content)
                        response_event(
                            self,
                            "response.output_text.delta",
                            {
                                "type": "response.output_text.delta",
                                "item_id": message_id,
                                "output_index": 0,
                                "content_index": 0,
                                "delta": content,
                            },
                        )

                    for call in delta.get("tool_calls") or []:
                        index = int(call.get("index", 0))
                        acc = tool_calls.setdefault(index, {"id": "", "name": "", "arguments": ""})
                        if call.get("id"):
                            acc["id"] = call["id"]
                        function = call.get("function") or {}
                        if function.get("name"):
                            acc["name"] = function["name"]
                        call_id = acc["id"] or f"call_{index}"
                        if index not in tool_items_added and acc["name"]:
                            tool_items_added.add(index)
                            response_event(
                                self,
                                "response.output_item.added",
                                {
                                    "type": "response.output_item.added",
                                    "output_index": index,
                                    "item": {
                                        "type": "function_call",
                                        "id": call_id,
                                        "call_id": call_id,
                                        "name": acc["name"],
                                        "arguments": "",
                                        "status": "in_progress",
                                    },
                                },
                            )
                        arguments_delta = function.get("arguments")
                        if arguments_delta:
                            acc["arguments"] += arguments_delta
                            response_event(
                                self,
                                "response.function_call_arguments.delta",
                                {
                                    "type": "response.function_call_arguments.delta",
                                    "item_id": call_id,
                                    "output_index": index,
                                    "delta": arguments_delta,
                                },
                            )

                    if choice.get("finish_reason") == "tool_calls":
                        for index, acc in tool_calls.items():
                            call_id = acc["id"] or f"call_{uuid.uuid4().hex}"
                            function_item = {
                                "type": "function_call",
                                "id": call_id,
                                "call_id": call_id,
                                "name": acc["name"] or "unknown_tool",
                                "arguments": acc["arguments"],
                                "status": "completed",
                            }
                            response_event(
                                self,
                                "response.function_call_arguments.done",
                                {
                                    "type": "response.function_call_arguments.done",
                                    "item_id": call_id,
                                    "output_index": index,
                                    "arguments": acc["arguments"],
                                },
                            )
                            response_event(
                                self,
                                "response.output_item.done",
                                {
                                    "type": "response.output_item.done",
                                    "output_index": index,
                                    "item": function_item,
                                },
                            )
                            output_items.append(function_item)
                        self.finish_completed(response_id, model, output_items)
                        self.close_connection = True
                        return
            self.finish_response(response_id, model, message_id, "".join(output_text), output_items)
            self.close_connection = True
        except Exception as exc:
            log_upstream_exception(exc)
            if tool_calls:
                for index, acc in tool_calls.items():
                    call_id = acc["id"] or f"call_{uuid.uuid4().hex}"
                    function_item = {
                        "type": "function_call",
                        "id": call_id,
                        "call_id": call_id,
                        "name": acc["name"] or "unknown_tool",
                        "arguments": acc["arguments"],
                        "status": "completed",
                    }
                    response_event(
                        self,
                        "response.function_call_arguments.done",
                        {
                            "type": "response.function_call_arguments.done",
                            "item_id": call_id,
                            "output_index": index,
                            "arguments": acc["arguments"],
                        },
                    )
                    response_event(
                        self,
                        "response.output_item.done",
                        {
                            "type": "response.output_item.done",
                            "output_index": index,
                            "item": function_item,
                        },
                    )
                    output_items.append(function_item)
                self.finish_completed(response_id, model, output_items)
                self.close_connection = True
                return
            if output_text:
                self.finish_response(response_id, model, message_id, "".join(output_text), output_items)
                self.close_connection = True
                return
            response_event(
                self,
                "response.failed",
                {
                    "type": "response.failed",
                    "response": base_response(
                        response_id,
                        model,
                        status="failed",
                        output=output_items,
                        error={"message": f"LLM7 request failed: {exc}"},
                    ),
                },
            )
            self.close_connection = True

    def finish_response(self, response_id, model, message_id, text, output_items):
        if text:
            message_item = {
                "type": "message",
                "id": message_id,
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
            response_event(
                self,
                "response.output_text.done",
                {
                    "type": "response.output_text.done",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "text": text,
                },
            )
            response_event(
                self,
                "response.content_part.done",
                {
                    "type": "response.content_part.done",
                    "item_id": message_id,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": text},
                },
            )
            response_event(
                self,
                "response.output_item.done",
                {
                    "type": "response.output_item.done",
                    "output_index": 0,
                    "item": message_item,
                },
            )
            output_items.append(message_item)
        self.finish_completed(response_id, model, output_items)

    def finish_completed(self, response_id, model, output_items=None):
        response_event(
            self,
            "response.completed",
            {
                "type": "response.completed",
                "response": base_response(response_id, model, status="completed", output=output_items or []),
            },
        )
        self.close_connection = True


def main():
    if OpenAI is None:
        print("Missing dependency: install the OpenAI SDK with `python -m pip install -r requirements.txt`.")
        return

    print("LLM7 Codex Python Proxy")
    print(f"Listening: http://{HOST}:{PORT}")
    print(f"Upstream:  {LLM7_BASE_URL}")
    print(f"Model:     {LLM7_MODEL}")
    print("")
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /v1/models")
    print("  POST /v1/chat/completions")
    print("  POST /v1/responses")
    print("")
    server = ThreadingHTTPServer((HOST, PORT), ProxyHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping proxy.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
