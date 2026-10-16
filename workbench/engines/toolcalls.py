"""Turn the text tool-call formats the bundled models emit into OpenAI `tool_calls`.

llama-cpp-python renders tools into each model's own chat template, but returns
the model's tool call as plain text. Clients (Pi, n8n agents, editors) expect
`message.tool_calls`. Formats handled:

  Qwen3.5 / Qwen3.6 / Granite 4.2:
      <tool_call><function=NAME><parameter=KEY>VALUE</parameter>...</function></tool_call>
  JSON in <tool_call> (Hermes style):
      <tool_call>{"name": NAME, "arguments": {...}}</tool_call>
  gpt-oss (Harmony):
      ...to=functions.NAME<|channel|>commentary json<|message|>{...}<|call|>
  Gemma 4:
      <|tool_call>call:NAME{key:<|"|>text<|"|>,n:3}<tool_call|>
"""

import json, re, uuid

_XML_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)
_XML_FUNC = re.compile(r"<function=([^>\s]+)>(.*?)(?:</function>|$)", re.S)
_XML_PARAM = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.S)
_HARMONY = re.compile(
    r"to=functions\.([\w.\-]+)\s*(?:<\|channel\|>\s*commentary)?[^<]*<\|message\|>(.*?)(?:<\|call\|>|<\|end\|>|$)",
    re.S,
)
_GEMMA = re.compile(r"<\|tool_call>\s*call:([\w.\-]+)\s*(\{.*?\})\s*<tool_call\|>", re.S)
_GEMMA_STR = '<|"|>'
_THINK = re.compile(r"<think>.*?</think>", re.S)


def _value(text, schema_type=None):
    """Parameter values arrive as raw text; decode JSON-looking values."""
    t = text.strip("\n")
    if schema_type == "string":
        return t
    try:
        return json.loads(t)
    except ValueError:
        return t


def _param_types(tools, name):
    for tool in tools or []:
        fn = tool.get("function", {})
        if fn.get("name") == name:
            return {k: v.get("type") for k, v in fn.get("parameters", {}).get("properties", {}).items()}
    return {}


def _gemma_args(body):
    """Parse Gemma's argument syntax: bare keys, <|"|>-quoted strings, JSON-ish otherwise."""
    out, i, n = [], 0, len(body)
    expect_key = False
    while i < n:
        if body.startswith(_GEMMA_STR, i):
            j = body.find(_GEMMA_STR, i + len(_GEMMA_STR))
            j = n if j < 0 else j
            out.append(json.dumps(body[i + len(_GEMMA_STR):j]))
            i = j + len(_GEMMA_STR)
            continue
        c = body[i]
        if c in "{,":
            out.append(c)
            i += 1
            expect_key = True
            continue
        if expect_key and not c.isspace():
            m = re.match(r"([A-Za-z_][\w\-]*)\s*:", body[i:])
            if m:
                out.append(json.dumps(m.group(1)) + ":")
                i += m.end()
            expect_key = False
            continue
        out.append(c)
        i += 1
    return json.loads("".join(out))


def _call(name, args):
    return {
        "id": "call_" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }


def extract(text, tools=None):
    """Return (remaining_text, tool_calls). tool_calls is [] when none found."""
    if not text:
        return text, []
    calls = []
    visible = _THINK.sub("", text)

    if "<tool_call>" in visible:
        for block in _XML_CALL.findall(visible):
            fm = _XML_FUNC.search(block)
            if fm:
                name = fm.group(1)
                types = _param_types(tools, name)
                args = {k: _value(v, types.get(k)) for k, v in _XML_PARAM.findall(fm.group(2))}
                calls.append(_call(name, args))
                continue
            try:
                obj = json.loads(block)
                calls.append(_call(obj["name"], obj.get("arguments", obj.get("parameters", {}))))
            except (ValueError, KeyError, TypeError):
                pass
        if calls:
            rest = visible[: visible.find("<tool_call>")].strip()
            return rest or None, calls

    if "to=functions." in text:
        for name, payload in _HARMONY.findall(text):
            try:
                args = json.loads(payload.strip() or "{}")
            except ValueError:
                args = {"input": payload.strip()}
            calls.append(_call(name, args))
        if calls:
            return None, calls

    if "<|tool_call>" in visible:
        for name, body in _GEMMA.findall(visible):
            try:
                calls.append(_call(name, _gemma_args(body)))
            except ValueError:
                pass
        if calls:
            rest = visible[: visible.find("<|tool_call>")].strip()
            return rest or None, calls

    return text, []


# Text that means "a tool call has started", used to hold back streamed tokens.
MARKERS = ("<tool_call>", "to=functions.", "<|tool_call>")


def might_start_call(buffer):
    """True if the buffer ends with a prefix of any marker (so keep buffering)."""
    tail = buffer[-16:]
    return any(m.startswith(tail[i:]) for m in MARKERS for i in range(len(tail)) if tail[i:])


# ---------- llama-cpp-python integration ----------

def _fix_response(resp, tools):
    for choice in resp.get("choices", []):
        msg = choice.get("message") or {}
        msg.setdefault("refusal", None)  # 0.4.2's response model requires the key
        if msg.get("tool_calls") or not tools:
            continue
        rest, calls = extract(msg.get("content"), tools)
        if calls:
            msg["content"] = rest
            msg["tool_calls"] = calls
            choice["finish_reason"] = "tool_calls"
    return resp


def _fix_stream(chunks, tools):
    """Stream text normally; once a tool-call marker shows up, buffer and emit tool_calls."""
    buf, sent, base, holding = "", 0, None, False
    for chunk in chunks:
        base = chunk
        choice = (chunk.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("tool_calls"):
            yield chunk
            continue
        piece = delta.get("content") or ""
        finish = choice.get("finish_reason")
        buf += piece
        if not holding and any(m in buf for m in MARKERS):
            holding = True
        if not holding and piece:
            safe = len(buf)
            if might_start_call(buf):
                safe = max(sent, len(buf) - 16)
            if safe > sent:
                out = dict(chunk, choices=[dict(choice, delta={"content": buf[sent:safe]}, finish_reason=None)])
                sent = safe
                yield out
        if piece == "" and not finish and not holding:
            yield chunk  # role / empty deltas
        if finish:
            break
    if base is None:
        return
    rest, calls = extract(buf, tools) if holding or might_start_call(buf) else (buf, [])
    choice0 = (base.get("choices") or [{}])[0]
    if calls:
        pre = (rest or "")[max(0, sent):] if rest and len(rest) > sent else ""
        if pre:
            yield dict(base, choices=[dict(choice0, delta={"content": pre}, finish_reason=None)])
        for i, c in enumerate(calls):
            yield dict(base, choices=[dict(choice0, finish_reason=None, delta={
                "role": "assistant" if i == 0 else None,
                "tool_calls": [dict(c, index=i)]})])
        yield dict(base, choices=[dict(choice0, delta={}, finish_reason="tool_calls")])
    else:
        if len(buf) > sent:
            yield dict(base, choices=[dict(choice0, delta={"content": buf[sent:]}, finish_reason=None)])
        yield dict(base, choices=[dict(choice0, delta={}, finish_reason=choice0.get("finish_reason") or "stop")])


def fix_request_types():
    """llama-cpp-python 0.4.2 types an assistant message's tool_calls as one object
    instead of a list, so its server rejects any request that replays tool-call
    history (every agent's second turn). Rebuild the request model with a list."""
    from typing import List, Literal, Optional, Union
    from typing_extensions import NotRequired, TypedDict
    from llama_cpp import llama_types as t
    from llama_cpp.server import types as st

    class AssistantMessage(TypedDict):
        role: Literal["assistant"]
        name: NotRequired[Optional[str]]
        content: NotRequired[Optional[Union[str, list]]]
        refusal: NotRequired[Optional[str]]
        tool_calls: NotRequired[Optional[List[Union[t.ChatCompletionMessageToolCall,
                                                     t.ChatCompletionMessageCustomToolCall]]]]
        function_call: NotRequired[t.ChatCompletionRequestAssistantMessageFunctionCall]

    message = Union[
        t.ChatCompletionRequestDeveloperMessage, t.ChatCompletionRequestSystemMessage,
        t.ChatCompletionRequestUserMessage, AssistantMessage,
        t.ChatCompletionRequestToolMessage, t.ChatCompletionRequestFunctionMessage,
    ]
    model = st.CreateChatCompletionRequest
    model.model_fields["messages"].annotation = List[message]
    model.model_rebuild(force=True)


def relax_response_check(app):
    """0.4.2's response model also types tool_calls as one object, so a reply with
    tool calls fails the server's own output check. The kit already returns
    OpenAI-shaped responses, so skip that check on the chat route.

    FastAPI builds a route's handler when the route is created, and FastAPI 0.143
    builds a separate copy for each app that includes the router (from the
    router's own route), so change the router's route and rebuild every handler."""
    from fastapi.routing import APIRoute, request_response
    from llama_cpp.server.app import router

    def clear(route):
        route.response_model = None
        route.response_field = None
        route.secure_cloned_response_field = None
        route.app = request_response(route.get_route_handler())

    for route in router.routes:
        if isinstance(route, APIRoute) and route.path == "/v1/chat/completions":
            clear(route)
    for route in app.routes:  # FastAPI before 0.143 copies routes into the app
        if isinstance(route, APIRoute) and route.path == "/v1/chat/completions":
            clear(route)


def install():
    """Patch Llama.create_chat_completion so every caller gets real tool_calls."""
    import llama_cpp

    original = llama_cpp.Llama.create_chat_completion
    if getattr(original, "_llmkit_toolcalls", False):
        return

    def create_chat_completion(self, *args, **kwargs):
        tools = kwargs.get("tools")
        result = original(self, *args, **kwargs)
        if kwargs.get("stream"):
            return _fix_stream(result, tools) if tools else result
        return _fix_response(result, tools)

    create_chat_completion._llmkit_toolcalls = True
    llama_cpp.Llama.create_chat_completion = create_chat_completion
