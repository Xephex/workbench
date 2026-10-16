"""python -m unittest discover -s tests   (from llm-kit/)"""

import json, pathlib, sys, unittest

from workbench.engines import toolcalls

TOOLS = [{"type": "function", "function": {"name": "write", "parameters": {"type": "object", "properties": {
    "path": {"type": "string"}, "content": {"type": "string"}, "count": {"type": "integer"}}}}}]


def args(call):
    return json.loads(call["function"]["arguments"])


class TestExtract(unittest.TestCase):
    def test_qwen_xml(self):
        # Verbatim output of Qwen3.5-4B for a write request.
        text = "<tool_call>\n<function=write>\n<parameter=path>\nhello.txt\n</parameter>\n<parameter=content>\nPIOK\n</parameter>\n</function>\n</tool_call>"
        rest, calls = toolcalls.extract(text, TOOLS)
        self.assertIsNone(rest)
        self.assertEqual(calls[0]["function"]["name"], "write")
        self.assertEqual(args(calls[0]), {"path": "hello.txt", "content": "PIOK"})

    def test_qwen_string_param_that_looks_numeric_stays_string(self):
        text = "<tool_call>\n<function=write>\n<parameter=content>\n42\n</parameter>\n<parameter=count>\n3\n</parameter>\n</function>\n</tool_call>"
        _, calls = toolcalls.extract(text, TOOLS)
        self.assertEqual(args(calls[0]), {"content": "42", "count": 3})

    def test_qwen_multiline_and_text_before_and_think(self):
        text = "<think>plan</think>Sure.\n<tool_call>\n<function=write>\n<parameter=content>\nline1\nline2\n</parameter>\n</function>\n</tool_call>"
        rest, calls = toolcalls.extract(text, TOOLS)
        self.assertEqual(rest, "Sure.")
        self.assertEqual(args(calls[0])["content"], "line1\nline2")

    def test_two_calls(self):
        one = "<tool_call>\n<function=read>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>"
        _, calls = toolcalls.extract(one + "\n" + one.replace(">\na\n<", ">\nb\n<"), TOOLS)
        self.assertEqual([args(c)["path"] for c in calls], ["a", "b"])

    def test_hermes_json(self):
        _, calls = toolcalls.extract('<tool_call>\n{"name": "write", "arguments": {"path": "x"}}\n</tool_call>')
        self.assertEqual(args(calls[0]), {"path": "x"})

    def test_harmony(self):
        text = '<|channel|>analysis<|message|>need a file<|end|><|start|>assistant to=functions.write<|channel|>commentary json<|message|>{"path":"a.txt","content":"hi"}<|call|>'
        rest, calls = toolcalls.extract(text)
        self.assertIsNone(rest)
        self.assertEqual(calls[0]["function"]["name"], "write")
        self.assertEqual(args(calls[0]), {"path": "a.txt", "content": "hi"})

    def test_gemma(self):
        text = '<|tool_call>call:write{content:<|"|>a, b: c<|"|>,count:3,path:<|"|>x.txt<|"|>}<tool_call|>'
        _, calls = toolcalls.extract(text)
        self.assertEqual(args(calls[0]), {"content": "a, b: c", "count": 3, "path": "x.txt"})

    def test_plain_text_untouched(self):
        self.assertEqual(toolcalls.extract("just an answer", TOOLS), ("just an answer", []))

    def test_stream_marker_prefix(self):
        self.assertTrue(toolcalls.might_start_call("hello <tool_"))
        self.assertTrue(toolcalls.might_start_call("x <|tool"))
        self.assertFalse(toolcalls.might_start_call("hello world"))


def _chunks(pieces):
    yield {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]}
    for p in pieces:
        yield {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]}
    yield {"id": "c", "object": "chat.completion.chunk", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}


class TestStream(unittest.TestCase):
    def collect(self, pieces):
        text, calls, finish = "", [], None
        for ch in toolcalls._fix_stream(_chunks(pieces), TOOLS):
            c = ch["choices"][0]
            text += c["delta"].get("content") or ""
            calls += c["delta"].get("tool_calls") or []
            finish = c.get("finish_reason") or finish
        return text, calls, finish

    def test_plain_text_streams_through(self):
        self.assertEqual(self.collect(["Hel", "lo ", "world"]), ("Hello world", [], "stop"))

    def test_tool_call_split_across_tokens(self):
        src = "<tool_call>\n<function=write>\n<parameter=path>\nhello.txt\n</parameter>\n</function>\n</tool_call>"
        pieces = [src[i:i + 3] for i in range(0, len(src), 3)]
        text, calls, finish = self.collect(pieces)
        self.assertEqual(text, "")
        self.assertEqual(finish, "tool_calls")
        self.assertEqual(args(calls[0]), {"path": "hello.txt"})
        self.assertEqual(calls[0]["index"], 0)

    def test_text_then_tool_call(self):
        src = "OK.\n<tool_call>\n<function=write>\n<parameter=path>\na\n</parameter>\n</function>\n</tool_call>"
        text, calls, finish = self.collect([src[i:i + 4] for i in range(0, len(src), 4)])
        self.assertEqual(text.strip(), "OK.")
        self.assertEqual(len(calls), 1)
        self.assertEqual(finish, "tool_calls")

    def test_lt_sign_in_plain_text(self):
        text, calls, _ = self.collect(["if a ", "< b then", " done"])
        self.assertEqual((text, calls), ("if a < b then done", []))


if __name__ == "__main__":
    unittest.main()
