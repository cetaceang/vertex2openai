import asyncio
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import httpx
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from diagnose_stream import probe_stream
from stream_diagnostics import SSEInspector, StreamDiagnosticsMiddleware


def event(delta=None, finish=None):
    return "data: " + json.dumps({"choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}]},
                                ensure_ascii=False) + "\n\n"


class SSEInspectorTests(unittest.TestCase):
    def test_fragmented_utf8_crlf_usage_tools_and_done(self):
        raw = (": keep-alive\n\n" + event({"content": "你好"}) + event({"reasoning_content": "private"})
               + event({"tool_calls": [{"index": 0, "function": {"name": "test", "arguments": ""}}]})
               + event({"tool_calls": [{"index": 0, "function": {"arguments": "{}"}}]}, "tool_calls")
               + 'data: {"choices":[],"usage":{"total_tokens":3}}\n\ndata: [DONE]\n\n')
        inspector = SSEInspector()
        for byte in raw.replace("\n", "\r\n").encode():
            inspector.feed(bytes([byte]))
        self.assertEqual(inspector.summary(), {
            "sse_events": 6, "text_chars": 2, "reasoning_chars": 7, "tool_calls": 1,
            "finish_reasons": ["tool_calls"], "done": True, "sse_errors": 0,
            "invalid_events": 0, "pending_event": False,
        })

    def test_truncated_and_error_events_are_distinguished(self):
        inspector = SSEInspector()
        inspector.feed(b'data: {"error":{"message":"do not log this"}}\n\ndata: {')
        self.assertEqual(inspector.errors, 1)
        self.assertTrue(inspector.summary()["pending_event"])
        self.assertFalse(inspector.done)

    def test_named_error_event_is_detected_without_logging_its_message(self):
        inspector = SSEInspector()
        inspector.feed(b'event: error\ndata: {"message":"private"}\n\n')
        self.assertEqual(inspector.errors, 1)
        self.assertNotIn("private", str(inspector.summary()))

    def test_native_finish_and_thoughts_do_not_require_done(self):
        inspector = SSEInspector(native=True)
        inspector.feed(b'data: {"candidates":[{"content":{"parts":[{"text":"a","thought":true},{"text":"ok"},{"functionCall":{"name":"test"}}]},"finishReason":"STOP"}]}\n\n')
        self.assertEqual(inspector.text_chars, 2)
        self.assertEqual(inspector.reasoning_chars, 1)
        self.assertEqual(inspector.native_tool_calls, 1)
        self.assertEqual(inspector.finish_reasons, {"STOP"})


class StreamDiagnosticsTests(unittest.IsolatedAsyncioTestCase):
    async def test_logs_report_complete_incomplete_and_error_without_changing_bytes(self):
        bodies = [
            (event({"content": "private reply"}, "stop") + "data: [DONE]\n\n", "complete"),
            (event({"reasoning_content": "private thought"}), "incomplete_sse"),
            ('data: {"error":{"message":"private error"}}\n\ndata: [DONE]\n\n', "upstream_error"),
        ]
        for body, outcome in bodies:
            with self.subTest(outcome=outcome):
                app = FastAPI()
                app.add_middleware(StreamDiagnosticsMiddleware)

                @app.post("/v1/chat/completions")
                async def endpoint():
                    async def stream():
                        yield body
                    return StreamingResponse(stream(), media_type="text/event-stream")

                with patch("stream_diagnostics.emit_diagnostic") as log:
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
                        response = await client.post("/v1/chat/completions")
                self.assertEqual(response.text, body)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(log.call_args.args, ("end",))
                self.assertEqual(log.call_args.kwargs["outcome"], outcome)
                self.assertTrue(log.call_args.kwargs["response_complete"])
                self.assertNotIn("private", str(log.call_args_list))

    async def test_downstream_disconnect_is_recorded_and_upstream_closes(self):
        first_sent = asyncio.Event()
        upstream_closed = asyncio.Event()

        async def stream():
            try:
                yield event({"reasoning_content": "private thought"})
                await asyncio.Event().wait()
            finally:
                upstream_closed.set()

        response = StreamingResponse(stream(), media_type="text/event-stream")

        async def app(scope, receive, send):
            await response(scope, receive, send)

        async def send(message):
            if message["type"] == "http.response.body":
                first_sent.set()

        async def receive():
            await first_sent.wait()
            return {"type": "http.disconnect"}

        with patch("stream_diagnostics.emit_diagnostic") as log:
            await asyncio.wait_for(StreamDiagnosticsMiddleware(app)(
                {"type": "http", "path": "/v1/chat/completions"}, receive, send,
            ), timeout=2)
        self.assertTrue(upstream_closed.is_set())
        self.assertEqual(log.call_args.kwargs["outcome"], "downstream_disconnected")
        self.assertFalse(log.call_args.kwargs["done"])
        self.assertEqual(log.call_args.kwargs["reasoning_chars"], 15)

    async def test_probe_checks_text_tools_and_truncation(self):
        for tools, truncated in ((False, False), (True, False), (False, True)):
            with self.subTest(tools=tools, truncated=truncated):
                def handle(request):
                    sent = json.loads(request.content)
                    self.assertEqual(sent["stream_options"], {"include_usage": True})
                    self.assertEqual(request.headers["authorization"], "Bearer secret")
                    delta = {"tool_calls": [{"index": 0, "function": {"name": "stream_probe", "arguments": "{}"}}]} if tools else {"content": "OK"}
                    body = event(delta, None if truncated else "tool_calls" if tools else "stop")
                    if not truncated:
                        body += "data: [DONE]\n\n"
                    return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=body)

                result = await probe_stream("http://adapter/v1", "test", "secret", tools=tools,
                                            transport=httpx.MockTransport(handle))
                self.assertEqual(result["ok"], not truncated)
                self.assertNotIn("secret", str(result))

    async def test_probe_reports_connection_failures(self):
        def handle(request):
            raise httpx.RemoteProtocolError("connection closed")

        result = await probe_stream("http://adapter/v1", "test", transport=httpx.MockTransport(handle))
        self.assertFalse(result["ok"])
        self.assertEqual(result["connection_error"], "RemoteProtocolError")


if __name__ == "__main__":
    unittest.main()
