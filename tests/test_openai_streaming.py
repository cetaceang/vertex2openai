import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from api_helpers import StreamingReasoningProcessor, openai_fake_stream_generator
from express_key_manager import ExpressKeyManager
from models import OpenAIRequest
from openai_handler import ExpressClientWrapper, FakeChatCompletion, FakeChatCompletionChunk, OpenAIDirectHandler
from routes.chat_api import router


def chunk(delta=None, finish=None, **extra):
    return {
        "id": "chatcmpl-upstream", "object": "chat.completion.chunk", "created": 123,
        "model": "google/gemini-test", "choices": [{"index": 0, "delta": delta or {}, "finish_reason": finish}],
        **extra,
    }


def tool_delta(arguments="{}"):
    return {"tool_calls": [{
        "index": 0, "id": "call_echo", "type": "function",
        "function": {"name": "echo", "arguments": arguments},
        "extra_content": {"google": {"thought_signature": "c2ln"}},
    }]}


def client_for(events, failure=None, closed=None):
    async def stream():
        try:
            for event in events:
                yield FakeChatCompletionChunk(event)
            if failure:
                raise failure
        finally:
            if closed is not None:
                closed.append(True)
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=AsyncMock(return_value=stream()),
    )))


@patch("config.FAKE_STREAMING_ENABLED", False)
class OpenAIStreamingTests(unittest.IsolatedAsyncioTestCase):
    async def response(self, events, **options):
        request = OpenAIRequest(model="[EXPRESS] gemini-test-openai", messages=[], stream=True)
        client = client_for(events, **options)
        return await OpenAIDirectHandler().handle_streaming_response(client, {}, {}, request)

    async def collect(self, events, **options):
        response = await self.response(events, **options)
        raw = [event async for event in response.body_iterator]
        self.assertEqual(raw.count("data: [DONE]\n\n"), 1)
        self.assertEqual(raw[-1], "data: [DONE]\n\n")
        return [json.loads(event.removeprefix("data: ")) for event in raw[:-1]]

    async def test_terminal_reason_and_identity_are_preserved(self):
        for finish in ("stop", "length", "content_filter", "tool_calls"):
            with self.subTest(finish=finish):
                delta = tool_delta() if finish == "tool_calls" else {"content": "answer"}
                frames = await self.collect([chunk(delta), chunk(finish=finish)])
                self.assertEqual([f["choices"][0]["finish_reason"] for f in frames], [None, finish])
                self.assertEqual({f["id"] for f in frames}, {"chatcmpl-upstream"})
                self.assertEqual({f["created"] for f in frames}, {123})
                self.assertEqual(frames[0]["choices"][0]["delta"]["role"], "assistant")
                if finish == "tool_calls":
                    self.assertEqual(frames[0]["choices"][0]["delta"]["tool_calls"], delta["tool_calls"])

    async def test_missing_terminal_is_completed_for_tool_calls(self):
        frames = await self.collect([chunk(tool_delta())])
        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(frames[-1]["id"], frames[0]["id"])

    async def test_tags_split_across_packets_and_usage_are_preserved(self):
        usage = {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
        frames = await self.collect([
            chunk({"content": "<vertex_think_"}),
            chunk({"content": "tag>think</vertex_think_tag>ans"}),
            chunk({"content": "wer"}, "length"),
            chunk(choices=[], usage=usage),
        ])
        deltas = [c["delta"] for frame in frames for c in frame["choices"]]
        self.assertEqual("".join(d.get("content") or "" for d in deltas), "answer")
        self.assertEqual("".join(d.get("reasoning_content") or "" for d in deltas), "think")
        self.assertEqual(frames[-1]["choices"], [])
        self.assertEqual(frames[-1]["usage"], usage)
        self.assertEqual({f["id"] for f in frames}, {"chatcmpl-upstream"})

    async def test_partial_literal_tag_is_flushed_before_finishing(self):
        frames = await self.collect([chunk({"content": "<vertex_"})])
        self.assertEqual(frames[-1]["choices"][0]["delta"]["content"], "<vertex_")

    async def test_initial_errors_and_empty_streams_fail_before_http_200(self):
        for events, failure in (([], None), ([chunk({"role": "assistant"})], RuntimeError("schema rejected"))):
            with self.subTest(events=events):
                with self.assertRaises((ValueError, RuntimeError)):
                    await self.response(events, failure=failure)

    async def test_midstream_errors_keep_the_error_and_close_the_upstream(self):
        closed = []
        frames = await self.collect([chunk({"content": "started"})],
                                    failure=RuntimeError("connection lost"), closed=closed)
        self.assertIn("connection lost", frames[-1]["error"]["message"])
        self.assertEqual(closed, [True])

    async def test_closing_downstream_closes_upstream(self):
        closed = []
        response = await self.response([chunk({"content": "one"}), chunk({"content": "two"})], closed=closed)
        await anext(response.body_iterator)
        await response.body_iterator.aclose()
        self.assertEqual(closed, [True])

    async def test_cleanup_failure_does_not_break_a_completed_response(self):
        client = client_for([chunk({"content": "answer"}, "stop")])
        source = client.chat.completions.create.return_value

        class Stream:
            def __aiter__(self):
                return source

            async def close(self):
                await source.aclose()
                raise RuntimeError("cleanup failed")

        client.chat.completions.create.return_value = Stream()
        request = OpenAIRequest(model="gemini-test", messages=[], stream=True)
        response = await OpenAIDirectHandler().handle_streaming_response(client, {}, {}, request)
        raw = [event async for event in response.body_iterator]
        self.assertEqual(raw[-1], "data: [DONE]\n\n")

    async def test_initial_upstream_http_status_is_preserved_by_the_route(self):
        app = FastAPI()
        app.include_router(router)
        app.state.credential_manager = SimpleNamespace(get_total_credentials=lambda: 0)
        app.state.express_key_manager = ExpressKeyManager(keys=["upstream-key"])
        upstream_request = httpx.Request("POST", "https://vertex.test")
        for status in (400, 429, 503):
            failure = httpx.HTTPStatusError("upstream rejected the request", request=upstream_request,
                                            response=httpx.Response(status, request=upstream_request))
            client = client_for([chunk({"role": "assistant"})], failure=failure)
            with self.subTest(status=status), patch("auth.API_KEY", ""), patch("auth.HUGGINGFACE", False), \
                 patch("openai_handler.discover_project_id", AsyncMock(return_value="test-project")), \
                 patch("openai_handler.ExpressClientWrapper", return_value=client):
                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://adapter") as http:
                    response = await http.post("/v1/chat/completions", json={
                        "model": "[EXPRESS] gemini-test-openai", "messages": [], "stream": True,
                    })
            self.assertEqual(response.status_code, status)
            self.assertIn("application/json", response.headers["content-type"])
            self.assertIn("upstream rejected", response.json()["error"]["message"])

    async def test_real_openai_sdk_consumes_the_direct_route(self):
        app = FastAPI()
        app.include_router(router)
        app.state.credential_manager = SimpleNamespace(get_total_credentials=lambda: 0)
        app.state.express_key_manager = ExpressKeyManager(keys=["upstream-key"])
        client = client_for([chunk(tool_delta('{"value":')), chunk({"tool_calls": [
            {"index": 0, "function": {"arguments": "1}"}},
        ]}), chunk(finish="tool_calls")])
        with patch("auth.API_KEY", ""), patch("auth.HUGGINGFACE", False), \
             patch("openai_handler.discover_project_id", AsyncMock(return_value="test-project")), \
             patch("openai_handler.ExpressClientWrapper", return_value=client):
            async with AsyncOpenAI(api_key="client-key", base_url="http://adapter/v1", max_retries=0,
                                   http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app))) as sdk:
                stream = await sdk.chat.completions.create(
                    model="[EXPRESS] gemini-test-openai", messages=[{"role": "user", "content": "Run echo"}], stream=True,
                )
                events = [event async for event in stream]
        self.assertEqual(events[-1].choices[0].finish_reason, "tool_calls")
        calls = [call for event in events for call in event.choices[0].delta.tool_calls or []]
        self.assertEqual("".join(call.function.arguments or "" for call in calls), '{"value":1}')
        self.assertEqual({event.id for event in events}, {"chatcmpl-upstream"})
        forwarded = client.chat.completions.create.call_args.kwargs
        self.assertIn("google", forwarded["extra_body"])
        self.assertNotIn("extra_body", forwarded["extra_body"])


class FakeStreamingTests(unittest.IsolatedAsyncioTestCase):
    @patch("config.FAKE_STREAMING_INTERVAL_SECONDS", 0)
    async def test_fake_stream_preserves_tool_extensions_reasoning_and_requested_usage(self):
        usage = {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
        completion = {
            "id": "chatcmpl-reply", "created": 123, "model": "gemini-test", "usage": usage,
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": "<vertex_think_tag>think</vertex_think_tag>answer",
                "tool_calls": [{key: value for key, value in tool_delta()["tool_calls"][0].items() if key != "index"}],
            }}],
        }
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
            create=AsyncMock(return_value=FakeChatCompletion(completion)),
        )))
        request = OpenAIRequest(model="gemini-test", messages=[], stream=True, stream_options={"include_usage": True})
        raw = [event async for event in openai_fake_stream_generator(client, {}, {}, request, False)]
        frames = [json.loads(event.removeprefix("data: ")) for event in raw[:-1]]
        deltas = [choice["delta"] for frame in frames for choice in frame["choices"]]
        calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
        self.assertEqual(calls[0]["extra_content"], {"google": {"thought_signature": "c2ln"}})
        self.assertEqual("".join(delta.get("reasoning_content") or "" for delta in deltas), "think")
        self.assertEqual("".join(delta.get("content") or "" for delta in deltas), "answer")
        self.assertEqual(frames[-1]["usage"], usage)
        self.assertEqual(frames[-1]["choices"], [])
        self.assertEqual(len({frame["id"] for frame in frames}), 1)
        self.assertEqual(len({frame["created"] for frame in frames}), 1)
        self.assertEqual(raw[-1], "data: [DONE]\n\n")


class ExpressSSETests(unittest.IsolatedAsyncioTestCase):
    async def decode(self, body):
        response = httpx.Response(200, request=httpx.Request("POST", "https://vertex.test"), text=body)
        wrapper = ExpressClientWrapper("project", "key")
        return [event.model_dump() async for event in wrapper._stream_generator(response)]

    async def test_optional_space_multiline_and_final_event_without_blank_line(self):
        events = await self.decode(': heartbeat\n\ndata:{"id":"one",\ndata: "choices":[]}\n\ndata: {"id":"two"}')
        self.assertEqual([event["id"] for event in events], ["one", "two"])
        self.assertEqual(await self.decode('data: [DONE]\n\ndata: {"ignored":true}\n\n'), [])

    async def test_upstream_errors_and_malformed_json_are_reported(self):
        for body, message in (
            ('data: {"error":{"message":"schema rejected"}}\n\n', "schema rejected"),
            ('event: error\ndata: {"message":"schema rejected"}\n\n', "schema rejected"),
            ('data: {broken}\n\n', "Invalid JSON"),
        ):
            with self.subTest(body=body), self.assertRaisesRegex(Exception, message):
                await self.decode(body)

    async def test_http_errors_preserve_google_diagnostics_without_the_key(self):
        response = httpx.Response(400, request=httpx.Request("POST", "https://vertex.test?key=server-secret"),
                                  json={"error": {"code": 400, "message": "sessionKey schema has no type"}})
        with self.assertRaises(Exception) as raised:
            ExpressClientWrapper._raise_for_status(response)
        self.assertEqual(raised.exception.status_code, 400)
        self.assertIn("sessionKey schema has no type", str(raised.exception))
        self.assertNotIn("server-secret", str(raised.exception))


class ReasoningProcessorTests(unittest.TestCase):
    def test_every_packet_boundary_preserves_all_visible_and_reasoning_text(self):
        text = "before<vertex_think_tag>one</vertex_think_tag>middle<vertex_think_tag>two</vertex_think_tag>after"
        for split in range(len(text) + 1):
            processor = StreamingReasoningProcessor()
            parts = [processor.process_chunk(text[:split]), processor.process_chunk(text[split:]), processor.flush_remaining()]
            self.assertEqual("".join(p[0] for p in parts), "beforemiddleafter")
            self.assertEqual("".join(p[1] for p in parts), "onetwo")

    def test_unclosed_reasoning_is_not_duplicated_or_exposed_as_content(self):
        processor = StreamingReasoningProcessor()
        first = processor.process_chunk("<vertex_think_tag>thinking</vertex_")
        last = processor.flush_remaining()
        self.assertEqual(first[0] + last[0], "")
        self.assertEqual(first[1] + last[1], "thinking</vertex_")


if __name__ == "__main__":
    unittest.main()
