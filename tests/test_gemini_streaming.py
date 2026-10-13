import asyncio
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from fastapi import FastAPI
from google.genai import types
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from api_helpers import execute_gemini_call
from auth import get_api_key
from message_processing import GeminiStreamState, convert_chunk_to_openai
from models import OpenAIRequest
from routes.chat_api import router as chat_router


def chunk(parts=None, finish=None):
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role="model", parts=parts or []), finish_reason=finish,
    )])


def function_call(name, call_id):
    return types.Part(function_call=types.FunctionCall(name=name, id=call_id, args={"city": name}))


def decode(event):
    return json.loads(event.removeprefix("data: "))["choices"][0]


def fake_client(stream):
    return SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(
        generate_content_stream=AsyncMock(return_value=stream),
    )))


def app_for_client(client):
    app = FastAPI()
    app.include_router(chat_router)
    app.state.credential_manager = SimpleNamespace(get_credentials=lambda: (object(), "test-project"))
    app.state.express_key_manager = SimpleNamespace(get_total_keys=lambda: 0)
    app.dependency_overrides[get_api_key] = lambda: "test"
    return app


class GeminiStreamConversionTests(unittest.TestCase):
    def test_parallel_calls_in_one_chunk_keep_all_calls_and_correct_finish_reason(self):
        event = convert_chunk_to_openai(chunk([
            function_call("weather", "fc-weather"), function_call("time", "fc-time"),
        ], "STOP"), "gemini-test", "response-id")

        choice = decode(event)
        calls = choice["delta"]["tool_calls"]
        self.assertEqual(choice["delta"]["role"], "assistant")
        self.assertEqual([call["index"] for call in calls], [0, 1])
        self.assertEqual([call["id"] for call in calls], ["fc-weather", "fc-time"])
        self.assertEqual(choice["finish_reason"], "tool_calls")

    def test_indices_and_tool_finish_reason_persist_across_chunks(self):
        state = GeminiStreamState()
        first = decode(convert_chunk_to_openai(
            chunk([function_call("weather", "fc-one")]), "gemini-test", "response-id", stream_state=state,
        ))
        second = decode(convert_chunk_to_openai(
            chunk([function_call("weather", "fc-two")]), "gemini-test", "response-id", stream_state=state,
        ))
        final = decode(convert_chunk_to_openai(
            chunk(finish="STOP"), "gemini-test", "response-id", stream_state=state,
        ))

        self.assertEqual(first["delta"]["tool_calls"][0]["index"], 0)
        self.assertEqual(second["delta"]["tool_calls"][0]["index"], 1)
        self.assertNotIn("role", second["delta"])
        self.assertEqual(final["finish_reason"], "tool_calls")

    def test_text_and_reasoning_are_preserved_alongside_tool_calls(self):
        event = convert_chunk_to_openai(chunk([
            types.Part(text="Thinking", thought=True),
            types.Part(text="Checking the weather."),
            function_call("weather", "fc-weather"),
        ], "STOP"), "gemini-test", "response-id")

        delta = decode(event)["delta"]
        self.assertEqual(delta["reasoning_content"], "Thinking")
        self.assertEqual(delta["content"], "Checking the weather.")
        self.assertEqual(len(delta["tool_calls"]), 1)

    def test_malformed_calls_are_reported_with_the_upstream_reason(self):
        with self.assertRaisesRegex(ValueError, "MALFORMED_FUNCTION_CALL"):
            convert_chunk_to_openai(chunk(finish="MALFORMED_FUNCTION_CALL"), "gemini-test", "response-id")


@patch("api_helpers.app_config.FAKE_STREAMING_ENABLED", False)
class GeminiStreamResponseTests(unittest.IsolatedAsyncioTestCase):
    async def test_deferred_upstream_errors_return_http_500_before_streaming(self):
        async def stream():
            raise ValueError("upstream rejected the tool schema")
            yield

        client = fake_client(stream())
        app = app_for_client(client)
        with patch("routes.chat_api.genai.Client", return_value=client):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://adapter") as http:
                response = await http.post("/v1/chat/completions", json={
                    "model": "gemini-test", "messages": [{"role": "user", "content": "Hello"}], "stream": True,
                })

        self.assertEqual(response.status_code, 500)
        self.assertIn("upstream rejected the tool schema", response.json()["error"]["message"])

    async def test_empty_and_blocked_responses_raise_before_http_200(self):
        cases = [
            ([], "empty response stream"),
            ([types.GenerateContentResponse(usage_metadata=types.GenerateContentResponseUsageMetadata())], "empty response stream"),
            ([chunk(finish="STOP")], "no text or tool calls"),
            ([chunk(finish="MALFORMED_FUNCTION_CALL")], "MALFORMED_FUNCTION_CALL"),
            ([types.GenerateContentResponse(prompt_feedback=types.GenerateContentResponsePromptFeedback(block_reason="SAFETY"))], "blocked"),
        ]
        for upstream_chunks, message in cases:
            with self.subTest(message=message):
                async def stream():
                    for upstream_chunk in upstream_chunks:
                        yield upstream_chunk

                request = OpenAIRequest(model="gemini-test", messages=[], stream=True)
                with self.assertRaisesRegex(ValueError, message):
                    await execute_gemini_call(fake_client(stream()), request.model, lambda messages: [], {}, request)

    async def test_reasoning_is_forwarded_without_waiting_for_the_final_answer(self):
        allow_answer = asyncio.Event()

        async def stream():
            yield chunk([types.Part(text="Thinking", thought=True)])
            await allow_answer.wait()
            yield chunk([types.Part(text="Answer")], "STOP")

        request = OpenAIRequest(model="gemini-test", messages=[], stream=True)
        response = await asyncio.wait_for(execute_gemini_call(
            fake_client(stream()), request.model, lambda messages: [], {}, request,
        ), timeout=1)
        first = await anext(response.body_iterator)
        self.assertEqual(decode(first)["delta"]["reasoning_content"], "Thinking")
        allow_answer.set()
        remaining = [event async for event in response.body_iterator]
        self.assertEqual(remaining[-1], "data: [DONE]\n\n")

    async def test_missing_terminal_chunk_is_completed_for_openai_clients(self):
        async def stream():
            yield chunk([function_call("weather", "fc-one"), function_call("time", "fc-two")])
            yield chunk([function_call("weather", "fc-three")])

        client = fake_client(stream())
        app = app_for_client(client)
        with patch("routes.chat_api.genai.Client", return_value=client):
            async with AsyncOpenAI(
                api_key="test", base_url="http://adapter/v1", max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.ASGITransport(app=app)),
            ) as openai_client:
                result = await openai_client.chat.completions.create(
                    model="gemini-test", messages=[{"role": "user", "content": "Run the tools"}], stream=True,
                )
                events = [event async for event in result]

        calls = [call for event in events for call in (event.choices[0].delta.tool_calls or [])]
        self.assertEqual([call.index for call in calls], [0, 1, 2])
        self.assertEqual([call.id for call in calls], ["fc-one", "fc-two", "fc-three"])
        self.assertEqual(events[0].choices[0].delta.role, "assistant")
        self.assertEqual(events[-1].choices[0].finish_reason, "tool_calls")


if __name__ == "__main__":
    unittest.main()
