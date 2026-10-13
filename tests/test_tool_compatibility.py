import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch

import httpx
from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from api_helpers import create_generation_config, execute_gemini_call
from message_processing import (
    _encode_tool_call_id_with_thought_signature,
    create_gemini_prompt,
)
from models import OpenAIMessage, OpenAIRequest


def complex_tool_schema():
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "job": {
                "type": "object",
                "properties": {
                    "delivery": {
                        "type": "object",
                        "properties": {
                            "failureDestination": {
                                "type": "object",
                                "properties": {
                                    "mode": {
                                        "anyOf": [
                                            {"type": "string", "const": "announce"},
                                            {"type": "string", "const": "webhook"},
                                        ]
                                    }
                                },
                            }
                        },
                    }
                },
            },
            "confirm": {"type": "boolean", "const": True},
            "env": {
                "type": "object",
                "patternProperties": {"^.*$": {"type": "string"}},
            },
            "args": {"type": "object", "propertyNames": {"type": "string"}},
            "max_length": {
                "type": "integer",
                "exclusiveMinimum": 0,
                "exclusiveMaximum": 1000000,
            },
            "operations": {
                "type": "array",
                "items": {
                    "oneOf": [
                        {
                            "type": "object",
                            "properties": {"type": {"const": "create"}},
                            "additionalProperties": False,
                        },
                        {
                            "type": "object",
                            "properties": {"type": {"const": "delete"}},
                            "additionalProperties": False,
                        },
                    ]
                },
            },
            "optional": {"type": ["string", "null"]},
            "options": {"$ref": "#/$defs/options"},
        },
        "$defs": {
            "options": {
                "type": "object",
                "properties": {"const": {"type": "string"}},
            }
        },
        "required": ["confirm"],
        "additionalProperties": False,
    }


def tool_call(call_id, name):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": "{}"},
    }


class ToolSchemaTests(unittest.TestCase):
    def test_nested_json_schema_passes_sdk_validation_without_losing_constraints(self):
        schema = complex_tool_schema()
        original = copy.deepcopy(schema)
        request = OpenAIRequest(
            model="gemini-test",
            messages=[OpenAIMessage(role="user", content="Run the tool")],
            tools=[{"type": "function", "function": {"name": "run", "parameters": schema}}],
        )

        config = types.GenerateContentConfig(**create_generation_config(request))
        declaration = config.tools[0].function_declarations[0]

        expected = {key: value for key, value in original.items() if key != "$schema"}
        self.assertEqual(declaration.parameters_json_schema, expected)
        self.assertIsNone(declaration.parameters)
        self.assertEqual(schema, original)
        self.assertEqual(request.tools[0]["function"]["parameters"], original)

    def test_parameterless_and_empty_schema_tools_remain_valid(self):
        request = OpenAIRequest(
            model="gemini-test",
            messages=[],
            tools=[
                {"type": "function", "function": {"name": "no_parameters"}},
                {"type": "function", "function": {"name": "empty_schema", "parameters": {}}},
            ],
        )

        config = types.GenerateContentConfig(**create_generation_config(request))
        declarations = config.tools[0].function_declarations
        self.assertIsNone(declarations[0].parameters_json_schema)
        self.assertEqual(declarations[1].parameters_json_schema, {})


class ToolMessageTests(unittest.TestCase):
    def test_parallel_tool_results_without_names_are_matched_by_id(self):
        messages = [
            OpenAIMessage(role="user", content="Check both cities"),
            OpenAIMessage(
                role="assistant",
                tool_calls=[tool_call("fc-first", "weather"), tool_call("fc-second", "time")],
            ),
            OpenAIMessage(role="tool", tool_call_id="fc-second", content='{"hour": 12}'),
            OpenAIMessage(role="tool", tool_call_id="fc-first", content="sunny"),
        ]

        converted = create_gemini_prompt(messages)

        self.assertEqual(len(converted), 3)
        self.assertEqual(converted[-1].role, "user")
        responses = [part.function_response for part in converted[-1].parts]
        self.assertEqual([response.name for response in responses], ["time", "weather"])
        self.assertEqual([response.id for response in responses], ["fc-second", "fc-first"])
        self.assertEqual([response.response for response in responses], [{"hour": 12}, {"result": "sunny"}])

    def test_tool_ids_and_thought_signatures_round_trip(self):
        signature = b"thought-signature\x00\xff"
        signed_id = _encode_tool_call_id_with_thought_signature("fc-original", signature)
        for response_id in (signed_id, "fc-original"):
            with self.subTest(response_id=response_id):
                converted = create_gemini_prompt([
                    OpenAIMessage(role="assistant", tool_calls=[tool_call(signed_id, "run")]),
                    OpenAIMessage(role="tool", tool_call_id=response_id, content="done"),
                ])

                self.assertEqual(converted[0].parts[0].function_call.id, "fc-original")
                self.assertEqual(converted[0].parts[0].thought_signature, signature)
                self.assertEqual(converted[1].parts[0].function_response.id, "fc-original")
                self.assertEqual(converted[1].parts[0].function_response.name, "run")

    def test_tool_result_arrays_scalars_and_content_blocks_become_response_objects(self):
        cases = [
            ('{"ok": true}', {"ok": True}),
            ("[1, 2]", {"result": [1, 2]}),
            ("plain text", {"result": "plain text"}),
            ("", {"result": ""}),
            ("null", {"result": None}),
            ("true", {"result": True}),
            ([], {"result": []}),
            ([{"type": "text", "text": "done"}], {"result": [{"type": "text", "text": "done"}]}),
        ]
        for content, expected in cases:
            with self.subTest(content=content):
                converted = create_gemini_prompt([
                    OpenAIMessage(role="assistant", tool_calls=[tool_call("fc-output", "run")]),
                    OpenAIMessage(role="tool", tool_call_id="fc-output", content=content),
                ])

                response = converted[-1].parts[0].function_response
                self.assertEqual(response.model_dump(mode="json")["response"], expected)

    def test_explicit_names_and_legacy_ids_still_work_without_call_history(self):
        for name, call_id in (("run", "opaque-id"), (None, "call_example_0_run_123")):
            with self.subTest(name=name, call_id=call_id):
                converted = create_gemini_prompt([
                    OpenAIMessage(role="tool", name=name, tool_call_id=call_id, content="done")
                ])

                self.assertEqual(converted[0].parts[0].function_response.name, "run")


class GeminiStreamingTests(unittest.IsolatedAsyncioTestCase):
    @patch("api_helpers.app_config.FAKE_STREAMING_ENABLED", False)
    async def test_sdk_initialization_errors_propagate_before_streaming_response(self):
        request = OpenAIRequest(model="gemini-test", messages=[], stream=True)
        generate = AsyncMock(side_effect=ValueError("invalid tool schema"))
        client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate)))

        with self.assertRaisesRegex(ValueError, "invalid tool schema"):
            await execute_gemini_call(client, request.model, lambda messages: [], {}, request)

        generate.assert_awaited_once()

    @patch("api_helpers.app_config.FAKE_STREAMING_ENABLED", False)
    async def test_midstream_errors_finish_as_sse_without_asgi_exception(self):
        async def failing_stream():
            yield types.GenerateContentResponse(
                candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text="started")]))]
            )
            raise RuntimeError("upstream connection lost")

        for is_auto_attempt in (False, True):
            with self.subTest(is_auto_attempt=is_auto_attempt):
                request = OpenAIRequest(model="gemini-test", messages=[], stream=True)
                generate = AsyncMock(return_value=failing_stream())
                client = SimpleNamespace(aio=SimpleNamespace(models=SimpleNamespace(generate_content_stream=generate)))

                response = await execute_gemini_call(
                    client, request.model, lambda messages: [], {}, request, is_auto_attempt=is_auto_attempt
                )
                chunks = [chunk async for chunk in response.body_iterator]

                self.assertEqual(chunks[-1], "data: [DONE]\n\n")
                error = json.loads(chunks[-2].removeprefix("data: ").strip())
                self.assertEqual(error["error"]["message"], "upstream connection lost")
                self.assertEqual(sum(chunk == "data: [DONE]\n\n" for chunk in chunks), 1)

    @patch("api_helpers.app_config.FAKE_STREAMING_ENABLED", False)
    async def test_real_sdk_serializes_complex_tools_and_results_for_vertex(self):
        requests = []

        def handle_request(http_request):
            requests.append(json.loads(http_request.content))
            response = {
                "candidates": [{
                    "content": {"role": "model", "parts": [{"text": "done"}]},
                    "finishReason": "STOP",
                }]
            }
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                text=f"data: {json.dumps(response)}\n\n",
            )

        client = genai.Client(
            vertexai=True,
            api_key="test-api-key",
            http_options=types.HttpOptions(
                base_url="https://vertex.test/v1/projects/test/locations/global",
                api_version="",
                async_client_args={"transport": httpx.MockTransport(handle_request)},
            ),
        )
        schema = complex_tool_schema()
        request = OpenAIRequest(
            model="gemini-test",
            messages=[
                OpenAIMessage(role="user", content="Run the tool"),
                OpenAIMessage(role="assistant", tool_calls=[tool_call("fc-run", "run")]),
                OpenAIMessage(role="tool", tool_call_id="fc-run", content="[1, 2]"),
            ],
            tools=[{"type": "function", "function": {"name": "run", "parameters": schema}}],
            stream=True,
        )
        try:
            response = await execute_gemini_call(
                client, request.model, create_gemini_prompt, create_generation_config(request), request
            )
            chunks = [chunk async for chunk in response.body_iterator]
        finally:
            await client.aio.aclose()
            client.close()

        self.assertEqual(len(requests), 1)
        declaration = requests[0]["tools"][0]["functionDeclarations"][0]
        self.assertNotIn("parameters", declaration)
        self.assertEqual(
            declaration["parametersJsonSchema"],
            {key: value for key, value in schema.items() if key != "$schema"},
        )
        self.assertEqual(requests[0]["contents"][-1], {
            "role": "user",
            "parts": [{"function_response": {"name": "run", "id": "fc-run", "response": {"result": [1, 2]}}}],
        })
        self.assertFalse(any('"error"' in chunk for chunk in chunks))
        self.assertIn('"content": "done"', "".join(chunks))
        self.assertEqual(chunks[-1], "data: [DONE]\n\n")


if __name__ == "__main__":
    unittest.main()
