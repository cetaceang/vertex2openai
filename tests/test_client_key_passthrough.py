import asyncio
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI
from google import genai

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

import config
from express_key_manager import ExpressKeyManager
from routes import chat_api, models_api


class ClientKeyPassthroughTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        for target, value in (
            ("auth.API_KEY", ""), ("auth.HUGGINGFACE", False),
            ("config.VERTEX_EXPRESS_API_KEY_VAL", []),
            ("config.VERTEX_EXPRESS_LOCATION", "global"),
            ("config.FAKE_STREAMING_ENABLED", False),
            ("config.PROXY_URL", None), ("config.SSL_CERT_FILE", None),
        ):
            patcher = patch(target, value)
            patcher.start()
            self.addCleanup(patcher.stop)

        self.app = FastAPI()
        self.app.include_router(chat_api.router)
        self.app.include_router(models_api.router)
        self.credentials = SimpleNamespace(
            get_total_credentials=Mock(return_value=0),
            get_credentials=Mock(return_value=(None, None)),
        )
        self.express_keys = ExpressKeyManager()
        self.app.state.credential_manager = self.credentials
        self.app.state.express_key_manager = self.express_keys
        self.http = httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://adapter")
        self.addAsyncCleanup(self.http.aclose)
        self.requests = []
        self.sdk_clients = []
        self.sdk_client_class = genai.Client

    async def asyncTearDown(self):
        for client in self.sdk_clients:
            await client.aio.aclose()
            client.close()

    def upstream(self, request):
        self.requests.append(request)
        reply = {"candidates": [{
            "content": {"role": "model", "parts": [{"text": "Client key reply"}]},
            "finishReason": "STOP",
        }]}
        return httpx.Response(200, json=reply)

    def sdk_client(self, **options):
        http_options = options["http_options"]
        http_options.async_client_args = {"transport": httpx.MockTransport(self.upstream)}
        client = self.sdk_client_class(**options)
        self.sdk_clients.append(client)
        return client

    async def chat(self, key=None, model="gemini-2.5-flash", stream=False, headers=None):
        if headers is None:
            headers = {"Authorization": f"Bearer {key}"} if key is not None else {}
        return await self.http.post("/v1/chat/completions", headers=headers, json={
            "model": model, "messages": [{"role": "user", "content": "Hello"}], "stream": stream,
        })

    async def test_concurrent_chat_requests_use_their_own_keys(self):
        with patch.object(chat_api.genai, "Client", side_effect=self.sdk_client), \
             patch.object(chat_api, "discover_project_id", AsyncMock(return_value="client-project")):
            responses = await asyncio.gather(
                self.chat("client-a"), self.chat("client-b", "[EXPRESS] gemini-2.5-flash"),
            )

        for response in responses:
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(response.json()["choices"][0]["message"]["content"], "Client key reply")
        self.assertEqual({request.headers["x-goog-api-key"] for request in self.requests}, {"client-a", "client-b"})
        for request in self.requests:
            self.assertIn("/projects/client-project/locations/global/", request.url.path)
        self.assertEqual(self.express_keys.get_total_keys(), 0)
        self.assertEqual(config.VERTEX_EXPRESS_API_KEY_VAL, [])
        missing = await self.chat()
        self.assertEqual(missing.status_code, 401)

    async def test_chat_stream_uses_client_key_and_preserves_sse(self):
        def stream(request):
            self.requests.append(request)
            return httpx.Response(200, headers={"Content-Type": "text/event-stream"}, content=(
                b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Stream reply"}]},"finishReason":"STOP"}]}\n\n'
            ))

        with patch.object(self, "upstream", side_effect=stream), \
             patch.object(chat_api.genai, "Client", side_effect=self.sdk_client), \
             patch.object(chat_api, "discover_project_id", AsyncMock(return_value="client-project")):
            response = await self.chat("stream-key", stream=True)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn("Stream reply", response.text)
        self.assertIn("data: [DONE]", response.text)
        self.assertEqual(self.requests[0].headers["x-goog-api-key"], "stream-key")

    async def test_direct_openai_variants_use_client_key(self):
        reply = {"id": "chatcmpl-test", "object": "chat.completion", "created": 1, "model": "google/gemini-2.5-flash", "choices": [
            {"index": 0, "message": {"role": "assistant", "content": "Direct reply"}, "finish_reason": "stop"},
        ]}
        http_client_class = httpx.AsyncClient

        def upstream(request):
            self.requests.append(request)
            return httpx.Response(200, json=reply)

        def upstream_client(**options):
            return http_client_class(transport=httpx.MockTransport(upstream), **options)

        with patch("openai_handler.discover_project_id", AsyncMock(return_value="direct-project")), \
             patch("openai_handler.httpx.AsyncClient", side_effect=upstream_client):
            for model in ("gemini-2.5-flash-openai", "[EXPRESS] gemini-2.5-flash-openaisearch"):
                with self.subTest(model=model):
                    response = await self.chat("direct-key", model)
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertEqual(response.json()["choices"][0]["message"]["content"], "Direct reply")

        for request in self.requests:
            self.assertEqual(request.url.params["key"], "direct-key")
            self.assertIn("/projects/direct-project/locations/global/endpoints/openapi/chat/completions", request.url.path)
        self.assertEqual(self.express_keys.get_total_keys(), 0)

    async def test_configured_express_key_takes_priority(self):
        self.app.state.express_key_manager = ExpressKeyManager(keys=["server-key"])
        with patch.object(chat_api.genai, "Client", side_effect=self.sdk_client), \
             patch.object(chat_api, "discover_project_id", AsyncMock(return_value="server-project")):
            response = await self.chat("client-key", "[EXPRESS] gemini-2.5-flash")
            anonymous = await self.chat(model="[EXPRESS] gemini-2.5-flash")

        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(anonymous.status_code, 200, anonymous.text)
        self.assertEqual([request.headers["x-goog-api-key"] for request in self.requests], ["server-key", "server-key"])

    async def test_configured_service_account_takes_priority(self):
        self.credentials.get_total_credentials.return_value = 1
        credential = object()
        self.credentials.get_credentials.return_value = (credential, "sa-project")
        with patch.object(chat_api.genai, "Client") as client, \
             patch.object(chat_api, "execute_gemini_call", AsyncMock(return_value={"ok": True})):
            response = await self.chat("client-key")

        self.assertEqual(response.status_code, 200)
        client.assert_called_once_with(vertexai=True, credentials=credential, project="sa-project", location="global")

    async def test_configured_adapter_auth_rejects_foreign_keys_and_does_not_forward_its_key(self):
        with patch("auth.API_KEY", "adapter-secret"), patch.object(chat_api.genai, "Client") as client:
            for key in (None, "client-key"):
                response = await self.chat(key)
                self.assertEqual(response.status_code, 401)
            response = await self.chat("adapter-secret", "[EXPRESS] gemini-2.5-flash")
            self.assertEqual(response.status_code, 401)
        client.assert_not_called()

    async def test_hugging_face_auth_still_requires_its_token(self):
        with patch("auth.HUGGINGFACE", True), patch.object(chat_api.genai, "Client") as client:
            response = await self.chat("client-key")
        self.assertEqual(response.status_code, 401)
        self.assertIn("x-ip-token", response.json()["detail"])
        client.assert_not_called()

    async def test_missing_or_malformed_client_keys_do_not_call_upstream(self):
        with patch.object(chat_api.genai, "Client") as client:
            for headers in ({}, {"Authorization": "Bearer "}, {"Authorization": "Basic value"}):
                with self.subTest(headers=headers):
                    response = await self.chat(headers=headers)
                    self.assertEqual(response.status_code, 401)
        client.assert_not_called()

    async def test_models_are_available_with_a_client_key(self):
        with patch.object(models_api, "refresh_models_config_cache", AsyncMock()), \
             patch.object(models_api, "get_vertex_models", AsyncMock(return_value=["gemini-sa"])), \
             patch.object(models_api, "get_vertex_express_models", AsyncMock(return_value=["gemini-2.5-flash"])):
            response = await self.http.get("/v1/models", headers={"Authorization": "Bearer client-key"})

        self.assertEqual(response.status_code, 200)
        model_ids = [model["id"] for model in response.json()["data"]]
        self.assertIn("[EXPRESS] gemini-2.5-flash", model_ids)
        self.assertIn("[EXPRESS] gemini-2.5-flash-openai", model_ids)
        self.assertFalse(any(model.startswith("[PAY]") for model in model_ids))


if __name__ == "__main__":
    unittest.main()
