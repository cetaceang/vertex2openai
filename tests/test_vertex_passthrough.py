import gzip
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch

import httpx
from fastapi import FastAPI, Request
from google import genai
from google.auth.exceptions import RefreshError
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from routes import vertex_api


class TrackingStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks
        self.consumed = 0
        self.closed = False

    async def __aiter__(self):
        for chunk in self.chunks:
            self.consumed += 1
            yield chunk

    async def aclose(self):
        self.closed = True


class Credentials:
    def before_request(self, auth_request, method, url, headers):
        headers["authorization"] = "Bearer google-token"
        headers["x-goog-user-project"] = "quota-project"


@patch("auth.API_KEY", "adapter-secret")
@patch("auth.HUGGINGFACE", False)
class VertexPassthroughTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.app = FastAPI()
        self.app.include_router(vertex_api.router)
        self.credentials = Credentials()
        self.credential_manager = SimpleNamespace(
            get_credentials=Mock(return_value=(self.credentials, "credential-project")),
            get_total_credentials=Mock(return_value=1),
        )
        self.express_manager = SimpleNamespace(get_total_keys=Mock(return_value=0), get_express_api_key=Mock(return_value=None))
        self.app.state.credential_manager = self.credential_manager
        self.app.state.express_key_manager = self.express_manager
        self.requests = []
        self.clients = []
        self.streams = []

    def factory(self, handler):
        def create():
            client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
            self.clients.append(client)
            return client
        return create

    def upstream_response(self, status=200, chunks=None, headers=None):
        stream = TrackingStream(chunks if chunks is not None else [b'{ "result": true }'])
        self.streams.append(stream)
        return httpx.Response(status, headers=headers, stream=stream)

    async def send(self, path, *, content=b"{}", headers=None, method="POST"):
        request_headers = {"Authorization": "Bearer adapter-secret", "Content-Type": "application/json"}
        if headers is not None:
            request_headers = headers
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://adapter") as client:
            return await client.request(method, path, content=content, headers=request_headers)

    async def test_native_body_query_and_response_are_preserved(self):
        body = b'{ "contents": [], "tools": [{"const": true}], "thoughtSignature": "opaque", "futureField": 1 }\n'
        upstream_body = b'{ "candidates": [], "futureResponseField": {"value": true} }\n'

        def handle(request):
            self.requests.append(request)
            return self.upstream_response(chunks=[upstream_body], headers={"Content-Type": "application/json", "x-request-id": "vertex-id"})

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            response = await self.send(
                "/v1beta1/projects/request-project/locations/us-central1/publishers/google/models/gemini-test:generateContent?tag=a%2Fb&tag=c+d",
                content=body,
            )

        sent = self.requests[0]
        self.assertEqual(sent.content, body)
        self.assertEqual(sent.url.host, "us-central1-aiplatform.googleapis.com")
        self.assertEqual(sent.url.path, "/v1beta1/projects/request-project/locations/us-central1/publishers/google/models/gemini-test:generateContent")
        self.assertEqual(sent.url.query, b"tag=a%2Fb&tag=c+d")
        self.assertEqual(sent.headers["authorization"], "Bearer google-token")
        self.assertEqual(sent.headers["x-goog-user-project"], "quota-project")
        self.assertNotIn("x-goog-api-key", sent.headers)
        self.assertEqual(response.content, upstream_body)
        self.assertEqual(response.headers["x-request-id"], "vertex-id")
        self.assertTrue(self.clients[0].is_closed)
        self.assertTrue(self.streams[0].closed)

    async def test_tool_schema_migration_is_opt_in_and_preserves_response(self):
        body = b'{ "tools": [{"functionDeclarations": [{"name": "cron", "parameters": {"type": "OBJECT", "properties": {"sessionKey": {}}}}]}] }\n'
        upstream_body = b'{ "originalVertexResponse": true }\n'

        def handle(request):
            self.requests.append(request)
            return self.upstream_response(chunks=[upstream_body], headers={"Content-Type": "application/json"})

        for enabled in (False, True):
            with self.subTest(enabled=enabled), \
                 patch.object(vertex_api.app_config, "VERTEX_TOOL_SCHEMA_COMPAT", enabled), \
                 patch.object(vertex_api, "_new_http_client", self.factory(handle)):
                response = await self.send("/v1/publishers/google/models/model:generateContent", content=body,
                                           headers={"Authorization": "Bearer adapter-secret", "Content-MD5": "stale"})
                sent = self.requests[-1]
                if enabled:
                    declaration = json.loads(sent.content)["tools"][0]["functionDeclarations"][0]
                    self.assertEqual(declaration["parametersJsonSchema"], {"type": "object", "properties": {"sessionKey": {}}})
                    self.assertNotIn("parameters", declaration)
                    self.assertNotIn("content-md5", sent.headers)
                else:
                    self.assertEqual(sent.content, body)
                self.assertEqual(response.content, upstream_body)
                self.assertEqual(int(sent.headers["content-length"]), len(sent.content))

    async def test_percent_encoded_method_reaches_schema_compat(self):
        body = b'{"tools":[{"functionDeclarations":[{"name":"cron","parameters":{"type":"OBJECT","properties":{"sessionKey":{}}}}]}]}'

        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        with patch.object(vertex_api.app_config, "VERTEX_TOOL_SCHEMA_COMPAT", True), \
             patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            await self.send("/v1/publishers/google/models/gemini-test%3AstreamGenerateContent?alt=sse", content=body)

        declaration = json.loads(self.requests[-1].content)["tools"][0]["functionDeclarations"][0]
        self.assertIn("parametersJsonSchema", declaration)
        self.assertNotIn(b"%3AstreamGenerateContent", self.requests[-1].url.raw_path)

    async def test_method_normalization_preserves_encoded_model_ids_and_queries(self):
        body = b'{"tools":[{"functionDeclarations":[{"name":"cron","parameters":{"type":"OBJECT"}}]}]}'

        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        prefixes = ("/v1/publishers/google/", "/v1beta1/projects/specified/locations/global/publishers/google/")
        for enabled in (False, True):
            for prefix in prefixes:
                for separator in (":", "%3A", "%3a"):
                    for method in ("generateContent", "streamGenerateContent", "countTokens"):
                        with self.subTest(enabled=enabled, prefix=prefix, separator=separator, method=method), \
                             patch.object(vertex_api.app_config, "VERTEX_TOOL_SCHEMA_COMPAT", enabled), \
                             patch.object(vertex_api, "_new_http_client", self.factory(handle)):
                            path = prefix + f"models/gemini-test%2Fvariant{separator}{method}?alt=sse&tag=a%2Fb&key=ingress-secret"
                            await self.send(path, content=body)
                            sent = self.requests[-1]
                            self.assertTrue(sent.url.raw_path.split(b"?", 1)[0].endswith(
                                f"models/gemini-test%2Fvariant:{method}".encode()))
                            self.assertEqual(sent.url.query, b"alt=sse&tag=a%2Fb")
                            declaration = json.loads(sent.content)["tools"][0]["functionDeclarations"][0]
                            self.assertEqual("parametersJsonSchema" in declaration, enabled)
                            if not enabled:
                                self.assertEqual(sent.content, body)

        for resource in (b"models/gemini%3Atest:generateContent", b"models/gemini%253AstreamGenerateContent"):
            self.assertEqual(vertex_api._normalize_method_delimiter(resource), resource)

    async def test_compatibility_does_not_rewrite_encoded_or_other_method_bodies(self):
        body = b'{"tools":[{"functionDeclarations":[{"name":"cron","parameters":{"type":"OBJECT"}}]}]}'

        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        with patch.object(vertex_api.app_config, "VERTEX_TOOL_SCHEMA_COMPAT", True), \
             patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            await self.send("/v1/publishers/google/models/model:otherMethod", content=body)
            compressed = gzip.compress(body)
            await self.send("/v1/publishers/google/models/model:generateContent", content=compressed,
                            headers={"Authorization": "Bearer adapter-secret", "Content-Encoding": "gzip"})
        self.assertEqual(self.requests[-2].content, body)
        self.assertEqual(self.requests[-1].content, compressed)

    async def test_native_google_auth_headers_and_query_use_the_adapter_key(self):
        for headers, query in (
            ({"x-goog-api-key": "adapter-secret"}, "alt=sse"),
            ({}, "%6Bey=adapter-secret&alt=sse&access_token=ingress-token"),
        ):
            with self.subTest(headers=headers):
                def handle(request):
                    self.requests.append(request)
                    return self.upstream_response()

                with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
                    response = await self.send(
                        f"/v1/projects/project/locations/global/publishers/google/models/model:countTokens?{query}",
                        headers=headers,
                    )

                self.assertEqual(response.status_code, 200)
                sent = self.requests[-1]
                self.assertEqual(sent.url.query, b"alt=sse")
                self.assertEqual(sent.headers["authorization"], "Bearer google-token")
                self.assertNotIn("x-goog-api-key", sent.headers)
                self.assertNotIn("adapter-secret", str(sent.url))

    async def test_express_short_path_uses_configured_key_project_and_global_region(self):
        self.express_manager.get_total_keys.return_value = 1
        self.express_manager.get_express_api_key.return_value = (0, "google-express-key")

        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)), \
             patch.object(vertex_api, "discover_project_id", AsyncMock(return_value="123456")), \
             patch.object(vertex_api.app_config, "VERTEX_EXPRESS_LOCATION", "europe-west4"):
            response = await self.send("/v1/publishers/google/models/gemini-test:countTokens")

        self.assertEqual(response.status_code, 200)
        sent = self.requests[0]
        self.assertEqual(sent.url.host, "aiplatform.googleapis.com")
        self.assertEqual(sent.url.path, "/v1/projects/123456/locations/global/publishers/google/models/gemini-test:countTokens")
        self.assertEqual(sent.headers["x-goog-api-key"], "google-express-key")
        self.assertNotIn("authorization", sent.headers)
        self.credential_manager.get_credentials.assert_not_called()

    async def test_full_path_falls_back_to_express_without_rewriting_the_resource(self):
        self.credential_manager.get_credentials.return_value = (None, None)
        self.express_manager.get_total_keys.return_value = 1
        self.express_manager.get_express_api_key.return_value = (0, "google-express-key")

        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)), \
             patch.object(vertex_api, "discover_project_id", AsyncMock()) as discover:
            response = await self.send("/v1/projects/specified/locations/global/publishers/google/models/gemini-test:generateContent")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.requests[0].url.host, "aiplatform.googleapis.com")
        self.assertIn("/projects/specified/locations/global/", self.requests[0].url.path)
        self.assertEqual(self.requests[0].headers["x-goog-api-key"], "google-express-key")
        discover.assert_not_awaited()

    async def test_client_key_passthrough_accepts_each_native_auth_input(self):
        self.credential_manager.get_credentials.return_value = (None, None)
        self.credential_manager.get_total_credentials.return_value = 0

        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        for headers, query, key in (
            ({"Authorization": "Bearer client-a"}, "alt=sse", "client-a"),
            ({"x-goog-api-key": "client-b"}, "alt=sse", "client-b"),
            ({}, "%6Bey=client-c&alt=sse", "client-c"),
        ):
            with self.subTest(key=key), patch("auth.API_KEY", ""), \
                 patch.object(vertex_api, "_new_http_client", self.factory(handle)):
                response = await self.send(
                    f"/v1/projects/client-project/locations/us-central1/publishers/google/models/model:generateContent?{query}",
                    headers=headers,
                )

            self.assertEqual(response.status_code, 200)
            sent = self.requests[-1]
            self.assertEqual(sent.headers["x-goog-api-key"], key)
            self.assertNotIn("authorization", sent.headers)
            self.assertEqual(sent.url.query, b"alt=sse")
            self.assertIn("/projects/client-project/locations/us-central1/", sent.url.path)

        self.assertEqual(self.express_manager.get_total_keys(), 0)

    async def test_client_key_short_path_discovers_its_own_project(self):
        self.credential_manager.get_credentials.return_value = (None, None)
        self.credential_manager.get_total_credentials.return_value = 0

        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        with patch("auth.API_KEY", ""), \
             patch.object(vertex_api, "_new_http_client", self.factory(handle)), \
             patch.object(vertex_api, "discover_project_id", AsyncMock(return_value="client-project")) as discover:
            response = await self.send("/v1/publishers/google/models/model:countTokens", headers={"x-goog-api-key": "client-key"})

        self.assertEqual(response.status_code, 200)
        discover.assert_awaited_once_with("client-key")
        self.assertIn("/projects/client-project/locations/global/", self.requests[0].url.path)
        self.assertEqual(self.requests[0].headers["x-goog-api-key"], "client-key")

    async def test_client_key_passthrough_requires_a_key_on_every_request(self):
        self.credential_manager.get_credentials.return_value = (None, None)
        self.credential_manager.get_total_credentials.return_value = 0
        with patch("auth.API_KEY", ""), patch.object(vertex_api, "_new_http_client") as factory:
            response = await self.send("/v1/publishers/google/models/model:generateContent", headers={})

        self.assertEqual(response.status_code, 401)
        factory.assert_not_called()

    async def test_configured_credentials_take_priority_without_adapter_auth(self):
        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        with patch("auth.API_KEY", ""), patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            response = await self.send("/v1/publishers/google/models/model:generateContent", headers={})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.requests[-1].headers["authorization"], "Bearer google-token")

        self.express_manager.get_total_keys.return_value = 1
        self.express_manager.get_express_api_key.return_value = (0, "server-key")
        with patch("auth.API_KEY", ""), \
             patch.object(vertex_api, "_new_http_client", self.factory(handle)), \
             patch.object(vertex_api, "discover_project_id", AsyncMock(return_value="server-project")):
            response = await self.send("/v1/publishers/google/models/model:generateContent", headers={"x-goog-api-key": "client-key"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.requests[-1].headers["x-goog-api-key"], "server-key")

    async def test_short_path_uses_service_account_project_when_no_express_key_exists(self):
        def handle(request):
            self.requests.append(request)
            return self.upstream_response()

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            response = await self.send("/v1/publishers/google/models", method="GET", content=b"")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(self.requests[0].method, "GET")
        self.assertEqual(self.requests[0].url.path, "/v1/projects/credential-project/locations/global/publishers/google/models")

    async def test_upstream_errors_keep_status_body_and_retry_headers(self):
        for status in (400, 401, 403, 429, 503):
            with self.subTest(status=status):
                body = b'{ "error": { "message": "Original Vertex error", "details": [1, 2] } }'

                def handle(request):
                    return self.upstream_response(status, [body], {"Content-Type": "application/json", "Retry-After": "5"})

                with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
                    response = await self.send("/v1/publishers/google/models/model:generateContent")

                self.assertEqual(response.status_code, status)
                self.assertEqual(response.content, body)
                self.assertEqual(response.headers["retry-after"], "5")

    async def test_upstream_json_error_reason_is_redacted_and_wire_bytes_are_preserved(self):
        body = json.dumps({"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": (
            "cron.sessionKey schema type is missing; adapter-secret google-token "
            "https://example.invalid/?key=private-query-value&alt=sse Bearer private-bearer-value"
        )}}).encode()
        for compressed in (False, True):
            with self.subTest(compressed=compressed):
                wire_body = gzip.compress(body) if compressed else body
                headers = {"Retry-After": "5", "Content-Length": str(len(wire_body))}
                if compressed:
                    headers["Content-Encoding"] = "gzip"

                def handle(request):
                    return self.upstream_response(400, [wire_body[:13], wire_body[13:]], headers)

                with patch.object(vertex_api, "_new_http_client", self.factory(handle)), \
                     patch.object(vertex_api, "emit_diagnostic") as log:
                    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://adapter") as client:
                        async with client.stream("POST", "/v1/publishers/google/models/model:generateContent",
                                                 headers={"Authorization": "Bearer adapter-secret"}, content=b"{}") as response:
                            raw = b"".join([data async for data in response.aiter_raw()])
                self.assertEqual(response.status_code, 400)
                self.assertEqual(raw, wire_body)
                self.assertEqual(response.headers["retry-after"], "5")
                details = next(call.kwargs for call in log.call_args_list if call.args == ("vertex_upstream_error",))
                self.assertIn("cron.sessionKey schema type is missing", details["message"])
                self.assertEqual(details["google_code"], 400)
                self.assertEqual(details["google_status"], "INVALID_ARGUMENT")
                self.assertFalse(details["body_truncated"])
                for secret in ("adapter-secret", "google-token", "private-query-value", "private-bearer-value"):
                    self.assertNotIn(secret, str(log.call_args_list))

    async def test_non_json_upstream_errors_have_bounded_diagnostics_without_content_type(self):
        body = b"<html><title>400 Bad Request</title><body>" + b"invalid route " * 1000 + b"</body></html>"

        def handle(request):
            return self.upstream_response(400, [body])

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)), \
             patch.object(vertex_api, "emit_diagnostic") as log:
            response = await self.send("/v1/publishers/google/models/model:generateContent")
        self.assertEqual(response.content, body)
        details = next(call.kwargs for call in log.call_args_list if call.args == ("vertex_upstream_error",))
        self.assertEqual(details["error_format"], "text")
        self.assertTrue(details["body_truncated"])
        self.assertIn("400 Bad Request", details["message"])
        self.assertNotIn("<html>", details["message"])
        self.assertLessEqual(len(details["message"]), 1536)

    async def test_hop_headers_are_removed_and_duplicate_end_to_end_headers_survive(self):
        def handle(request):
            self.requests.append(request)
            return self.upstream_response(headers=[
                ("Content-Type", "application/json"), ("Connection", "keep-alive, x-upstream-hop"),
                ("x-upstream-hop", "remove"), ("Transfer-Encoding", "chunked"),
                ("x-debug", "one"), ("x-debug", "two"),
            ])

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            response = await self.send("/v1/publishers/google/models/model:generateContent", headers={
                "Authorization": "Bearer adapter-secret", "Host": "adapter",
                "Connection": "keep-alive, x-client-hop", "x-client-hop": "remove", "x-request-id": "keep",
            })

        self.assertNotIn("x-client-hop", self.requests[0].headers.get("connection", ""))
        self.assertNotIn("x-client-hop", self.requests[0].headers)
        self.assertNotEqual(self.requests[0].headers["host"], "adapter")
        self.assertEqual(self.requests[0].headers["x-request-id"], "keep")
        self.assertNotIn("connection", response.headers)
        self.assertNotIn("transfer-encoding", response.headers)
        self.assertNotIn("x-upstream-hop", response.headers)
        self.assertEqual(response.headers.get_list("x-debug"), ["one", "two"])

    async def test_gzip_bytes_and_content_encoding_remain_consistent(self):
        body = gzip.compress(b'{"candidates": [{"text": "Native reply"}]}')

        def handle(request):
            return self.upstream_response(chunks=[body], headers={
                "Content-Type": "application/json", "Content-Encoding": "gzip", "Content-Length": str(len(body)),
            })

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            async with httpx.AsyncClient(transport=httpx.ASGITransport(app=self.app), base_url="http://adapter") as client:
                async with client.stream("POST", "/v1/publishers/google/models/model:generateContent", headers={
                    "Authorization": "Bearer adapter-secret",
                }, content=b"{}") as response:
                    raw = b"".join([data async for data in response.aiter_raw()])

        self.assertEqual(raw, body)
        self.assertEqual(response.headers["content-encoding"], "gzip")
        self.assertEqual(response.headers["content-length"], str(len(body)))

    async def test_sse_is_unmodified_and_connections_close_when_consumption_stops(self):
        fragments = [b'data: {"candidates": [', b'{"content": {"parts": []}}]}\n\n']

        def handle(request):
            return self.upstream_response(chunks=fragments, headers={"Content-Type": "text/event-stream"})

        path = "/v1/projects/project/locations/global/publishers/google/models/model:streamGenerateContent"
        async def receive():
            return {"type": "http.request", "body": b"{}", "more_body": False}

        request = Request({
            "type": "http", "method": "POST", "scheme": "http", "path": path, "raw_path": path.encode(),
            "query_string": b"alt=sse", "headers": [], "app": self.app,
        }, receive)
        with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            response = await vertex_api.vertex_resource(request, "v1", "project", "global", "models/model:streamGenerateContent")
            self.assertEqual(self.streams[0].consumed, 0)
            first = await anext(response.body_iterator)
            self.assertEqual(first, fragments[0])
            self.assertEqual(self.streams[0].consumed, 1)
            await response.body_iterator.aclose()
            await response.background()

        self.assertTrue(self.clients[0].is_closed)
        self.assertTrue(self.streams[0].closed)

        with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
            complete = await self.send(path + "?alt=sse")
        self.assertEqual(complete.content, b"".join(fragments))
        self.assertEqual(complete.headers["content-type"], "text/event-stream")

    async def test_missing_adapter_auth_is_rejected_before_any_upstream_call(self):
        with patch.object(vertex_api, "_new_http_client") as factory:
            response = await self.send("/v1/publishers/google/models/model:generateContent", headers={})

        self.assertEqual(response.status_code, 401)
        factory.assert_not_called()

    async def test_native_google_sdk_can_use_the_adapter_as_its_base_url(self):
        def handle(request):
            self.requests.append(request)
            return self.upstream_response(chunks=[
                b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Native reply"}]},"finishReason":"STOP"}]}\n\n',
            ], headers={"Content-Type": "text/event-stream"})

        client = genai.Client(
            vertexai=True, api_key="adapter-secret",
            http_options=types.HttpOptions(
                base_url="http://adapter", api_version="v1",
                async_client_args={"transport": httpx.ASGITransport(app=self.app)},
            ),
        )
        try:
            with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
                stream = await client.aio.models.generate_content_stream(model="gemini-test", contents="Hello")
                responses = [response async for response in stream]
        finally:
            await client.aio.aclose()
            client.close()

        self.assertEqual("".join(response.text or "" for response in responses), "Native reply")
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.requests[0].headers["authorization"], "Bearer google-token")
        self.assertNotIn("x-goog-api-key", self.requests[0].headers)

    async def test_missing_and_failed_credentials_return_explicit_errors(self):
        self.credential_manager.get_credentials.return_value = (None, None)
        with patch.object(vertex_api, "_new_http_client") as factory:
            missing = await self.send("/v1/publishers/google/models/model:generateContent")
        self.assertEqual(missing.status_code, 503)
        factory.assert_not_called()

        self.credential_manager.get_credentials.return_value = (Mock(before_request=Mock(side_effect=RefreshError("expired"))), "project")
        with patch.object(vertex_api, "_new_http_client") as factory:
            failed = await self.send("/v1/publishers/google/models/model:generateContent")
        self.assertEqual(failed.status_code, 503)
        self.assertEqual(failed.json()["error"]["status"], "UNAUTHENTICATED")
        factory.assert_not_called()

    async def test_connection_failures_return_gateway_errors_and_close_clients(self):
        for exception, status in ((httpx.ConnectError, 502), (httpx.ConnectTimeout, 504)):
            with self.subTest(status=status):
                def handle(request):
                    raise exception("offline", request=request)

                with patch.object(vertex_api, "_new_http_client", self.factory(handle)):
                    response = await self.send("/v1/publishers/google/models/model:generateContent")

                self.assertEqual(response.status_code, status)
                self.assertEqual(response.json()["error"]["code"], status)
                self.assertTrue(self.clients[-1].is_closed)


if __name__ == "__main__":
    unittest.main()
