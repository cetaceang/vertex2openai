"""Forward native Vertex requests, with optional tool schema compatibility."""

import logging
import re
import ssl
from typing import Literal, Optional
from urllib.parse import quote, unquote_to_bytes

import anyio
import httpx
from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse, StreamingResponse
from google.auth.exceptions import GoogleAuthError
from google.auth.transport.requests import Request as AuthRequest
from starlette.background import BackgroundTask
from starlette.concurrency import run_in_threadpool

import config as app_config
from auth import client_key_passthrough_enabled, get_api_key
from project_id_discovery import discover_project_id
from stream_diagnostics import emit_diagnostic
from tool_schema_compat import rewrite_vertex_tool_schemas


logger = logging.getLogger(__name__)
HOP_BY_HOP_HEADERS = {
    b"connection", b"keep-alive", b"proxy-authenticate", b"proxy-authorization",
    b"te", b"trailer", b"transfer-encoding", b"upgrade",
}
CLIENT_AUTH_HEADERS = {b"authorization", b"x-goog-api-key", b"x-ip-token"}


async def get_vertex_api_key(request: Request):
    authorization = request.headers.get("authorization")
    if authorization is None:
        api_key = request.headers.get("x-goog-api-key") or request.query_params.get("key")
        if api_key is not None:
            authorization = f"Bearer {api_key}"
    return await get_api_key(authorization=authorization, x_ip_token=request.headers.get("x-ip-token"))


router = APIRouter(tags=["Vertex AI passthrough"])


def _google_error(code: int, message: str, status: str):
    return JSONResponse(status_code=code, content={"error": {"code": code, "message": message, "status": status}})


def _forward_headers(raw_headers, excluded=()):
    blocked = HOP_BY_HOP_HEADERS | set(excluded)
    for name, value in raw_headers:
        if name.lower() == b"connection":
            blocked.update(token.strip().lower() for token in value.split(b","))
    return [(name, value) for name, value in raw_headers if name.lower() not in blocked]


def _forward_query(raw_query: bytes) -> bytes:
    # Remove ingress credentials; the selected key is sent in an upstream header.
    return b"&".join(
        part for part in raw_query.split(b"&")
        if unquote_to_bytes(part.partition(b"=")[0].replace(b"+", b" "))
        not in {b"key", b"access_token", b"oauth_token"}
    )


def _new_http_client():
    options = {"timeout": httpx.Timeout(None, connect=30.0), "follow_redirects": False}
    if app_config.PROXY_URL:
        options["proxy"] = app_config.PROXY_URL
    if app_config.SSL_CERT_FILE:
        options["verify"] = ssl.create_default_context(cafile=app_config.SSL_CERT_FILE)
    return httpx.AsyncClient(**options)


async def _forward_vertex(
    request: Request,
    api_version: str,
    project: Optional[str],
    location: Optional[str],
    api_key: str,
):
    express_manager = request.app.state.express_key_manager
    credentials = None
    credential_project = None
    # The short publisher path selects Express when keys are configured.
    # A full resource path selects service accounts first and keeps its project/region.
    if project is not None or express_manager.get_total_keys() == 0:
        credentials, credential_project = request.app.state.credential_manager.get_credentials()

    headers = httpx.Headers(_forward_headers(
        request.headers.raw, CLIENT_AUTH_HEADERS | {b"host", b"content-length"},
    ))
    if credentials is None:
        selected_key = express_manager.get_express_api_key()
        if selected_key is None:
            if not client_key_passthrough_enabled(request.app.state.credential_manager, express_manager):
                return _google_error(503, "No Vertex credentials are configured.", "FAILED_PRECONDITION")
            if not api_key:
                return _google_error(401, "Missing client Vertex Express API key.", "UNAUTHENTICATED")
            express_key = api_key
        else:
            _, express_key = selected_key
        headers["x-goog-api-key"] = express_key
        if project is None:
            try:
                project = await discover_project_id(express_key)
            except Exception:
                logger.exception("Could not discover the Vertex Express project")
                return _google_error(502, "Could not determine the Vertex Express project.", "UNAVAILABLE")
    else:
        project = project or credential_project

    location = location or "global"
    if not re.fullmatch(r"[a-z][a-z0-9-]{0,62}", location):
        return _google_error(400, "Invalid Vertex location.", "INVALID_ARGUMENT")
    if not project:
        return _google_error(503, "No Vertex project is available.", "FAILED_PRECONDITION")

    hostname = "aiplatform.googleapis.com" if location == "global" else f"{location}-aiplatform.googleapis.com"
    raw_path = request.scope.get("raw_path", request.url.path.encode("utf-8"))
    raw_resource = raw_path.split(b"/publishers/google/", 1)[1]
    prefix = f"/{api_version}/projects/{quote(project, safe='')}/locations/{location}/publishers/google/"
    raw_query = _forward_query(request.scope["query_string"])
    upstream_path = prefix.encode("ascii") + raw_resource
    if raw_query:
        upstream_path += b"?" + raw_query
    url = httpx.URL(scheme="https", host=hostname, raw_path=upstream_path)

    if credentials is not None:
        auth_headers = {}
        try:
            await run_in_threadpool(credentials.before_request, AuthRequest(), request.method, str(url), auth_headers)
        except GoogleAuthError:
            logger.exception("Could not authenticate with the configured Vertex service account")
            return _google_error(503, "Could not authenticate with the Vertex service account.", "UNAUTHENTICATED")
        headers.update(auth_headers)

    body = await request.body()
    if (app_config.VERTEX_TOOL_SCHEMA_COMPAT and request.method == "POST"
            and raw_resource.endswith((b":generateContent", b":streamGenerateContent", b":countTokens"))
            and not request.headers.get("content-encoding")):
        body, changed = rewrite_vertex_tool_schemas(body)
        if changed:
            for name in ("content-md5", "digest"):
                headers.pop(name, None)
            emit_diagnostic("vertex_tool_schema_compat",
                            request_id=getattr(request.state, "stream_diagnostic_id", None), declarations=changed)
    client = _new_http_client()
    try:
        upstream_request = client.build_request(request.method, url, headers=headers, content=body)
        upstream = await client.send(upstream_request, stream=True)
    except httpx.TimeoutException:
        await client.aclose()
        return _google_error(504, "The Vertex upstream connection timed out.", "DEADLINE_EXCEEDED")
    except httpx.HTTPError:
        await client.aclose()
        return _google_error(502, "Could not connect to Vertex AI.", "UNAVAILABLE")
    except BaseException:
        await client.aclose()
        raise

    closed = False

    async def close_upstream():
        nonlocal closed
        with anyio.CancelScope(shield=True):
            if closed:
                return
            closed = True
            try:
                await upstream.aclose()
            finally:
                await client.aclose()

    async def stream_body():
        try:
            async for data in upstream.aiter_raw():
                yield data
        finally:
            await close_upstream()

    response = StreamingResponse(
        stream_body(), status_code=upstream.status_code, background=BackgroundTask(close_upstream),
    )
    response.raw_headers = _forward_headers(upstream.headers.raw)
    return response


async def vertex_resource(
    request: Request,
    api_version: Literal["v1", "v1beta1"],
    project: str,
    location: str,
    resource: str,
    api_key: str = Depends(get_vertex_api_key),
):
    return await _forward_vertex(request, api_version, project, location, api_key)


async def vertex_publisher(
    request: Request,
    api_version: Literal["v1", "v1beta1"],
    resource: str,
    api_key: str = Depends(get_vertex_api_key),
):
    return await _forward_vertex(request, api_version, None, None, api_key)


for path, endpoint, summary in (
    (
        "/{api_version}/projects/{project}/locations/{location}/publishers/google/{resource:path}",
        vertex_resource,
        "Forward a native Vertex model request",
    ),
    (
        "/{api_version}/publishers/google/{resource:path}",
        vertex_publisher,
        "Forward a Vertex request using the configured project and credentials",
    ),
):
    for method in ("GET", "POST", "HEAD"):
        router.add_api_route(
            path, endpoint, methods=[method], summary=summary,
            name=f"{endpoint.__name__}_{method.lower()}", include_in_schema=method != "HEAD",
        )
