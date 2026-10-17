"""Payload-free diagnostics at the ASGI boundary and for the stream probe."""

import codecs
import json
import logging
import re
import time
from urllib.parse import quote, unquote_plus
from uuid import uuid4


_CREDENTIAL_QUERY_NAMES = {"key", "access_token", "oauth_token", "api_key", "x-goog-api-key", "x-ip-token"}
_QUERY_PARAMETER = re.compile(r"([?&])([^=&#\s]+)=([^&#\s]*)")


def redact_query_credentials(value):
    def replace(match):
        if unquote_plus(match[2]).lower() in _CREDENTIAL_QUERY_NAMES:
            return f"{match[1]}{match[2]}=[REDACTED]"
        return match[0]
    return _QUERY_PARAMETER.sub(replace, value)


def redact_error_message(message, secrets=()):
    for secret in sorted({value for value in secrets if isinstance(value, str) and value}, key=len, reverse=True):
        message = message.replace(secret, "[REDACTED]").replace(quote(secret, safe=""), "[REDACTED]")
    message = re.sub(r"(?i)\bBearer\s+[^\s\"<>]+", "Bearer [REDACTED]", message)
    return redact_query_credentials(message)[:1536]


class CredentialQueryFilter(logging.Filter):
    def filter(self, record):
        # Uvicorn access records use (client, method, path, version, status).
        if isinstance(record.args, tuple) and len(record.args) == 5 and isinstance(record.args[2], str):
            args = list(record.args)
            args[2] = redact_query_credentials(args[2])
            record.args = tuple(args)
        return True


def install_access_log_redaction():
    logger = logging.getLogger("uvicorn.access")
    if not any(isinstance(item, CredentialQueryFilter) for item in logger.filters):
        logger.addFilter(CredentialQueryFilter())


class SSEInspector:
    def __init__(self, native=False):
        self.native = native
        self.events = 0
        self.text_chars = 0
        self.reasoning_chars = 0
        self.tool_calls = set()
        self.native_tool_calls = 0
        self.finish_reasons = set()
        self.done = False
        self.errors = 0
        self.invalid_events = 0
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._buffer = ""
        self._data = []
        self._event_name = ""

    def feed(self, data):
        self._buffer += self._decoder.decode(data)
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            self._line(line.removesuffix("\r"))

    def _line(self, line):
        if not line:
            if self._data:
                self._event("\n".join(self._data))
                self._data = []
            self._event_name = ""
        elif line.startswith("data:"):
            self._data.append(line[5:].removeprefix(" "))
        elif line.startswith("event:"):
            self._event_name = line[6:].removeprefix(" ")

    def _event(self, data):
        self.events += 1
        if data.strip() == "[DONE]":
            self.done = True
            return
        try:
            payload = json.loads(data)
            if not isinstance(payload, dict):
                raise ValueError("SSE payload is not an object")
            if payload.get("error") or self._event_name == "error":
                self.errors += 1
            if self.native:
                for candidate in payload.get("candidates") or []:
                    finish = candidate.get("finishReason") or candidate.get("finish_reason")
                    if finish:
                        self.finish_reasons.add(str(finish))
                    for part in (candidate.get("content") or {}).get("parts") or []:
                        text = part.get("text") or ""
                        if part.get("thought"):
                            self.reasoning_chars += len(text)
                        else:
                            self.text_chars += len(text)
                        if part.get("functionCall") or part.get("function_call"):
                            self.native_tool_calls += 1
            else:
                for choice in payload.get("choices") or []:
                    if choice.get("finish_reason"):
                        self.finish_reasons.add(str(choice["finish_reason"]))
                    delta = choice.get("delta") or {}
                    self.text_chars += len(delta.get("content") or "")
                    self.reasoning_chars += len(delta.get("reasoning_content") or delta.get("reasoning") or "")
                    for call in delta.get("tool_calls") or []:
                        self.tool_calls.add((choice.get("index", 0), call.get("index", 0)))
        except (ValueError, TypeError, AttributeError, KeyError):
            self.invalid_events += 1

    def summary(self):
        return {
            "sse_events": self.events,
            "text_chars": self.text_chars,
            "reasoning_chars": self.reasoning_chars,
            "tool_calls": len(self.tool_calls) + self.native_tool_calls,
            "finish_reasons": sorted(self.finish_reasons),
            "done": self.done,
            "sse_errors": self.errors,
            "invalid_events": self.invalid_events,
            "pending_event": bool(self._buffer or self._data),
        }


def emit_diagnostic(event, **details):
    print("INFO: stream_trace " + json.dumps({"event": event, **details}, ensure_ascii=True), flush=True)


class StreamDiagnosticsMiddleware:
    """Observe accepted response bytes without changing headers or SSE data.

    A completed ASGI send is evidence at this server, not an acknowledgement
    from the final client. No prompts, generated text, tool arguments or keys
    are included in these logs.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        native = "/publishers/google/" in path
        if scope["type"] != "http" or (path != "/v1/chat/completions" and not native):
            return await self.app(scope, receive, send)

        request_id = uuid4().hex[:12]
        scope.setdefault("state", {})["stream_diagnostic_id"] = request_id
        started = time.monotonic()
        inspector = SSEInspector(native=native)
        status = None
        is_sse = False
        complete = False
        disconnected = False
        error_type = None
        body_bytes = 0
        first_output = False
        emit_diagnostic("request", request_id=request_id, protocol="vertex" if native else "openai", version=1)

        async def observed_receive():
            nonlocal disconnected
            message = await receive()
            if message["type"] == "http.disconnect":
                disconnected = True
            return message

        async def observed_send(message):
            nonlocal status, is_sse, complete, body_bytes, first_output
            await send(message)
            if message["type"] == "http.response.start":
                status = message["status"]
                headers = dict(message.get("headers", []))
                content_type = headers.get(b"content-type", b"").decode("latin-1")
                # Native passthrough may contain compressed bytes; leave them alone.
                is_sse = "text/event-stream" in content_type and not headers.get(b"content-encoding")
                emit_diagnostic("response", request_id=request_id, status=status, content_type=content_type,
                                elapsed_ms=round((time.monotonic() - started) * 1000))
            elif message["type"] == "http.response.body":
                body = message.get("body", b"")
                body_bytes += len(body)
                if is_sse:
                    inspector.feed(body)
                    if not first_output and (inspector.text_chars or inspector.reasoning_chars or
                                             inspector.tool_calls or inspector.native_tool_calls):
                        first_output = True
                        emit_diagnostic("first_output", request_id=request_id, **inspector.summary(),
                                        elapsed_ms=round((time.monotonic() - started) * 1000))
                complete = not message.get("more_body", False)

        try:
            await self.app(scope, observed_receive, observed_send)
        except BaseException as error:
            error_type = type(error).__name__
            raise
        finally:
            if complete:
                outcome = "upstream_error" if inspector.errors else "complete"
                if is_sse and not native and (not inspector.done or not inspector.finish_reasons
                                              or inspector.invalid_events or inspector.summary()["pending_event"]):
                    outcome = "upstream_error" if inspector.errors else "incomplete_sse"
                if status is not None and status >= 400:
                    outcome = "http_error"
            else:
                outcome = "downstream_disconnected" if disconnected else "interrupted"
            emit_diagnostic("end", request_id=request_id, outcome=outcome, status=status,
                            body_bytes=body_bytes, response_complete=complete, error_type=error_type,
                            elapsed_ms=round((time.monotonic() - started) * 1000), **inspector.summary())
