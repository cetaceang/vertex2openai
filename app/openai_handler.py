"""
OpenAI handler module for creating clients and processing OpenAI Direct mode responses.
This module encapsulates all OpenAI-specific logic that was previously in chat_api.py.
"""
import json
import time
import httpx
import inspect
from typing import Dict, Any, AsyncGenerator
from uuid import uuid4

from fastapi.responses import JSONResponse, StreamingResponse
import openai

from models import OpenAIRequest
from config import VERTEX_REASONING_TAG
import config as app_config
from api_helpers import (
    create_openai_error_response,
    upstream_error_status,
    openai_fake_stream_generator,
    StreamingReasoningProcessor
)
from message_processing import extract_reasoning_by_tags
from credentials_manager import _refresh_auth
from project_id_discovery import discover_project_id


def _stream_chunk_has_output(payload: Dict[str, Any]) -> bool:
    return any(
        any((choice.get("delta") or {}).get(key) for key in
            ("content", "reasoning_content", "tool_calls", "refusal"))
        for choice in payload.get("choices") or []
    )


# Wrapper classes to mimic OpenAI SDK responses for direct httpx calls
class FakeChatCompletionChunk:
    """A fake ChatCompletionChunk to wrap the dictionary from a direct API stream."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def model_dump(self, exclude_unset=True, exclude_none=True) -> Dict[str, Any]:
        return self._data

class FakeChatCompletion:
    """A fake ChatCompletion to wrap the dictionary from a direct non-streaming API call."""
    def __init__(self, data: Dict[str, Any]):
        self._data = data

    def model_dump(self, exclude_unset=True, exclude_none=True) -> Dict[str, Any]:
        return self._data

class ExpressClientWrapper:
    """
    A wrapper that mimics the openai.AsyncOpenAI client interface but uses direct
    httpx calls for Vertex AI Express Mode. This allows it to be used with the
    existing response handling logic.
    """
    def __init__(self, project_id: str, api_key: str, location: str = "global"):
        self.project_id = project_id
        self.api_key = api_key
        self.location = location
        hostname = "aiplatform.googleapis.com" if self.location == "global" else f"{self.location}-aiplatform.googleapis.com"
        self.base_url = f"https://{hostname}/v1beta1/projects/{self.project_id}/locations/{self.location}/endpoints/openapi"
        
        # The 'chat.completions' structure mimics the real OpenAI client
        self.chat = self
        self.completions = self

    @staticmethod
    def _raise_for_status(response: httpx.Response):
        if not response.is_error:
            response.raise_for_status()
            return
        try:
            body = response.json()
        except ValueError:
            body = None
        error = body.get("error", body) if isinstance(body, dict) else None
        message = error.get("message") if isinstance(error, dict) else None
        # Preserve Google's diagnostic without exposing the key in the URL.
        raise openai.APIStatusError(message or f"Vertex returned HTTP {response.status_code}.",
                                    response=response, body=error)

    async def _stream_generator(self, response: httpx.Response) -> AsyncGenerator[FakeChatCompletionChunk, None]:
        """Decode complete SSE events, including data fields without a space."""
        def decode(data, event_name=""):
            if data.strip() == "[DONE]":
                return None
            try:
                payload = json.loads(data)
            except json.JSONDecodeError as exc:
                raise ValueError("Invalid JSON in the upstream OpenAI SSE stream.") from exc
            if not isinstance(payload, dict):
                raise ValueError("The upstream OpenAI SSE event must be an object.")
            if payload.get("error") or event_name == "error":
                error = payload.get("error") or payload
                message = error.get("message", str(error)) if isinstance(error, dict) else str(error)
                raise openai.APIError(message, request=response.request, body=error)
            return FakeChatCompletionChunk(payload)

        data_lines = []
        event_name = ""
        async for line in response.aiter_lines():
            if line == "":
                if data_lines:
                    event = decode("\n".join(data_lines), event_name)
                    data_lines = []
                    if event is None:
                        return
                    yield event
                event_name = ""
            elif line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
            elif line.startswith("event:"):
                event_name = line[6:].removeprefix(" ")
        if data_lines:
            event = decode("\n".join(data_lines), event_name)
            if event is not None:
                yield event

    async def _streaming_create(self, **kwargs) -> AsyncGenerator[FakeChatCompletionChunk, None]:
        """Handles the creation of a streaming request using httpx."""
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        payload = kwargs.copy()
        if 'extra_body' in payload:
            payload.update(payload.pop('extra_body'))

        client_args = {'timeout': 300}
        if app_config.PROXY_URL:
            client_args['proxy'] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
        async with httpx.AsyncClient(**client_args) as client:
            async with client.stream("POST", endpoint, headers=headers, params=params, json=payload, timeout=None) as response:
                if response.is_error:
                    await response.aread()
                self._raise_for_status(response)
                async for chunk in self._stream_generator(response):
                    yield chunk

    async def create(self, **kwargs) -> Any:
        """
        Mimics the 'create' method of the OpenAI client.
        It builds and sends a direct HTTP request using httpx, delegating
        to the appropriate streaming or non-streaming handler.
        """
        is_streaming = kwargs.get("stream", False)

        if is_streaming:
            return self._streaming_create(**kwargs)
        
        # Non-streaming logic
        endpoint = f"{self.base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        params = {"key": self.api_key}
        
        payload = kwargs.copy()
        if 'extra_body' in payload:
            payload.update(payload.pop('extra_body'))

        client_args = {'timeout': 300}
        if app_config.PROXY_URL:
            client_args['proxy'] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
        async with httpx.AsyncClient(**client_args) as client:
            response = await client.post(endpoint, headers=headers, params=params, json=payload, timeout=None)
            self._raise_for_status(response)
            return FakeChatCompletion(response.json())


class OpenAIDirectHandler:
    """Handles OpenAI Direct mode operations including client creation and response processing."""
    
    def __init__(self, credential_manager=None, express_key_manager=None):
        self.credential_manager = credential_manager
        self.express_key_manager = express_key_manager
        safety_threshold = "BLOCK_NONE"
        self.safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": 'HARM_CATEGORY_CIVIC_INTEGRITY', "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_UNSPECIFIED", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_IMAGE_HATE", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_IMAGE_DANGEROUS_CONTENT", "threshold": safety_threshold},
            {"category": "HARM_CATEGORY_IMAGE_HARASSMENT", "threshold": safety_threshold},
            {"category": 'HARM_CATEGORY_IMAGE_SEXUALLY_EXPLICIT', "threshold": safety_threshold},
            {"category": 'HARM_CATEGORY_JAILBREAK', "threshold": safety_threshold}
        ]

    def create_openai_client(self, project_id: str, gcp_token: str, location: str = "global") -> openai.AsyncOpenAI:
        """Create an OpenAI client configured for Vertex AI endpoint."""
        endpoint_url = (
            f"https://aiplatform.googleapis.com/v1beta1/"
            f"projects/{project_id}/locations/{location}/endpoints/openapi"
        )
        
        client_args = {}
        if app_config.PROXY_URL:
            client_args['proxy'] = app_config.PROXY_URL
        if app_config.SSL_CERT_FILE:
            client_args['verify'] = app_config.SSL_CERT_FILE
        
        http_client = httpx.AsyncClient(**client_args) if client_args else None
        return openai.AsyncOpenAI(
            base_url=endpoint_url,
            api_key=gcp_token,  # OAuth token
            http_client=http_client,
        )
    
    def prepare_openai_params(self, request: OpenAIRequest, model_id: str, is_openai_search: bool = False) -> Dict[str, Any]:
        """
        Prepare parameters for OpenAI API call by converting the request to a dictionary,
        and then overriding the model. This is more robust than manually picking parameters.
        """
        # Convert the request to a dict, excluding unset values. `None` values inside
        # nested models (like messages) are preserved.
        params = request.model_dump(exclude_unset=True)
        
        # Update model and filter out top-level None values.
        params['model'] = model_id
        
        if is_openai_search:
            params['web_search_options'] = {}
            
        openai_params = {k: v for k, v in params.items() if v is not None}
        if "reasoning_effort" in openai_params and openai_params["reasoning_effort"] not in ["low", "medium", "high"]:
            del openai_params["reasoning_effort"]
        return openai_params
    
    
    def prepare_extra_body(self) -> Dict[str, Any]:
        """Prepare extra body parameters for OpenAI API call."""
        return {
            'google': {
                'safety_settings': self.safety_settings,
                'thought_tag_marker': VERTEX_REASONING_TAG,
                "thinking_config": {"include_thoughts": True},
            }
        }
    
    async def handle_streaming_response(
        self,
        openai_client: Any, # Can be openai.AsyncOpenAI or our wrapper
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> StreamingResponse:
        """Handle streaming responses for OpenAI Direct mode."""
        if app_config.FAKE_STREAMING_ENABLED:
            print(f"INFO: OpenAI Fake Streaming (SSE Simulation) ENABLED for model '{request.model}'.")
            return StreamingResponse(
                openai_fake_stream_generator(
                    openai_client=openai_client,
                    openai_params=openai_params,
                    openai_extra_body=openai_extra_body,
                    request_obj=request,
                    is_auto_attempt=False
                ),
                media_type="text/event-stream"
            )
        else:
            print(f"INFO: OpenAI True Streaming ENABLED for model '{request.model}'.")
            stream = self._true_stream_generator(openai_client, openai_params, openai_extra_body, request)
            initial = []
            try:
                async for event in stream:
                    if event == "data: [DONE]\n\n":
                        raise ValueError("The upstream returned an empty OpenAI response stream.")
                    initial.append(event)
                    if _stream_chunk_has_output(json.loads(event.removeprefix("data: "))):
                        break
                else:
                    raise ValueError("The upstream returned an empty OpenAI response stream.")
            except BaseException:
                await stream.aclose()
                raise

            async def body():
                try:
                    for event in initial:
                        yield event
                    async for event in stream:
                        yield event
                finally:
                    await stream.aclose()

            return StreamingResponse(body(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    async def _true_stream_generator(
        self,
        openai_client: Any,
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest,
    ) -> AsyncGenerator[str, None]:
        """Preserve stream identity and each choice's terminal reason."""
        stream_response = None
        has_output = False
        metadata = {}
        processors = {}
        finished = set()
        tool_choices = set()
        roles = set()

        def frame(choices, **extra):
            return f"data: {json.dumps({**metadata, 'choices': choices, **extra})}\n\n"

        try:
            stream_response = await openai_client.chat.completions.create(
                **{**openai_params, "stream": True}, extra_body=openai_extra_body,
            )
            async for chunk in stream_response:
                data = chunk.model_dump(exclude_unset=True, exclude_none=True)
                if data.get("error"):
                    error = data["error"]
                    raise ValueError(error.get("message", str(error)) if isinstance(error, dict) else str(error))
                if not metadata:
                    metadata = {
                        "id": data.get("id") or f"chatcmpl-{uuid4().hex}",
                        "object": "chat.completion.chunk",
                        "created": data.get("created") or int(time.time()),
                        "model": data.get("model") or request.model,
                    }
                choices = []
                for choice in data.get("choices") or []:
                    choice = dict(choice)
                    index = choice.get("index", 0)
                    processor = processors.setdefault(index, StreamingReasoningProcessor(VERTEX_REASONING_TAG))
                    delta = dict(choice.get("delta") or {})
                    if index not in roles:
                        delta.setdefault("role", "assistant")
                        roles.add(index)
                    if delta.get("tool_calls"):
                        tool_choices.add(index)
                    content = delta.get("content")
                    if isinstance(content, str) and content:
                        visible, reasoning = processor.process_chunk(content)
                        delta.pop("content", None)
                        if visible:
                            delta["content"] = visible
                        if reasoning:
                            delta["reasoning_content"] = (delta.get("reasoning_content") or "") + reasoning
                    finish = choice.get("finish_reason")
                    if finish:
                        visible, reasoning = processor.flush_remaining()
                        if visible:
                            delta["content"] = (delta.get("content") or "") + visible
                        if reasoning:
                            delta["reasoning_content"] = (delta.get("reasoning_content") or "") + reasoning
                        if finish == "stop" and index in tool_choices:
                            finish = "tool_calls"
                        finished.add(index)
                    # Some compatible providers put usage on the choice.
                    choice_usage = choice.pop("usage", None)
                    if choice_usage is not None and data.get("usage") is None:
                        data["usage"] = choice_usage
                    choices.append({**choice, "delta": delta, "finish_reason": finish})
                payload = {**data, **metadata, "choices": choices}
                has_output = has_output or _stream_chunk_has_output(payload)
                yield f"data: {json.dumps(payload)}\n\n"

            terminal_choices = []
            for index, processor in processors.items():
                if index in finished:
                    continue
                visible, reasoning = processor.flush_remaining()
                delta = {}
                if visible:
                    delta["content"] = visible
                if reasoning:
                    delta["reasoning_content"] = reasoning
                terminal_choices.append({"index": index, "delta": delta,
                                         "finish_reason": "tool_calls" if index in tool_choices else "stop"})
            has_output = has_output or _stream_chunk_has_output({"choices": terminal_choices})
            if not has_output:
                raise ValueError("The upstream returned no text, reasoning, refusal, or tool calls.")
            if terminal_choices:
                yield frame(terminal_choices)
            yield "data: [DONE]\n\n"
        except Exception as error:
            # Until meaningful output is available, callers can still return an
            # HTTP error. Later failures must travel through the established SSE.
            if not has_output:
                raise
            message = f"Error during OpenAI streaming for {request.model}: {error}"
            print(f"ERROR: {message}")
            yield f"data: {json.dumps(create_openai_error_response(500, message, 'server_error'))}\n\n"
            yield "data: [DONE]\n\n"
        finally:
            if stream_response is not None:
                close = getattr(stream_response, "close", None) or getattr(stream_response, "aclose", None)
                if close is not None:
                    try:
                        result = close()
                        if inspect.isawaitable(result):
                            await result
                    except Exception as error:
                        print(f"WARNING: Could not close the OpenAI response stream: {error}")

    async def handle_non_streaming_response(
        self,
        openai_client: Any, # Can be openai.AsyncOpenAI or our wrapper
        openai_params: Dict[str, Any],
        openai_extra_body: Dict[str, Any],
        request: OpenAIRequest
    ) -> JSONResponse:
        """Handle non-streaming responses for OpenAI Direct mode."""
        try:
            # Ensure stream=False is explicitly passed
            openai_params_non_stream = {**openai_params, "stream": False}
            response = await openai_client.chat.completions.create(
                **openai_params_non_stream,
                extra_body=openai_extra_body
            )
            response_dict = response.model_dump(exclude_unset=True, exclude_none=True)
            
            try:
                choices = response_dict.get('choices')
                if choices and isinstance(choices, list) and len(choices) > 0:
                    message_dict = choices[0].get('message')
                    if message_dict and isinstance(message_dict, dict):
                        # Always remove extra_content from the message if it exists
                        if 'extra_content' in message_dict:
                            del message_dict['extra_content']
                        
                        # Extract reasoning from content
                        full_content = message_dict.get('content')
                        actual_content = full_content if isinstance(full_content, str) else ""
                        
                        if actual_content:
                            print(f"INFO: OpenAI Direct Non-Streaming - Applying tag extraction with fixed marker: '{VERTEX_REASONING_TAG}'")
                            reasoning_text, actual_content = extract_reasoning_by_tags(actual_content, VERTEX_REASONING_TAG)
                            message_dict['content'] = actual_content
                            if reasoning_text:
                                message_dict['reasoning_content'] = reasoning_text
                                # print(f"DEBUG: Tag extraction success. Reasoning len: {len(reasoning_text)}, Content len: {len(actual_content)}")
                            # else:
                            #     print(f"DEBUG: No content found within fixed tag '{VERTEX_REASONING_TAG}'.")
                        else:
                            print(f"WARNING: OpenAI Direct Non-Streaming - No initial content found in message.")
                            message_dict['content'] = ""
                            
            except Exception as e_reasoning:
                print(f"WARNING: Error during non-streaming reasoning processing for model {request.model}: {e_reasoning}")
            
            return JSONResponse(content=response_dict)
            
        except Exception as e:
            error_msg = f"Error calling OpenAI client for {request.model}: {str(e)}"
            print(f"ERROR: {error_msg}")
            status = upstream_error_status(e)
            return JSONResponse(
                status_code=status,
                content=create_openai_error_response(status, error_msg, "server_error")
            )
    
    async def process_request(self, request: OpenAIRequest, base_model_name: str, is_express: bool = False, is_openai_search: bool = False):
        """Main entry point for processing OpenAI Direct mode requests."""
        print(f"INFO: Using OpenAI Direct Path for model: {request.model} (Express: {is_express})")
        
        client: Any = None # Can be openai.AsyncOpenAI or our wrapper

        try:
            if is_express:
                if not self.express_key_manager:
                    raise Exception("Express mode requires an ExpressKeyManager, but it was not provided.")
                
                key_tuple = self.express_key_manager.get_express_api_key()
                if not key_tuple:
                    raise Exception("OpenAI Express Mode requires an API key, but none were available.")
                
                _, express_api_key = key_tuple
                project_id = await discover_project_id(express_api_key)
                
                client = ExpressClientWrapper(
                    project_id=project_id,
                    api_key=express_api_key,
                    location=app_config.VERTEX_EXPRESS_LOCATION,
                )
                print(f"INFO: [OpenAI Express Path] Using ExpressClientWrapper for project: {project_id} (location: {app_config.VERTEX_EXPRESS_LOCATION})")

            else: # Standard SA-based OpenAI SDK Path
                if not self.credential_manager:
                    raise Exception("Standard OpenAI Direct mode requires a CredentialManager.")

                rotated_credentials, rotated_project_id = self.credential_manager.get_credentials()
                if not rotated_credentials or not rotated_project_id:
                    raise Exception("OpenAI Direct Mode requires GCP credentials, but none were available.")

                print(f"INFO: [OpenAI Direct Path] Using credentials for project: {rotated_project_id}")
                gcp_token = _refresh_auth(rotated_credentials)
                if not gcp_token:
                    raise Exception(f"Failed to obtain valid GCP token for OpenAI client (Project: {rotated_project_id}).")
                client = self.create_openai_client(rotated_project_id, gcp_token)

            model_id = f"google/{base_model_name}"
            openai_params = self.prepare_openai_params(request, model_id, is_openai_search)
            openai_extra_body = self.prepare_extra_body()
            
            if request.stream:
                return await self.handle_streaming_response(
                    client, openai_params, openai_extra_body, request
                )
            else:
                return await self.handle_non_streaming_response(
                    client, openai_params, openai_extra_body, request
                )
        except Exception as e:
            error_msg = f"Error in process_request for {request.model}: {e}"
            print(f"ERROR: {error_msg}")
            status = upstream_error_status(e)
            return JSONResponse(status_code=status, content=create_openai_error_response(status, error_msg, "server_error"))
