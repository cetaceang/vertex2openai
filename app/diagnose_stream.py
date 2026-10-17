"""Probe OpenAI SSE without printing prompts, generated text, arguments or keys.

Inside the container:
    python diagnose_stream.py --model '[EXPRESS] gemini-3.7-flash'
    python diagnose_stream.py --model '[EXPRESS] gemini-3.7-flash' --tools
"""

import argparse
import asyncio
import json
import os

import httpx

from stream_diagnostics import SSEInspector


async def probe_stream(base_url, model, api_key="", *, tools=False, timeout=120, transport=None):
    payload = {
        "model": model, "stream": True, "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": "Reply with exactly OK."}],
    }
    if tools:
        payload["messages"][0]["content"] = "Call stream_probe with message OK."
        payload["tools"] = [{"type": "function", "function": {
            "name": "stream_probe", "description": "Return a short test message.",
            "parameters": {"type": "object", "properties": {"message": {"type": "string"}},
                           "required": ["message"]},
        }}]
        payload["tool_choice"] = {"type": "function", "function": {"name": "stream_probe"}}
    headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    inspector = SSEInspector()
    result = {"test": "tool_call" if tools else "text", "status": None, "content_type": "",
              "body_bytes": 0, "connection_error": None}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=15), transport=transport) as client:
            async with client.stream("POST", base_url.rstrip("/") + "/chat/completions",
                                     json=payload, headers=headers) as response:
                result["status"] = response.status_code
                result["content_type"] = response.headers.get("content-type", "")
                if response.status_code == 200 and "text/event-stream" in result["content_type"]:
                    async for data in response.aiter_bytes():
                        result["body_bytes"] += len(data)
                        inspector.feed(data)
                # Do not echo arbitrary HTTP error bodies (which may contain credentials).
    except httpx.HTTPError as error:
        result["connection_error"] = type(error).__name__
    result.update(inspector.summary())
    result["ok"] = bool(
        result["status"] == 200 and not result["connection_error"] and inspector.done
        and inspector.finish_reasons and not inspector.errors and not inspector.invalid_events
        and not result["pending_event"]
        and (inspector.tool_calls if tools else inspector.text_chars)
    )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:7860/v1")
    parser.add_argument("--api-key-env", default=None,
                        help="Environment variable containing the endpoint key. Defaults to API_KEY, then the first VERTEX_EXPRESS_API_KEY.")
    parser.add_argument("--tools", action="store_true", help="Test a forced tool call instead of plain text.")
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    if args.api_key_env:
        api_key = os.environ.get(args.api_key_env, "")
        if not api_key:
            parser.error(f"Environment variable {args.api_key_env} is empty or unset.")
    else:
        api_key = os.environ.get("API_KEY") or os.environ.get("VERTEX_EXPRESS_API_KEY", "").split(",")[0].strip()
    result = asyncio.run(probe_stream(args.base_url, args.model, api_key, tools=args.tools, timeout=args.timeout))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
