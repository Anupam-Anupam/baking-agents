#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from dotenv import load_dotenv

import tinker
from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import get_tokenizer


def _clean_text(text: str) -> str:
    cleaned = text
    for marker in ("<|im_end|>", "<|im_start|>"):
        cleaned = cleaned.replace(marker, "")
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>", 1)[1]
    return cleaned.strip()


@dataclass
class ServerState:
    tokenizer_model: str
    renderer_name: str
    default_model: str

    def __post_init__(self) -> None:
        self.service = tinker.ServiceClient()
        self.tokenizer = get_tokenizer(self.tokenizer_model)
        self.renderer = renderers.get_renderer(self.renderer_name, self.tokenizer)
        self._client_cache: dict[str, Any] = {}

    def get_sampling_client(self, model_name: str) -> Any:
        key = model_name.strip() or self.default_model
        if key in self._client_cache:
            return self._client_cache[key]
        if key.startswith("tinker://"):
            client = self.service.create_sampling_client(model_path=key)
        else:
            client = self.service.create_sampling_client(base_model=key)
        self._client_cache[key] = client
        return client

    def list_models(self) -> list[str]:
        cached = sorted(self._client_cache.keys())
        if self.default_model not in cached:
            cached.insert(0, self.default_model)
        return cached


class Handler(BaseHTTPRequestHandler):
    server_version = "TinkerOpenAIServer/1.0"

    @property
    def state(self) -> ServerState:
        return self.server.state  # type: ignore[attr-defined]

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        data = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:  # noqa: N802
        if self.path.endswith("/models"):
            models = [
                {"id": model_id, "object": "model", "owned_by": "tinker"}
                for model_id in self.state.list_models()
            ]
            self._send_json({"object": "list", "data": models})
            return
        self._send_json({"error": {"message": "not_found"}}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if not self.path.endswith("/chat/completions"):
            self._send_json({"error": {"message": "not_found"}}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b"{}"
        try:
            payload = json.loads(body.decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": {"message": "invalid_json"}}, status=400)
            return

        model_name = str(payload.get("model") or self.state.default_model).strip()
        messages = payload.get("messages", [])
        if not isinstance(messages, list):
            self._send_json({"error": {"message": "messages_must_be_list"}}, status=400)
            return

        max_tokens = int(payload.get("max_tokens", 512))
        temperature = float(payload.get("temperature", 0.7))
        top_p = float(payload.get("top_p", 1.0))
        stop_raw = payload.get("stop")
        if isinstance(stop_raw, list):
            stop = [str(item) for item in stop_raw]
        elif isinstance(stop_raw, str):
            stop = [stop_raw]
        else:
            stop = self.state.renderer.get_stop_sequences()

        try:
            mi = self.state.renderer.build_generation_prompt(messages)
            sp = tinker.SamplingParams(
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=stop,
            )
            client = self.state.get_sampling_client(model_name)
            result = client.sample(prompt=mi, num_samples=1, sampling_params=sp).result()
            text = self.state.tokenizer.decode(result.sequences[0].tokens)
            content = _clean_text(text)
            response = {
                "id": "chatcmpl-tinker",
                "object": "chat.completion",
                "created": 0,
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            }
            self._send_json(response)
        except Exception as exc:  # pragma: no cover - runtime server path
            self._send_json({"error": {"message": str(exc)}}, status=500)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="OpenAI-compatible chat server backed by Tinker sampling.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--default-model", default=os.environ.get("POLICY_BASE_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--tokenizer-model", default=os.environ.get("TINKER_TOKENIZER_MODEL", "Qwen/Qwen3-8B"))
    parser.add_argument("--renderer", default=os.environ.get("TINKER_RENDERER", "qwen3_disable_thinking"))
    parser.add_argument("--env-file", default="./vendor/baking_with_tinker/care package/.env")
    args = parser.parse_args()

    load_dotenv(args.env_file, override=False)

    state = ServerState(
        tokenizer_model=args.tokenizer_model,
        renderer_name=args.renderer,
        default_model=args.default_model,
    )
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.state = state  # type: ignore[attr-defined]
    print(f"tinker_openai_server listening on http://{args.host}:{args.port}/v1", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
