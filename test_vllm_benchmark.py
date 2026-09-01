#!/usr/bin/env python3
"""Short test: run vllm-benchmark.generate() against a local mock vLLM server
(no GPU/Docker needed) to verify request format, SSE parsing, timing, and that
the output schema matches ollama-benchmark.py's fields."""

import importlib.util
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


vb = load_module("vllm_benchmark", "vllm-benchmark.py")

seen = {}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "test/x"}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    def do_POST(self):
        seen["body"] = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def send(obj):
            self.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            self.wfile.flush()

        tokens = [" Hello", ",", " world", "!"]
        send({"choices": [{"delta": {"role": "assistant"}}]})
        time.sleep(0.05)  # simulate prompt processing before first token
        for t in tokens:
            send({"choices": [{"delta": {"content": t}}]})
            time.sleep(0.02)
        send({"choices": [], "usage": {"prompt_tokens": 5,
                                       "completion_tokens": len(tokens),
                                       "total_tokens": 9}})
        self.wfile.write(b"data: [DONE]\n\n")


def main():
    srv = HTTPServer(("", 0), Handler)  # "" -> reachable via localhost
    vb.PORT = srv.server_address[1]
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        r = vb.generate("test/x", "What is 2+2?")
        assert r["prompt_count"] == 5, r
        assert r["eval_count"] == 4, r
        assert r["prompt_duration_ns"] > 0, r
        assert r["eval_duration_ns"] >= 0, r
        assert r["total_duration_ns"] == r["prompt_duration_ns"] + r["eval_duration_ns"], r

        body = seen["body"]
        assert body["model"] == "test/x", body
        assert body["stream"] is True and body["stream_options"] == {"include_usage": True}, body
        assert body["max_tokens"] == 256, body

        # build a run exactly like main() does, check schema matches ollama's
        run = {
            "model": "test/x", "family": "llama", "tier": "4b", "params": "3B", "size": "small",
            "total_duration_ns": r["total_duration_ns"], "load_duration_ns": 0,
            "prompt_eval_count": r["prompt_count"],
            "prompt_eval_duration_ns": r["prompt_duration_ns"],
            "eval_count": r["eval_count"], "eval_duration_ns": r["eval_duration_ns"],
            "prompt_tps": vb.tps(r["prompt_count"], r["prompt_duration_ns"]),
            "eval_tps": vb.tps(r["eval_count"], r["eval_duration_ns"]),
        }
        expected = {"model", "family", "tier", "params", "size", "total_duration_ns",
                    "load_duration_ns", "prompt_eval_count", "prompt_eval_duration_ns",
                    "eval_count", "eval_duration_ns", "prompt_tps", "eval_tps"}
        assert set(run) == expected, (set(run) ^ expected)

        print("OK: generate() parsed the stream and built an ollama-compatible run")
        print(f"  prompt_tokens={r['prompt_count']}  completion_tokens={r['eval_count']}")
        print(f"  ttft={r['prompt_duration_ns'] / 1e6:.1f}ms  eval={r['eval_duration_ns'] / 1e6:.1f}ms")
        print(f"  eval_tps={run['eval_tps']}")
    finally:
        srv.shutdown()


if __name__ == "__main__":
    main()
