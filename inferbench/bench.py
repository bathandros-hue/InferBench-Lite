"""Non-streaming endpoint benchmark with explicit reported-vs-missing token counts."""
from __future__ import annotations

import json
import math
import os
import statistics
import time
from dataclasses import dataclass
from typing import Callable, Protocol
from urllib.request import Request, urlopen
from urllib.parse import urlsplit


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class Backend(Protocol):
    name: str
    def generate(self, prompt: str, model: str) -> Completion: ...


class MockBackend:
    name = "mock (not a model performance benchmark)"

    def generate(self, prompt: str, model: str) -> Completion:
        return Completion(f"mock:{prompt[::-1]}")


class ChatCompletionsBackend:
    """HTTP adapter for an OpenAI-compatible /chat/completions endpoint."""
    def __init__(self, url: str, timeout: float = 30.0, api_key: str | None = None):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("A full http(s) endpoint URL is required")
        if timeout <= 0:
            raise ValueError("Timeout must be positive")
        self.url, self.timeout, self.key = url, timeout, api_key
        self.name = "chat-completions"

    def generate(self, prompt: str, model: str) -> Completion:
        body = json.dumps({
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
        }).encode()
        headers = {"Content-Type": "application/json"}
        if self.key:
            headers["Authorization"] = f"Bearer {self.key}"
        request = Request(self.url, body, headers, method="POST")
        with urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read(8_000_000).decode("utf-8"))
        answer = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return Completion(
            str(answer),
            usage.get("prompt_tokens"),
            usage.get("completion_tokens"),
        )


def percentile(values: list[float], q: float) -> float:
    if not values or not 0 <= q <= 100:
        raise ValueError("Nonempty values and a percentile from 0 to 100 are required")
    ordered = sorted(values)
    # Nearest-rank percentile; transparent for small samples.
    return ordered[max(0, math.ceil(q / 100 * len(ordered)) - 1)]


def summarize(samples: list[dict], backend_name: str, warmups: int) -> dict:
    succeeded = [sample for sample in samples if "error" not in sample]
    times = [sample["latency_ms"] for sample in succeeded]
    total_seconds = sum(sample["latency_ms"] for sample in samples) / 1000
    report = {
        "backend": backend_name, "warmups": warmups,
        "requests": len(samples), "successful": len(succeeded),
        "failed": len(samples) - len(succeeded),
        "wall_time_ms": round(sum(s["latency_ms"] for s in samples), 3),
        "latency_p50_ms": round(statistics.median(times), 3) if times else None,
        "latency_p95_ms": round(percentile(times, 95), 3) if times else None,
        "throughput_requests_per_second": round(len(succeeded) / total_seconds, 3)
        if total_seconds > 0 else None,
        "samples": samples,
    }
    if succeeded and all(s.get("output_tokens") is not None for s in succeeded):
        total_tokens = sum(s["output_tokens"] for s in succeeded)
        report["reported_output_tokens"] = total_tokens
        report["output_tokens_per_second"] = (
            round(total_tokens / total_seconds, 3) if total_seconds > 0 else None
        )
    else:
        report["reported_output_tokens"] = None
        report["output_tokens_per_second"] = None
    return report


class Benchmark:
    def __init__(self, backend: Backend, model: str, clock: Callable[[], float] | None = None):
        self.backend, self.model = backend, model
        self.clock = clock or time.perf_counter

    def run(self, prompts: list[str], warmups: int = 1, repeats: int = 3) -> dict:
        if not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError("Provide at least one nonempty prompt")
        if warmups < 0 or repeats < 1:
            raise ValueError("warmups must be >=0 and repeats >=1")
        for i in range(warmups):
            self.backend.generate(prompts[i % len(prompts)], self.model)
        samples = []
        for i in range(repeats):
            prompt = prompts[i % len(prompts)]
            started = self.clock()
            try:
                completion = self.backend.generate(prompt, self.model)
                sample = {
                    "prompt_index": i % len(prompts),
                    "text_chars": len(completion.text),
                    "input_tokens": completion.input_tokens,
                    "output_tokens": completion.output_tokens,
                }
            except Exception as exc:
                # An API can fail by timeout/HTTP error, or with a response schema issue.
                sample = {"prompt_index": i % len(prompts),
                          "error": f"{type(exc).__name__}: {exc}"}
            sample["latency_ms"] = max(0, (self.clock() - started) * 1000)
            samples.append(sample)
        return summarize(samples, self.backend.name, warmups)
