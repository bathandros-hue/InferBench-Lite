"""Run a mock demo or explicitly contact a configured model endpoint."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .bench import Benchmark, ChatCompletionsBackend, MockBackend


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure non-streaming LLM API requests")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="Local demo only; no meaningful model timing")
    mode.add_argument("--url", help="OpenAI-compatible chat completion endpoint URL")
    parser.add_argument("--model", default="local-model")
    parser.add_argument("--prompts", type=Path, default=Path("examples/prompts.json"))
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--output", type=Path, help="Optional JSON report file")
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    backend = (
        MockBackend() if args.mock else ChatCompletionsBackend(
            args.url, args.timeout, os.environ.get("INFERBENCH_API_KEY")
        )
    )
    report = Benchmark(backend, args.model).run(prompts, args.warmups, args.repeats)
    rendered = json.dumps(report, indent=2)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
