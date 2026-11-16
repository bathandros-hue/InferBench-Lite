# InferBench Lite

A lightweight measurement harness for **non-streaming** text-generation APIs. Records request latency, p50/p95, successes/failures, total throughput, and token throughput only when the endpoint reports token counts.

## Quickstart

Python 3.10+; no runtime dependencies:

    python -m unittest discover -s tests -v
    python -m inferbench.cli --mock --prompts examples/prompts.json --repeats 5

The mock mode tests the harness only. Its latency is **not** a meaningful model performance benchmark.

To test an OpenAI-compatible server you operate locally:

    python -m inferbench.cli --url http://localhost:11434/v1/chat/completions --model YOUR_MODEL --prompts examples/prompts.json --repeats 5 --output report.json

Optional endpoint credentials: INFERBENCH_API_KEY environment variable.

## Measurement methodology

- Non-streaming POST chat-completion calls with fixed temperature 0.
- Warmup requests are excluded from measured trials.
- Sequential requests; no concurrency and no time-to-first-token claim.
- p95 uses nearest-rank percentiles; report includes individual samples.
- Output tokens/sec only if every successful response includes completion_tokens.
- Request throughput includes the measured time of failed calls.

## Limitations

Endpoint compatibility varies. Latencies include HTTP overhead and network effects. This is not a model-quality evaluation and does not compare GPUs or deployments unless you control conditions. Verify remote endpoint costs and privacy before sending prompts. No network requests are made by the mock mode.

## GitHub topics

llm, benchmarking, inference, latency, performance-testing, python

## License

MIT; see LICENSE.
