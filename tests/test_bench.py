import unittest
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from inferbench import Benchmark, Completion, MockBackend, ChatCompletionsBackend, summarize
from inferbench.bench import percentile


class FakeClock:
    def __init__(self):
        self.now = 0.0
    def __call__(self):
        self.now += 0.1
        return self.now


class TokensBackend:
    name = "measured"
    def generate(self, prompt, model):
        return Completion("response", 3, 8)


class FailureBackend:
    name = "failure"
    def generate(self, prompt, model):
        raise TimeoutError("deliberate test timeout")


class TestBenchmark(unittest.TestCase):
    def test_real_reported_tokens(self):
        report = Benchmark(TokensBackend(), "test", FakeClock()).run(["hello"], 0, 4)
        self.assertEqual(report["successful"], 4)
        self.assertEqual(report["reported_output_tokens"], 32)
        self.assertAlmostEqual(report["latency_p50_ms"], 100)
        self.assertAlmostEqual(report["latency_p95_ms"], 100)
        self.assertAlmostEqual(report["output_tokens_per_second"], 80)

    def test_no_fabricated_tokens_for_mock(self):
        report = Benchmark(MockBackend(), "test", FakeClock()).run(["test"], 0, 2)
        self.assertIsNone(report["reported_output_tokens"])
        self.assertEqual(report["successful"], 2)

    def test_failures_reported(self):
        report = Benchmark(FailureBackend(), "test", FakeClock()).run(["test"], 0, 2)
        self.assertEqual(report["failed"], 2)
        self.assertIsNone(report["latency_p50_ms"])

    def test_percentile(self):
        self.assertEqual(percentile([10, 30, 20, 40], 50), 20)
        self.assertEqual(percentile([10, 30, 20, 40], 95), 40)

    def test_arguments(self):
        with self.assertRaises(ValueError):
            Benchmark(MockBackend(), "test").run([])
        with self.assertRaises(ValueError):
            Benchmark(MockBackend(), "test").run(["x"], repeats=0)
        with self.assertRaises(ValueError):
            ChatCompletionsBackend("file:///etc/passwd")

    def test_local_http_integration(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.server.request_model = body["model"]
                payload = json.dumps({
                    "choices": [{"message": {"content": "pong"}}],
                    "usage": {"prompt_tokens": 2, "completion_tokens": 3}
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        server = HTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            backend = ChatCompletionsBackend(
                f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
            )
            result = backend.generate("ping", "test-model")
            self.assertEqual(result.text, "pong")
            self.assertEqual(result.output_tokens, 3)
            self.assertEqual(server.request_model, "test-model")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
