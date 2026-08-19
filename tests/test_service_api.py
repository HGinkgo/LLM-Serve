import json
import unittest

from fastapi.testclient import TestClient

from llmserve.sampling_params import SamplingParams
from tests.test_service_runtime import (
    FailingStepEngine,
    FakeEngine,
    FakeTokenizer,
    StalledEngine,
)


class APITokenizer(FakeTokenizer):
    def encode(self, text, **kwargs):
        del kwargs
        return [ord(character) for character in text]

    def apply_chat_template(self, messages, **kwargs):
        self.last_messages = messages
        self.last_chat_kwargs = kwargs
        return "chat:" + messages[-1]["content"]


class ServiceAPITests(unittest.TestCase):
    def setUp(self):
        from llmserve.service.api import create_app
        from llmserve.service.runtime import EngineServiceRuntime

        self.engine = FakeEngine()
        self.tokenizer = APITokenizer()
        self.runtime = EngineServiceRuntime(lambda: self.engine, tokenizer=self.tokenizer)
        self.runtime.start()
        self.client = TestClient(create_app(self.runtime, model_name="Qwen3-8B"))

    def tearDown(self):
        self.runtime.close()

    def test_health_endpoints_report_live_and_ready(self):
        self.assertEqual(self.client.get("/health/live").json(), {"status": "live"})
        self.assertEqual(self.client.get("/health/ready").json(), {"status": "ready"})

    def test_models_endpoint_exposes_the_configured_model(self):
        response = self.client.get("/v1/models")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {
            "object": "list",
            "data": [{"id": "Qwen3-8B", "object": "model", "owned_by": "llmserve"}],
        })

    def test_completion_encodes_prompt_and_returns_openai_shape(self):
        response = self.client.post(
            "/v1/completions",
            json={
                "model": "Qwen3-8B",
                "prompt": "hi",
                "max_tokens": 3,
                "temperature": 0.01,
            },
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["object"], "text_completion")
        self.assertEqual(payload["model"], "Qwen3-8B")
        self.assertEqual(payload["choices"][0]["text"], "abc")
        self.assertEqual(payload["choices"][0]["finish_reason"], "stop")
        self.assertEqual(payload["usage"], {
            "prompt_tokens": 2,
            "completion_tokens": 3,
            "total_tokens": 5,
        })

    def test_chat_stream_emits_delta_chunks_and_done_marker(self):
        with self.client.stream(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "Qwen3-8B",
                "messages": [{"role": "user", "content": "hello"}],
                "max_tokens": 2,
                "temperature": 0.01,
                "stream": True,
            },
        ) as response:
            self.assertEqual(response.status_code, 200)
            lines = [line for line in response.iter_lines() if line]

        self.assertEqual(self.tokenizer.last_messages, [{"role": "user", "content": "hello"}])
        self.assertEqual(self.tokenizer.last_chat_kwargs, {
            "tokenize": False,
            "add_generation_prompt": True,
        })
        chunks = [
            json.loads(line.removeprefix("data: "))
            for line in lines[:-1]
        ]
        self.assertEqual(lines[-1], "data: [DONE]")
        self.assertEqual(chunks[0]["object"], "chat.completion.chunk")
        self.assertEqual(chunks[0]["choices"][0]["delta"], {
            "role": "assistant",
            "content": "a",
        })
        self.assertEqual(chunks[1]["choices"][0]["delta"], {"content": "b"})
        self.assertEqual(chunks[-1]["choices"][0]["finish_reason"], "stop")

    def test_metrics_endpoint_exposes_runtime_counters(self):
        response = self.client.get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("llmserve_requests_accepted_total 0", response.text)
        self.assertIn("llmserve_requests_inflight 0", response.text)

    def test_closed_stream_cancels_an_unfinished_engine_request(self):
        from llmserve.service.api import _completion_stream

        request = self.runtime.submit(
            [1, 2], SamplingParams(temperature=0.01, max_tokens=10000)
        )
        stream = _completion_stream(
            self.runtime,
            request,
            "cmpl-test",
            0,
            "Qwen3-8B",
        )

        next(stream)
        stream.close()

        self.assertEqual(self.engine.abort_calls, [request.engine_request_id])

    def test_overload_returns_429_with_retry_after(self):
        from llmserve.service.api import create_app
        from llmserve.service.runtime import EngineServiceRuntime

        engine = StalledEngine()
        runtime = EngineServiceRuntime(
            lambda: engine,
            tokenizer=self.tokenizer,
            max_inflight_requests=1,
        )
        runtime.start()
        self.addCleanup(runtime.close)
        client = TestClient(create_app(runtime, model_name="Qwen3-8B"))
        runtime.submit([1], SamplingParams(temperature=0.01, max_tokens=8))

        response = client.post(
            "/v1/completions",
            json={"model": "Qwen3-8B", "prompt": "blocked"},
        )

        self.assertEqual(response.status_code, 429)
        self.assertEqual(response.headers["retry-after"], "1")
        self.assertEqual(response.json()["detail"]["code"], "overloaded")

    def test_fatal_driver_error_returns_503_with_root_cause(self):
        from llmserve.service.api import create_app
        from llmserve.service.runtime import EngineServiceRuntime

        runtime = EngineServiceRuntime(
            lambda: FailingStepEngine(),
            tokenizer=self.tokenizer,
        )
        runtime.start()
        self.addCleanup(runtime.close)
        request = runtime.submit(
            [1], SamplingParams(temperature=0.01, max_tokens=1)
        )
        next(request.iter_events(timeout=1.0))
        client = TestClient(create_app(runtime, model_name="Qwen3-8B"))

        response = client.post(
            "/v1/completions",
            json={"model": "Qwen3-8B", "prompt": "after failure"},
        )

        self.assertEqual(response.status_code, 503)
        self.assertIn("synthetic engine failure", response.json()["detail"]["message"])


if __name__ == "__main__":
    unittest.main()
