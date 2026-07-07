import os
import unittest
from unittest.mock import patch

from satellite_paper_rag.cli import build_embedder, build_parser
from satellite_paper_rag.embeddings.client import DashScopeEmbeddingClient


class FakeEmbeddingTransport:
    def __init__(self) -> None:
        self.calls = []

    def post_json(self, url, headers, payload, timeout_seconds):
        self.calls.append(
            {
                "url": url,
                "headers": headers,
                "payload": payload,
                "timeout_seconds": timeout_seconds,
            }
        )
        return {
            "data": [
                {"embedding": [0.1, 0.2, 0.3]},
                {"embedding": [0.4, 0.5, 0.6]},
            ]
        }


class EchoEmbeddingTransport:
    def __init__(self) -> None:
        self.batch_sizes = []

    def post_json(self, url, headers, payload, timeout_seconds):
        inputs = payload["input"]
        self.batch_sizes.append(len(inputs))
        return {"data": [{"embedding": [float(index)]} for index, _ in enumerate(inputs)]}


class DashScopeEmbeddingClientTest(unittest.TestCase):
    def test_posts_openai_compatible_embedding_request(self):
        transport = FakeEmbeddingTransport()
        client = DashScopeEmbeddingClient(
            api_key="test-key",
            model_name="text-embedding-v4",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            transport=transport,
        )

        embeddings = client.embed_texts(["cloud threshold", "clear sky"])

        self.assertEqual(embeddings, [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        self.assertEqual(client.dimension, 3)
        self.assertEqual(transport.calls[0]["url"], "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings")
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(transport.calls[0]["payload"]["model"], "text-embedding-v4")
        self.assertEqual(transport.calls[0]["payload"]["input"], ["cloud threshold", "clear sky"])

    def test_reads_dashscope_api_key_from_environment(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}, clear=False):
            client = DashScopeEmbeddingClient()

        self.assertEqual(client.api_key, "env-key")
        self.assertEqual(client.model_name, "text-embedding-v4")

    def test_batches_embedding_requests_to_dashscope_limit(self):
        transport = EchoEmbeddingTransport()
        client = DashScopeEmbeddingClient(api_key="test-key", transport=transport)

        embeddings = client.embed_texts([f"text {index}" for index in range(25)])

        self.assertEqual(len(embeddings), 25)
        self.assertEqual(transport.batch_sizes, [10, 10, 5])

    def test_reports_embedding_batch_progress(self):
        transport = EchoEmbeddingTransport()
        progress_events = []
        client = DashScopeEmbeddingClient(
            api_key="test-key",
            transport=transport,
            progress_callback=lambda current, total, batch_size: progress_events.append((current, total, batch_size)),
        )

        client.embed_texts([f"text {index}" for index in range(25)])

        self.assertEqual(progress_events, [(1, 3, 10), (2, 3, 10), (3, 3, 5)])

    def test_missing_dashscope_api_key_fails_fast(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY"):
                DashScopeEmbeddingClient()


class CliEmbeddingProviderTest(unittest.TestCase):
    def test_build_embedder_can_select_dashscope_provider(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "index-paper",
                "--file",
                "tests/fixtures/sample_sentinel3_paper.md",
                "--embedding-provider",
                "dashscope",
            ]
        )
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}, clear=False):
            embedder = build_embedder(args)

        self.assertIsInstance(embedder, DashScopeEmbeddingClient)
        self.assertEqual(embedder.model_name, "text-embedding-v4")


if __name__ == "__main__":
    unittest.main()
