import os
import unittest
from unittest.mock import patch

from satellite_paper_rag.cli import build_chat_client, build_parser
from satellite_paper_rag.llm.client import DashScopeChatClient, parse_json_object


class FakeChatTransport:
    def __init__(self, content='{"rules": []}') -> None:
        self.content = content
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
        return {"choices": [{"message": {"content": self.content}}]}


class DashScopeChatClientTest(unittest.TestCase):
    def test_posts_openai_compatible_chat_completion_request(self):
        transport = FakeChatTransport('{"rules": [{"target_class": "cloud"}]}')
        client = DashScopeChatClient(
            api_key="test-key",
            model_name="qwen-plus",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            transport=transport,
        )

        payload = client.complete_json("system prompt", "user prompt")

        self.assertEqual(payload["rules"][0]["target_class"], "cloud")
        self.assertEqual(transport.calls[0]["url"], "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions")
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], "Bearer test-key")
        self.assertEqual(transport.calls[0]["payload"]["model"], "qwen-plus")
        self.assertEqual(transport.calls[0]["payload"]["temperature"], 0)
        self.assertEqual(
            transport.calls[0]["payload"]["messages"],
            [
                {"role": "system", "content": "system prompt"},
                {"role": "user", "content": "user prompt"},
            ],
        )

    def test_reads_dashscope_api_key_from_environment(self):
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}, clear=False):
            client = DashScopeChatClient()

        self.assertEqual(client.api_key, "env-key")
        self.assertEqual(client.model_name, "qwen-plus")

    def test_missing_dashscope_api_key_fails_fast(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "DASHSCOPE_API_KEY"):
                DashScopeChatClient()

    def test_parse_json_object_accepts_fenced_json(self):
        payload = parse_json_object('```json\n{"answer_type": "rule"}\n```')

        self.assertEqual(payload["answer_type"], "rule")


class CliLlmProviderTest(unittest.TestCase):
    def test_build_chat_client_can_select_dashscope_provider(self):
        parser = build_parser()
        args = parser.parse_args(
            [
                "llm-extract-rules",
                "--file",
                "tests/fixtures/sample_sentinel3_paper.md",
                "--query",
                "What threshold creates a cloud mask?",
                "--llm-provider",
                "dashscope",
            ]
        )
        with patch.dict(os.environ, {"DASHSCOPE_API_KEY": "env-key"}, clear=False):
            chat_client = build_chat_client(args)

        self.assertIsInstance(chat_client, DashScopeChatClient)
        self.assertEqual(chat_client.model_name, "qwen-plus")


if __name__ == "__main__":
    unittest.main()
