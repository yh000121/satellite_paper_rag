from __future__ import annotations

import json
import os
import re
from typing import Any, Protocol

from satellite_paper_rag.embeddings.client import JsonPostTransport, UrllibJsonPostTransport


class ChatCompletionClient(Protocol):
    model_name: str

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        ...


class DashScopeChatClient:
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "qwen-plus"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 90.0,
        transport: JsonPostTransport | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope LLM requests.")
        self.model_name = (
            model_name
            or os.getenv("DASHSCOPE_LLM_MODEL")
            or os.getenv("QWEN_LLM_MODEL")
            or self.DEFAULT_MODEL
        )
        self.base_url = (
            base_url
            or os.getenv("DASHSCOPE_LLM_BASE_URL")
            or os.getenv("DASHSCOPE_BASE_URL")
            or self.DEFAULT_BASE_URL
        ).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.transport = transport or UrllibJsonPostTransport()

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        response = self.transport.post_json(
            url=f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}"},
            payload={
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "response_format": {"type": "json_object"},
            },
            timeout_seconds=self.timeout_seconds,
        )
        return parse_json_object(self._message_content(response))

    def _message_content(self, response: dict[str, object]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError("DashScope LLM response is missing choices.")
        first_choice = choices[0]
        if not isinstance(first_choice, dict):
            raise RuntimeError("DashScope LLM response contains an invalid choice.")
        message = first_choice.get("message")
        if not isinstance(message, dict) or not isinstance(message.get("content"), str):
            raise RuntimeError("DashScope LLM response is missing message content.")
        return message["content"]


def parse_json_object(content: str) -> dict[str, Any]:
    stripped = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", stripped, re.IGNORECASE | re.DOTALL)
    if fenced:
        stripped = fenced.group(1).strip()
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"LLM response was not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("LLM response JSON must be an object.")
    return payload
