from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Protocol


class EmbeddingClient(Protocol):
    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class JsonPostTransport(Protocol):
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        ...


class UrllibJsonPostTransport:
    def post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> dict[str, object]:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url,
            data=body,
            headers={**headers, "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DashScope embedding request failed with HTTP {exc.code}: {error_body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"DashScope embedding request failed: {exc.reason}") from exc
        decoded = json.loads(response_body)
        if not isinstance(decoded, dict):
            raise RuntimeError("DashScope embedding response must be a JSON object.")
        return decoded


class DashScopeEmbeddingClient:
    DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    DEFAULT_MODEL = "text-embedding-v4"

    def __init__(
        self,
        api_key: str | None = None,
        model_name: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 60.0,
        batch_size: int | None = None,
        transport: JsonPostTransport | None = None,
        progress_callback: Callable[[int, int, int], None] | None = None,
    ) -> None:
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("DASHSCOPE_API_KEY is required for DashScope embeddings.")
        self.model_name = model_name or os.getenv("DASHSCOPE_EMBEDDING_MODEL") or self.DEFAULT_MODEL
        self.base_url = (base_url or os.getenv("DASHSCOPE_BASE_URL") or self.DEFAULT_BASE_URL).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.batch_size = batch_size or int(os.getenv("DASHSCOPE_EMBEDDING_BATCH_SIZE", "10"))
        if self.batch_size < 1:
            raise ValueError("DashScope embedding batch_size must be greater than 0.")
        self.transport = transport or UrllibJsonPostTransport()
        self.progress_callback = progress_callback
        self.dimension = int(os.getenv("DASHSCOPE_EMBEDDING_DIMENSION", "0"))

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        embeddings: list[list[float]] = []
        total_batches = math.ceil(len(texts) / self.batch_size)
        for batch_number, batch_start in enumerate(range(0, len(texts), self.batch_size), start=1):
            batch = texts[batch_start : batch_start + self.batch_size]
            if self.progress_callback:
                self.progress_callback(batch_number, total_batches, len(batch))
            payload = {"model": self.model_name, "input": batch}
            response = self.transport.post_json(
                url=f"{self.base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.api_key}"},
                payload=payload,
                timeout_seconds=self.timeout_seconds,
            )
            embeddings.extend(self._parse_embeddings(response, expected_count=len(batch)))
        self.dimension = len(embeddings[0]) if embeddings else self.dimension
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def _parse_embeddings(self, response: dict[str, object], expected_count: int) -> list[list[float]]:
        raw_data = response.get("data")
        if not isinstance(raw_data, list):
            raise RuntimeError("DashScope embedding response is missing a data list.")
        if len(raw_data) != expected_count:
            raise RuntimeError(
                f"DashScope embedding response returned {len(raw_data)} vectors for {expected_count} inputs."
            )
        embeddings: list[list[float]] = []
        for item in raw_data:
            if not isinstance(item, dict) or not isinstance(item.get("embedding"), list):
                raise RuntimeError("DashScope embedding response contains an invalid embedding item.")
            embedding = [float(value) for value in item["embedding"]]
            if embeddings and len(embedding) != len(embeddings[0]):
                raise RuntimeError("DashScope embedding response contains inconsistent vector dimensions.")
            embeddings.append(embedding)
        return embeddings


class DeterministicEmbeddingClient:
    model_name = "deterministic-local-v1"

    def __init__(self, dimension: int = 128) -> None:
        self.dimension = dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = self._tokens(text)
        for token in tokens:
            index = self._bucket(token)
            vector[index] += 1.0
        for first, second in zip(tokens, tokens[1:]):
            index = self._bucket(f"{first}_{second}")
            vector[index] += 0.5
        return self._normalize(vector)

    def _tokens(self, text: str) -> list[str]:
        raw_tokens = re.findall(r"[A-Za-z0-9-]+", text.lower())
        tokens: list[str] = []
        for raw in raw_tokens:
            normalized = self._normalize_token(raw)
            if len(normalized) > 1:
                tokens.append(normalized)
        return tokens

    def _normalize_token(self, token: str) -> str:
        aliases = {
            "cloudy": "cloud",
            "cloud-contaminated": "cloud",
            "cloud-contaminated": "cloud",
            "identified": "classify",
            "identify": "classify",
            "classified": "classify",
            "classification": "classify",
            "detected": "detect",
            "detecting": "detect",
            "pixels": "pixel",
            "thresholds": "threshold",
            "criteria": "criterion",
            "cutoff": "threshold",
            "cutoffs": "threshold",
            "brightness": "bt",
            "temperature": "bt",
        }
        return aliases.get(token, token)

    def _bucket(self, token: str) -> int:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "big") % self.dimension

    def _normalize(self, vector: list[float]) -> list[float]:
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return vector
        return [value / norm for value in vector]


def dot_product(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))
