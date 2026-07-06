from __future__ import annotations

import hashlib
import math
import re
from typing import Protocol


class EmbeddingClient(Protocol):
    model_name: str
    dimension: int

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


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
