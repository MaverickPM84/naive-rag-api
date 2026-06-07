"""
Embedder implementations.

The app uses FakeEmbedder by default so it works completely offline.
Switch to OpenAIEmbedder by setting EMBEDDING_MODEL=openai in your .env.
"""

import hashlib
from abc import ABC, abstractmethod

import numpy as np


class BaseEmbedder(ABC):
    """All embedders must return a normalised float32 numpy vector."""

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        ...


class FakeEmbedder(BaseEmbedder):
    """
    Deterministic local embedder — no API key required.

    Generates a consistent vector for any text using its MD5 hash as an RNG
    seed.  Same text always → same vector.  Semantically meaningless (similar
    texts will NOT have similar vectors), but good enough to exercise the full
    RAG pipeline without any external dependencies.
    """

    DIMENSION = 384

    def embed(self, text: str) -> np.ndarray:
        seed = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16) % (2**32)
        rng = np.random.RandomState(seed)
        vec = rng.randn(self.DIMENSION).astype(np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm  # return a unit vector

    @property
    def dimension(self) -> int:
        return self.DIMENSION


class OpenAIEmbedder(BaseEmbedder):
    """
    Embedder backed by OpenAI's text-embedding-3-small model.
    Requires: pip install openai  and  OPENAI_API_KEY set in .env
    """

    MODEL = "text-embedding-3-small"
    DIMENSION = 1536

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError(
                "openai package is not installed. "
                "Run: pip install openai"
            )
        self._client = OpenAI(api_key=api_key)

    def embed(self, text: str) -> np.ndarray:
        # Strip newlines — recommended by OpenAI docs
        clean = text.replace("\n", " ")
        response = self._client.embeddings.create(input=[clean], model=self.MODEL)
        vec = np.array(response.data[0].embedding, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return vec / norm

    @property
    def dimension(self) -> int:
        return self.DIMENSION


def get_embedder(model: str, openai_api_key: str | None = None) -> BaseEmbedder:
    """Factory — returns the right embedder based on configuration."""
    if model == "openai":
        if not openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY must be set when EMBEDDING_MODEL=openai"
            )
        return OpenAIEmbedder(api_key=openai_api_key)
    return FakeEmbedder()
