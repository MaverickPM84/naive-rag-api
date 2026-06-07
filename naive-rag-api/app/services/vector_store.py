"""
Naive in-memory vector store with optional file-based persistence.

Limitations (intentional — good candidates for model evaluation tasks):
- All vectors live in RAM; the store must be re-loaded on startup.
- No support for metadata filtering.
- Linear scan over all vectors (O(n) per query) — fine for small corpora.
- Deleting a document from the database does NOT automatically remove its
  vector here; callers must explicitly call remove().
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class NaiveVectorStore:
    """
    Stores document vectors as a (N, D) matrix alongside a list of doc IDs.
    Retrieval is a brute-force cosine similarity scan.
    """

    def __init__(self) -> None:
        self._doc_ids: List[int] = []
        self._vectors: List[np.ndarray] = []  # each entry is a 1-D unit vector

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(self, doc_id: int, vector: np.ndarray) -> None:
        """Add or replace a document's vector."""
        if doc_id in self._doc_ids:
            idx = self._doc_ids.index(doc_id)
            self._vectors[idx] = vector
        else:
            self._doc_ids.append(doc_id)
            self._vectors.append(vector)

    def remove(self, doc_id: int) -> bool:
        """
        Remove a document's vector. Returns True if it was present.
        Note: the DELETE /documents/{id} endpoint does NOT call this by default.
        """
        if doc_id not in self._doc_ids:
            return False
        idx = self._doc_ids.index(doc_id)
        self._doc_ids.pop(idx)
        self._vectors.pop(idx)
        return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def search(
        self, query_vector: np.ndarray, top_k: int
    ) -> List[Tuple[int, float]]:
        """
        Return the top-k (doc_id, cosine_similarity) pairs, highest score first.
        Returns an empty list if the store is empty.
        """
        if not self._vectors:
            return []

        matrix = np.stack(self._vectors)           # (N, D)
        scores = matrix @ query_vector             # cosine sim (vectors are unit)
        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._doc_ids[i], float(scores[i])) for i in top_indices]

    @property
    def size(self) -> int:
        return len(self._doc_ids)

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Persist the store to disk (numpy binary format)."""
        data = {
            "doc_ids": np.array(self._doc_ids, dtype=np.int64),
            "vectors": np.stack(self._vectors) if self._vectors else np.array([]),
        }
        np.save(path, data, allow_pickle=True)
        logger.info("Vector store saved to %s (%d vectors)", path, self.size)

    def load(self, path: str) -> None:
        """Load a previously saved store from disk. Silently skips missing files."""
        p = Path(path)
        if not p.exists():
            logger.info("No vector store file at %s — starting fresh.", path)
            return
        data = np.load(path, allow_pickle=True).item()
        self._doc_ids = data["doc_ids"].tolist()
        self._vectors = (
            [data["vectors"][i] for i in range(len(self._doc_ids))]
            if len(self._doc_ids) > 0
            else []
        )
        logger.info("Vector store loaded from %s (%d vectors)", path, self.size)
