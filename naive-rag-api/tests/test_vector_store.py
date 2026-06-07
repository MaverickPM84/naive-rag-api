"""Tests for NaiveVectorStore."""

import numpy as np
import pytest

from app.services.vector_store import NaiveVectorStore


def make_vec(dim=384, seed=0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store():
    return NaiveVectorStore()


class TestVectorStoreBasics:
    def test_empty_store_size_is_zero(self, store):
        assert store.size == 0

    def test_add_increases_size(self, store):
        store.add(1, make_vec())
        assert store.size == 1

    def test_add_multiple(self, store):
        for i in range(5):
            store.add(i, make_vec(seed=i))
        assert store.size == 5

    def test_add_same_id_replaces(self, store):
        store.add(1, make_vec(seed=0))
        store.add(1, make_vec(seed=1))
        assert store.size == 1

    def test_remove_existing(self, store):
        store.add(1, make_vec())
        assert store.remove(1) is True
        assert store.size == 0

    def test_remove_nonexistent_returns_false(self, store):
        assert store.remove(999) is False

    def test_search_empty_store_returns_empty(self, store):
        results = store.search(make_vec(), top_k=3)
        assert results == []


class TestVectorStoreSearch:
    def test_search_returns_correct_doc_id(self, store):
        vec = make_vec(seed=42)
        store.add(7, vec)
        results = store.search(vec, top_k=1)
        assert len(results) == 1
        assert results[0][0] == 7

    def test_identical_vector_has_score_near_one(self, store):
        vec = make_vec(seed=99)
        store.add(1, vec)
        results = store.search(vec, top_k=1)
        assert pytest.approx(results[0][1], abs=1e-5) == 1.0

    def test_top_k_limits_results(self, store):
        for i in range(10):
            store.add(i, make_vec(seed=i))
        results = store.search(make_vec(seed=0), top_k=3)
        assert len(results) == 3

    def test_top_k_larger_than_store_returns_all(self, store):
        for i in range(4):
            store.add(i, make_vec(seed=i))
        results = store.search(make_vec(seed=0), top_k=10)
        assert len(results) == 4

    def test_results_are_sorted_descending_by_score(self, store):
        for i in range(5):
            store.add(i, make_vec(seed=i))
        results = store.search(make_vec(seed=0), top_k=5)
        scores = [r[1] for r in results]
        assert scores == sorted(scores, reverse=True)


class TestVectorStorePersistence:
    def test_save_and_load_round_trip(self, store, tmp_path):
        vec = make_vec(seed=7)
        store.add(42, vec)

        path = str(tmp_path / "test_store.npy")
        store.save(path)

        new_store = NaiveVectorStore()
        new_store.load(path)

        assert new_store.size == 1
        results = new_store.search(vec, top_k=1)
        assert results[0][0] == 42

    def test_load_missing_file_is_silent(self, store):
        """Loading a non-existent file should not raise."""
        store.load("/tmp/does_not_exist_xyzabc.npy")
        assert store.size == 0
