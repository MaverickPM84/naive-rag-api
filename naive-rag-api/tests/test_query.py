"""Tests for the /query endpoints."""
import pytest


def ingest(client, title="Doc", content="Python is a programming language."):
    resp = client.post("/documents", json={"title": title, "content": content})
    assert resp.status_code == 201
    return resp.json()


# ── POST /query ───────────────────────────────────────────────────────────────

class TestQuery:
    def test_query_with_no_docs_returns_400(self, client):
        resp = client.post("/query", json={"question": "What is Python?"})
        assert resp.status_code == 400

    def test_query_returns_answer_and_sources(self, client):
        ingest(client)
        resp = client.post("/query", json={"question": "What is Python?"})
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert "sources" in data
        assert len(data["sources"]) >= 1

    def test_query_response_has_question_field(self, client):
        ingest(client)
        resp = client.post("/query", json={"question": "Tell me about Python."})
        assert resp.json()["question"] == "Tell me about Python."

    def test_query_top_k_limits_sources(self, client):
        for i in range(5):
            ingest(client, title=f"Doc {i}", content=f"Content number {i}.")
        resp = client.post("/query", json={"question": "content", "top_k": 2})
        assert resp.status_code == 200
        assert len(resp.json()["sources"]) <= 2

    def test_query_empty_question_returns_422(self, client):
        resp = client.post("/query", json={"question": ""})
        assert resp.status_code == 422

    def test_query_is_saved_to_history(self, client):
        ingest(client)
        client.post("/query", json={"question": "What is Python?"})
        history = client.get("/query/history").json()
        questions = [r["question"] for r in history]
        assert "What is Python?" in questions


# ── GET /query/history ────────────────────────────────────────────────────────

class TestQueryHistory:
    def test_history_empty_by_default(self, client):
        resp = client.get("/query/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_history_records_multiple_queries(self, client):
        ingest(client)
        client.post("/query", json={"question": "First question"})
        client.post("/query", json={"question": "Second question"})
        history = client.get("/query/history").json()
        assert len(history) == 2

    def test_history_returns_newest_first(self, client):
        ingest(client)
        client.post("/query", json={"question": "Older query"})
        client.post("/query", json={"question": "Newer query"})
        history = client.get("/query/history").json()
        assert history[0]["question"] == "Newer query"
