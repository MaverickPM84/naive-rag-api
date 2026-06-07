"""Tests for the /documents endpoints."""


def ingest(client, title="Test Doc", content="Some content about Python.", source=None):
    """Helper to ingest a single document and assert success."""
    payload = {"title": title, "content": content}
    if source:
        payload["source"] = source
    resp = client.post("/documents", json=payload)
    assert resp.status_code == 201, resp.text
    return resp.json()


# ── POST /documents ───────────────────────────────────────────────────────────

class TestIngestDocument:
    def test_ingest_returns_201_with_id(self, client):
        data = ingest(client)
        assert data["id"] is not None
        assert data["title"] == "Test Doc"

    def test_ingest_with_optional_source(self, client):
        data = ingest(client, source="https://example.com/doc")
        assert data["source"] == "https://example.com/doc"

    def test_ingest_without_source(self, client):
        data = ingest(client)
        assert data["source"] is None

    def test_ingest_adds_to_vector_store(self, client, vector_store):
        assert vector_store.size == 0
        ingest(client)
        assert vector_store.size == 1

    def test_ingest_empty_title_returns_422(self, client):
        resp = client.post("/documents", json={"title": "", "content": "some content"})
        assert resp.status_code == 422

    def test_ingest_empty_content_returns_422(self, client):
        resp = client.post("/documents", json={"title": "Title", "content": ""})
        assert resp.status_code == 422

    def test_ingest_missing_fields_returns_422(self, client):
        resp = client.post("/documents", json={"title": "Only a title"})
        assert resp.status_code == 422


# ── GET /documents ────────────────────────────────────────────────────────────

class TestListDocuments:
    def test_empty_list(self, client):
        resp = client.get("/documents")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_lists_all_documents(self, client):
        ingest(client, title="Doc A")
        ingest(client, title="Doc B")
        data = client.get("/documents").json()
        titles = {d["title"] for d in data}
        assert "Doc A" in titles
        assert "Doc B" in titles


# ── GET /documents/{id} ───────────────────────────────────────────────────────

class TestGetDocument:
    def test_get_existing(self, client):
        created = ingest(client, title="Specific Doc")
        resp = client.get(f"/documents/{created['id']}")
        assert resp.status_code == 200
        assert resp.json()["title"] == "Specific Doc"

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/documents/99999")
        assert resp.status_code == 404


# ── DELETE /documents/{id} ────────────────────────────────────────────────────

class TestDeleteDocument:
    def test_delete_returns_204(self, client):
        doc = ingest(client)
        resp = client.delete(f"/documents/{doc['id']}")
        assert resp.status_code == 204

    def test_deleted_document_not_in_list(self, client):
        doc = ingest(client)
        client.delete(f"/documents/{doc['id']}")
        ids = [d["id"] for d in client.get("/documents").json()]
        assert doc["id"] not in ids

    def test_delete_nonexistent_returns_404(self, client):
        resp = client.delete("/documents/99999")
        assert resp.status_code == 404

    def test_delete_does_not_remove_from_vector_store(self, client, vector_store):
        """
        Known limitation: deleting a doc from the DB leaves its vector in the store.
        This test documents the current (imperfect) behaviour.
        """
        doc = ingest(client)
        assert vector_store.size == 1
        client.delete(f"/documents/{doc['id']}")
        # Vector is still there even though the DB row is gone
        assert vector_store.size == 1
