# Naive RAG API

A minimal **Retrieval-Augmented Generation** backend built with FastAPI.
Ingest text documents, then query them — the app embeds your question, finds the
most relevant documents, and returns an answer with its sources.

Works **fully offline** with no API keys out of the box.

---

## What's inside

```
naive-rag-api/
├── app/
│   ├── main.py                  # FastAPI app + startup/shutdown lifecycle
│   ├── api/
│   │   ├── documents.py         # POST/GET/DELETE /documents
│   │   └── query.py             # POST /query, GET /query/history
│   ├── core/
│   │   ├── config.py            # All settings (reads from .env)
│   │   └── database.py          # SQLAlchemy engine + session
│   ├── models/
│   │   ├── db_models.py         # Document and QueryRecord ORM models
│   │   └── schemas.py           # Pydantic request/response schemas
│   └── services/
│       ├── embedder.py          # FakeEmbedder (default) + OpenAIEmbedder
│       ├── vector_store.py      # Naive in-memory vector store (cosine sim)
│       └── rag.py               # RAG pipeline + answer generators
└── tests/
    ├── conftest.py              # Fixtures: in-memory DB, isolated vector store
    ├── test_documents.py        # Document CRUD tests
    ├── test_query.py            # Query endpoint tests
    └── test_vector_store.py     # Vector store unit tests
```

### How a query flows

```
POST /query
  │
  ├─ 1. Embed the question  (embedder.embed)
  ├─ 2. Find top-K docs     (vector_store.search → cosine similarity)
  ├─ 3. Fetch from DB       (SQLAlchemy → Document rows)
  ├─ 4. Generate answer     (generator.generate → passthrough or LLM)
  └─ 5. Save to history     (QueryRecord insert)
```

---

## Quick start

### 1. Clone and install

```bash
git clone <your-fork-url>   # or copy the folder
cd naive-rag-api

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure (optional)

```bash
cp .env.example .env
# Edit .env if you want to use OpenAI embeddings or the GPT-4o-mini generator.
# The defaults (EMBEDDING_MODEL=fake, LLM_MODEL=none) work with no API key.
```

### 3. Run the server

```bash
uvicorn app.main:app --reload
```

The API is now at **http://localhost:8000**.
Interactive docs: **http://localhost:8000/docs**

### 4. Try it

```bash
# Ingest two documents
curl -s -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "FastAPI basics", "content": "FastAPI is a modern Python web framework for building APIs with type hints."}' | python -m json.tool

curl -s -X POST http://localhost:8000/documents \
  -H "Content-Type: application/json" \
  -d '{"title": "SQLAlchemy ORM", "content": "SQLAlchemy provides a full-featured ORM for interacting with relational databases in Python."}' | python -m json.tool

# Query
curl -s -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How do I build an API in Python?"}' | python -m json.tool

# See query history
curl -s http://localhost:8000/query/history | python -m json.tool

# Health check
curl -s http://localhost:8000/health | python -m json.tool
```

---

## Run the tests

```bash
pytest tests/ -v
```

All 38 tests should pass. Tests use an in-memory database and a temporary vector
store file — they never modify `rag.db` or `vector_store.npy`.

---

## API reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/documents` | Ingest a document (saves to DB + vector store) |
| `GET` | `/documents` | List all documents |
| `GET` | `/documents/{id}` | Get a single document |
| `DELETE` | `/documents/{id}` | Delete a document from the DB |
| `POST` | `/query` | Run a RAG query |
| `GET` | `/query/history` | Last 50 queries |
| `GET` | `/health` | Document + vector count |

Full schema at **http://localhost:8000/docs** when the server is running.

---

## Configuration

All settings live in `.env` (copy from `.env.example`):

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `fake` | `fake` (offline) or `openai` |
| `LLM_MODEL` | `none` | `none` (returns context) or `openai` |
| `OPENAI_API_KEY` | — | Required only when either model is `openai` |
| `TOP_K` | `3` | Documents retrieved per query |
| `DATABASE_URL` | `sqlite:///./rag.db` | SQLAlchemy DB URL |
| `VECTOR_STORE_PATH` | `vector_store.npy` | Where vectors are persisted |

---

## Known limitations

These are intentional — they make good starting points for improvement tasks:

1. **Delete doesn't clean the vector store.** `DELETE /documents/{id}` removes
   the DB row but leaves the vector in `NaiveVectorStore`. The orphan vector
   takes up a retrieval slot on future queries. See `NaiveVectorStore.remove()`
   and the `delete_document` endpoint in `app/api/documents.py`.

2. **No pagination.** `GET /documents` returns every document with no limit or
   offset. With a large corpus this becomes slow and memory-heavy.

3. **In-memory vector store.** Vectors are held in RAM and only written to disk
   on graceful shutdown. A crash or `kill -9` loses recent additions.

4. **Linear search.** Retrieval scans all vectors (`O(n)`). Fine for hundreds of
   docs; impractical at scale.

5. **No duplicate detection.** Ingesting the same document twice creates two DB
   rows and two vectors.

6. **`datetime.utcnow()` deprecation warning.** The ORM models use
   `datetime.utcnow()` which is deprecated in Python 3.12+. Should be
   `datetime.now(timezone.utc)`.
