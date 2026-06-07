"""
Shared pytest fixtures.

Key design decisions:
- Each test gets an in-memory SQLite DB (never touches rag.db).
- Each test gets its own temp vector store path (never touches vector_store.npy).
- app.state is overridden *after* TestClient startup so the lifespan
  doesn't clobber our test doubles.
"""

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.core.database import Base, get_db
from app.main import app
from app.services.embedder import FakeEmbedder
from app.services.rag import PassthroughGenerator, RAGPipeline
from app.services.vector_store import NaiveVectorStore

# ── In-memory database ────────────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite:///:memory:"

test_engine = create_engine(
    TEST_DATABASE_URL,
    connect_args={"check_same_thread": False},
)
TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)


@pytest.fixture(scope="session", autouse=True)
def create_tables():
    Base.metadata.create_all(bind=test_engine)
    yield
    Base.metadata.drop_all(bind=test_engine)


@pytest.fixture
def db_session():
    """Yields a DB session that rolls back after each test."""
    connection = test_engine.connect()
    transaction = connection.begin()
    session = TestSessionLocal(bind=connection)
    yield session
    session.close()
    transaction.rollback()
    connection.close()


# ── Redirect vector store to a temp file per test ────────────────────────────

@pytest.fixture(autouse=True)
def isolated_vector_store_path(tmp_path):
    """
    Prevents tests from reading/writing the real vector_store.npy by
    pointing settings at a fresh temp file for every test.
    """
    from app.core import config
    original = config.settings.vector_store_path
    config.settings.vector_store_path = str(tmp_path / "test_vs.npy")
    yield
    config.settings.vector_store_path = original


# ── App fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def vector_store():
    return NaiveVectorStore()


@pytest.fixture
def embedder():
    return FakeEmbedder()


@pytest.fixture
def pipeline(embedder, vector_store):
    return RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=PassthroughGenerator(),
        top_k=3,
    )


@pytest.fixture
def client(db_session, vector_store, embedder, pipeline):
    """
    TestClient with all services overridden to use test doubles.

    State is applied *after* TestClient.__enter__ so the lifespan startup
    (which sets its own app.state) doesn't overwrite our fixtures.
    """
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app, raise_server_exceptions=True) as c:
        # Override app.state AFTER lifespan startup completes
        app.state.embedder = embedder
        app.state.vector_store = vector_store
        app.state.pipeline = pipeline
        yield c

    app.dependency_overrides.clear()
