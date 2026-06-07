import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import Base, SessionLocal, engine
from app.models.db_models import Document  # noqa: F401 – ensures table is registered
from app.models.schemas import HealthResponse
from app.services.embedder import get_embedder
from app.services.rag import RAGPipeline, get_generator
from app.services.vector_store import NaiveVectorStore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup / shutdown.

    On startup:
      1. Create database tables.
      2. Initialise the embedder and vector store.
      3. Load persisted vectors from disk (if any).
      4. Re-embed any documents in the DB whose vectors are missing from the
         store (e.g. after a fresh start with an existing database).
    """
    # 1. Create tables
    Base.metadata.create_all(bind=engine)

    # 2. Build services
    embedder = get_embedder(settings.embedding_model, settings.openai_api_key)
    generator = get_generator(settings.llm_model, settings.openai_api_key)
    vector_store = NaiveVectorStore()

    # 3. Load persisted vectors
    vector_store.load(settings.vector_store_path)

    # 4. Re-embed any DB documents not already in the store
    db: Session = SessionLocal()
    try:
        docs = db.query(Document).all()
        missing = [d for d in docs if d.id not in vector_store._doc_ids]
        if missing:
            logger.info("Re-embedding %d document(s) missing from the vector store.", len(missing))
            for doc in missing:
                text = f"{doc.title}\n\n{doc.content}"
                vector_store.add(doc.id, embedder.embed(text))
    finally:
        db.close()

    # 5. Wire up the RAG pipeline
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        top_k=settings.top_k,
    )

    # Attach to app.state so routers can access them
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.pipeline = pipeline

    logger.info(
        "Startup complete — embedder=%s, llm=%s, docs_in_store=%d",
        settings.embedding_model,
        settings.llm_model,
        vector_store.size,
    )

    yield  # Application is running

    # Shutdown: persist vectors so they survive restarts
    vector_store.save(settings.vector_store_path)
    logger.info("Vector store saved. Shutting down.")


app = FastAPI(
    title="Naive RAG API",
    description=(
        "A minimal Retrieval-Augmented Generation API built with FastAPI. "
        "Ingest documents, then query them. Works fully offline using the fake embedder."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# ── Routers ───────────────────────────────────────────────────────────────────

from app.api import documents, query  # noqa: E402 – imported after app is defined

app.include_router(documents.router)
app.include_router(query.router)


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["health"])
def health():
    """Quick status check — how many docs are in the DB and the vector store."""
    db: Session = SessionLocal()
    try:
        doc_count = db.query(Document).count()
    finally:
        db.close()

    return HealthResponse(
        status="ok",
        document_count=doc_count,
        vector_count=app.state.vector_store.size,
    )
