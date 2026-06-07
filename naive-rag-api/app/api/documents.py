from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import Document
from app.models.schemas import DocumentCreate, DocumentOut
from app.services.embedder import BaseEmbedder
from app.services.vector_store import NaiveVectorStore

router = APIRouter(prefix="/documents", tags=["documents"])


def get_embedder() -> BaseEmbedder:
    """Resolved at request time via app.state (set during startup)."""
    from app.main import app
    return app.state.embedder


def get_vector_store() -> NaiveVectorStore:
    from app.main import app
    return app.state.vector_store


@router.post("", response_model=DocumentOut, status_code=201)
def ingest_document(
    payload: DocumentCreate,
    db: Session = Depends(get_db),
    embedder: BaseEmbedder = Depends(get_embedder),
    vector_store: NaiveVectorStore = Depends(get_vector_store),
):
    """
    Ingest a new document: save to DB and add its embedding to the vector store.
    """
    doc = Document(
        title=payload.title,
        content=payload.content,
        source=payload.source,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    # Embed title + content so both are searchable
    embedding_text = f"{doc.title}\n\n{doc.content}"
    vector = embedder.embed(embedding_text)
    vector_store.add(doc.id, vector)

    return doc


@router.get("", response_model=List[DocumentOut])
def list_documents(db: Session = Depends(get_db)):
    """Return all ingested documents."""
    return db.query(Document).order_by(Document.created_at.desc()).all()


@router.get("/{doc_id}", response_model=DocumentOut)
def get_document(doc_id: int, db: Session = Depends(get_db)):
    """Return a single document by ID."""
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return doc


@router.delete("/{doc_id}", status_code=204)
def delete_document(
    doc_id: int,
    db: Session = Depends(get_db),
):
    """
    Delete a document from the database.

    Known limitation: the document's vector is NOT removed from the vector store.
    This means it may still surface in similarity searches until the store is
    restarted or manually cleaned.  See NaiveVectorStore.remove() for the fix.
    """
    doc = db.get(Document, doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Document not found")
    db.delete(doc)
    db.commit()
