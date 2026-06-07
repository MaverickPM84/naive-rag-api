from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.db_models import QueryRecord
from app.models.schemas import QueryRecordOut, QueryRequest, QueryResponse
from app.services.rag import RAGPipeline

router = APIRouter(prefix="/query", tags=["query"])


def get_pipeline() -> RAGPipeline:
    from app.main import app
    return app.state.pipeline


@router.post("", response_model=QueryResponse)
def query(
    payload: QueryRequest,
    db: Session = Depends(get_db),
    pipeline: RAGPipeline = Depends(get_pipeline),
):
    """
    Run a RAG query: embed the question, retrieve relevant documents,
    and return a generated answer with its sources.
    """
    if pipeline.vector_store.size == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents have been ingested yet. POST to /documents first.",
        )

    result = pipeline.run(question=payload.question, db=db, top_k=payload.top_k)

    # Persist the query for history
    record = QueryRecord(
        question=payload.question,
        answer=result["answer"],
        doc_ids_used=",".join(str(doc.id) for doc in result["sources"]),
    )
    db.add(record)
    db.commit()

    return QueryResponse(
        question=payload.question,
        answer=result["answer"],
        sources=result["sources"],
    )


@router.get("/history", response_model=List[QueryRecordOut])
def query_history(db: Session = Depends(get_db)):
    """Return the 50 most recent queries, newest first."""
    return (
        db.query(QueryRecord)
        .order_by(QueryRecord.created_at.desc())
        .limit(50)
        .all()
    )
