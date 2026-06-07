from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Documents ────────────────────────────────────────────────────────────────

class DocumentCreate(BaseModel):
    title: str = Field(..., min_length=1, max_length=255)
    content: str = Field(..., min_length=1)
    source: Optional[str] = Field(None, max_length=500)


class DocumentOut(BaseModel):
    id: int
    title: str
    content: str
    source: Optional[str]
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Queries ───────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1)
    # Override the global TOP_K setting for this request
    top_k: Optional[int] = Field(None, ge=1, le=20)


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[DocumentOut]


class QueryRecordOut(BaseModel):
    id: int
    question: str
    answer: str
    doc_ids_used: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    document_count: int
    vector_count: int
