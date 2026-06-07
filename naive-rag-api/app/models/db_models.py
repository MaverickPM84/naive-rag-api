from datetime import datetime

from sqlalchemy import Column, DateTime, Integer, String, Text

from app.core.database import Base


class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    content = Column(Text, nullable=False)
    # Optional: where the document came from (URL, filename, etc.)
    source = Column(String(500), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class QueryRecord(Base):
    __tablename__ = "query_records"

    id = Column(Integer, primary_key=True, index=True)
    question = Column(Text, nullable=False)
    answer = Column(Text, nullable=False)
    # Comma-separated list of document IDs that were retrieved
    doc_ids_used = Column(String(500), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
