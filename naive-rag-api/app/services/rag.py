"""
RAG pipeline: embed query → retrieve docs → generate answer.

Two answer generators are included:
- PassthroughGenerator  (default, no API key needed): formats retrieved context
  as a readable answer.
- OpenAIGenerator       (optional): calls gpt-4o-mini to synthesise an answer.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

from sqlalchemy.orm import Session

from app.models.db_models import Document
from app.services.embedder import BaseEmbedder
from app.services.vector_store import NaiveVectorStore

logger = logging.getLogger(__name__)


# ── Answer generators ─────────────────────────────────────────────────────────

class BaseGenerator(ABC):
    @abstractmethod
    def generate(self, question: str, context_docs: List[Document]) -> str:
        ...


class PassthroughGenerator(BaseGenerator):
    """
    Returns the retrieved document contents as a formatted answer.
    No LLM call required — great for offline testing and evaluation.
    """

    def generate(self, question: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "No relevant documents found."

        parts = [f"**Question:** {question}\n\n**Retrieved context:**\n"]
        for i, doc in enumerate(context_docs, 1):
            parts.append(f"{i}. [{doc.title}]\n{doc.content}")
        return "\n\n".join(parts)


class OpenAIGenerator(BaseGenerator):
    """
    Synthesises an answer using gpt-4o-mini.
    Requires: pip install openai  and  OPENAI_API_KEY set in .env
    """

    MODEL = "gpt-4o-mini"

    def __init__(self, api_key: str):
        try:
            from openai import OpenAI
        except ImportError:
            raise RuntimeError("openai package is not installed. Run: pip install openai")
        self._client = OpenAI(api_key=api_key)

    def generate(self, question: str, context_docs: List[Document]) -> str:
        if not context_docs:
            return "No relevant documents found."

        context = "\n\n---\n\n".join(
            f"Title: {doc.title}\n{doc.content}" for doc in context_docs
        )
        prompt = (
            f"Answer the question using only the provided context.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )
        response = self._client.chat.completions.create(
            model=self.MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        return response.choices[0].message.content.strip()


def get_generator(model: str, openai_api_key: str | None = None) -> BaseGenerator:
    if model == "openai":
        if not openai_api_key:
            raise ValueError("OPENAI_API_KEY must be set when LLM_MODEL=openai")
        return OpenAIGenerator(api_key=openai_api_key)
    return PassthroughGenerator()


# ── Pipeline ──────────────────────────────────────────────────────────────────

class RAGPipeline:
    """
    Orchestrates the full RAG loop:
      1. Embed the query
      2. Retrieve the top-k matching document IDs from the vector store
      3. Fetch the full document rows from the database
      4. Generate an answer using the configured generator
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: NaiveVectorStore,
        generator: BaseGenerator,
        top_k: int = 3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.top_k = top_k

    def run(self, question: str, db: Session, top_k: int | None = None) -> dict:
        """
        Run the pipeline for a question.

        Returns a dict with keys:
          - answer  (str)
          - sources (List[Document])  — docs actually found in the DB
        """
        k = top_k or self.top_k

        # 1. Embed
        query_vector = self.embedder.embed(question)

        # 2. Retrieve candidate doc IDs
        candidates = self.vector_store.search(query_vector, top_k=k)
        logger.debug("Vector search returned %d candidates: %s", len(candidates), candidates)

        # 3. Fetch documents from DB (some IDs may be missing if docs were deleted
        #    but their vectors were not removed from the store)
        sources: List[Document] = []
        for doc_id, score in candidates:
            doc = db.get(Document, doc_id)
            if doc is not None:
                sources.append(doc)
            else:
                logger.warning(
                    "Doc ID %d is in the vector store but missing from the DB "
                    "(it may have been deleted without removing its vector).",
                    doc_id,
                )

        # 4. Generate
        answer = self.generator.generate(question, sources)
        return {"answer": answer, "sources": sources}
