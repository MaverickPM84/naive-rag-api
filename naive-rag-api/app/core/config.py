from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    # Embedder: "fake" (default, no API key) or "openai"
    embedding_model: str = "fake"

    # LLM for answer generation: "none" (returns context) or "openai"
    llm_model: str = "none"

    # OpenAI key — only required when embedding_model or llm_model is "openai"
    openai_api_key: Optional[str] = None

    # How many documents to retrieve per query
    top_k: int = 3

    # SQLite database file
    database_url: str = "sqlite:///./rag.db"

    # Where the vector store is persisted on disk
    vector_store_path: str = "vector_store.npy"

    model_config = {"env_file": ".env"}


settings = Settings()
