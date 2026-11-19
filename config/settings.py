from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    gemini_api_key: str = ""
    groq_api_key: str = ""

    # Neo4j Knowledge Graph
    neo4j_uri: str = "neo4j://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j123"

    # Provider Mode: "auto" (hybrid), "groq" (Groq only), or "gemini" (Gemini primary)
    llm_provider: str = "groq"

    # Model Configuration (Cost-effective free/low-cost defaults)
    gemini_model: str = "gemini-2.5-flash"
    groq_model: str = "openai/gpt-oss-20b"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Agent Loop Guardrails
    max_research_iterations: int = 3
    llm_timeout_seconds: int = 30
    llm_max_retries: int = 2

    # Paths
    base_dir: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = base_dir / "data"
    storage_dir: Path = base_dir / "storage"
    faiss_index_dir: Path = storage_dir / "research_papers_faiss"
    sqlite_db_path: Path = storage_dir / "checkpoints.db"


settings = Settings()
