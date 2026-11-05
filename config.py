from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    google_api_key: str = ""
    llm_model: str = "gemini-2.5-flash"
    embed_model: str = "BAAI/bge-small-en-v1.5"
    data_dir: str = "data"
    collection: str = "company_docs"

    # Email (IMAP)
    imap_host: str = "imap.gmail.com"
    imap_user: str = ""
    imap_password: str = ""

    # RAG tuning
    chunk_size: int = 900
    chunk_overlap: int = 150
    dense_k: int = 12
    bm25_k: int = 12
    final_k: int = 6
    max_retrieval_rounds: int = 3
    rrf_k: int = 60

    @property
    def data_path(self) -> Path:
        p = Path(self.data_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
