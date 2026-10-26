import os
from pathlib import Path
from typing import Optional

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"

# Ensure .env is always loaded into os.environ
def _load_env_file(path: Path):
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except Exception:
            pass

_load_env_file(ENV_FILE)

try:
    from pydantic_settings import BaseSettings, SettingsConfigDict

    class Settings(BaseSettings):
        BASE_DIR: Path = BASE_DIR
        CHROMA_PERSIST_DIRECTORY: str = str(BASE_DIR / "data" / "chroma")
        UPLOAD_DIR: str = str(BASE_DIR / "data" / "uploads")

        # Embedding settings
        EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
        EMBEDDING_DEVICE: str = "cpu"

        # Vector store & Retrieval
        COLLECTION_NAME: str = "rag_documents"
        CACHE_SIZE: int = 512
        TOP_K: int = 4
        CHUNK_SIZE: int = 1000
        CHUNK_OVERLAP: int = 200

        # LLM Settings (Groq / OpenAI / Ollama)
        GROQ_API_KEY: Optional[str] = None
        OPENAI_API_KEY: Optional[str] = None
        OPENAI_BASE_URL: Optional[str] = None
        LLM_MODEL: str = "openai/gpt-oss-20b"
        LLM_TEMPERATURE: float = 1.0
        LLM_MAX_TOKENS: int = 2048
        LLM_TOP_P: float = 1.0
        REASONING_EFFORT: Optional[str] = "medium"

        model_config = SettingsConfigDict(
            env_file=str(ENV_FILE),
            env_file_encoding="utf-8",
            extra="ignore",
        )

        def get_active_api_key(self) -> Optional[str]:
            return self.GROQ_API_KEY or self.OPENAI_API_KEY

        def is_groq(self) -> bool:
            key = self.get_active_api_key() or ""
            return bool(self.GROQ_API_KEY or key.startswith("gsk_") or "groq" in (self.OPENAI_BASE_URL or "").lower())

    settings = Settings()

except ImportError:
    class FallbackSettings:
        BASE_DIR: Path = BASE_DIR
        CHROMA_PERSIST_DIRECTORY: str = os.getenv(
            "CHROMA_PERSIST_DIRECTORY",
            str(BASE_DIR / "data" / "chroma"),
        )
        UPLOAD_DIR: str = os.getenv(
            "UPLOAD_DIR",
            str(BASE_DIR / "data" / "uploads"),
        )
        EMBEDDING_MODEL_NAME: str = os.getenv(
            "EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"
        )
        EMBEDDING_DEVICE: str = os.getenv("EMBEDDING_DEVICE", "cpu")
        COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "rag_documents")
        CACHE_SIZE: int = int(os.getenv("CACHE_SIZE", "512"))
        TOP_K: int = int(os.getenv("TOP_K", "4"))
        CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
        CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

        GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")
        OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        OPENAI_BASE_URL: Optional[str] = os.getenv("OPENAI_BASE_URL")
        LLM_MODEL: str = os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "1.0"))
        LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
        LLM_TOP_P: float = float(os.getenv("LLM_TOP_P", "1.0"))
        REASONING_EFFORT: Optional[str] = os.getenv("REASONING_EFFORT", "medium")

        def get_active_api_key(self) -> Optional[str]:
            return self.GROQ_API_KEY or self.OPENAI_API_KEY

        def is_groq(self) -> bool:
            key = self.get_active_api_key() or ""
            return bool(self.GROQ_API_KEY or key.startswith("gsk_") or "groq" in (self.OPENAI_BASE_URL or "").lower())

    settings = FallbackSettings()

# Ensure required directories exist
os.makedirs(settings.CHROMA_PERSIST_DIRECTORY, exist_ok=True)
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
