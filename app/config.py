import os
from pathlib import Path
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
ENV_FILE = BASE_DIR / ".env"


class Settings(BaseSettings):
    # App settings
    APP_NAME: str = "Resume Parser"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False

    # File upload settings
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: set = {".pdf", ".docx", ".doc", ".txt", ".png", ".jpg", ".jpeg"}

    # Redis/Celery settings
    REDIS_URL: str = "redis://localhost:6379/0"
    CELERY_BROKER_URL: str = "redis://localhost:6379/0"
    CELERY_RESULT_BACKEND: str = "redis://localhost:6379/0"
    CELERY_QUEUE: str = "resume_parser_v2"

    # Redis settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0

    # PostgreSQL settings
    DATABASE_URL: str = "postgresql+asyncpg://postgres:postgres@localhost/smartrecruitz"
    GEMINI_API_KEY: Optional[str] = None

    # Logging
    LOG_LEVEL: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()

# Keep broker/backend aligned with REDIS_URL if not explicitly set in env.
if settings.CELERY_BROKER_URL == "redis://localhost:6379/0":
    settings.CELERY_BROKER_URL = settings.REDIS_URL
if settings.CELERY_RESULT_BACKEND == "redis://localhost:6379/0":
    settings.CELERY_RESULT_BACKEND = settings.REDIS_URL

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
