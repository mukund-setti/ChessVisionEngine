"""Configuration management using Pydantic settings."""

from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Stockfish Configuration
    stockfish_path: str = "/usr/local/bin/stockfish"
    engine_depth: int = 20
    engine_threads: int = 4
    engine_hash_mb: int = 256

    # Model Configuration
    model_path: str = "models/piece_classifier.onnx"
    confidence_threshold: float = 0.85

    # Detection Settings
    board_detection_method: str = "hough"
    min_board_size: int = 200
    max_board_size: int = 2000

    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8080
    debug: bool = False
    cors_origins: list[str] = ["http://localhost:3000"]

    # Logging
    log_level: str = "INFO"
    log_file: str = "logs/app.log"

    @property
    def stockfish_executable(self) -> Path:
        """Get Stockfish executable path."""
        return Path(self.stockfish_path)

    @property
    def model_file(self) -> Path:
        """Get model file path."""
        return Path(self.model_path)


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()