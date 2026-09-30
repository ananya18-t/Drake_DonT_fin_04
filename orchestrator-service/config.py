"""
config.py

Centralized configuration management for the Financial Crime & Insider Risk Intelligence Platform.
Uses pydantic-settings to load configuration from environment variables (or an optional
``.env`` next to this file) with safe fallbacks for single-machine local development.
"""

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


SERVICE_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables or an optional .env file.
    Provides type validation and default values for platform microservices.
    """

    # Neo4j Graph Database Configuration
    NEO4J_URI: str = Field(default="bolt://localhost:7687", description="Neo4j Bolt endpoint")
    NEO4J_USER: str = Field(default="neo4j", description="Neo4j authentication username")
    NEO4J_PASSWORD: str = Field(default="fincrime-dev-2026", description="Neo4j authentication password")

    # ML Engine Configuration
    ML_SERVICE_URL: str = Field(default="http://localhost:8001", description="Base URL for the ML Anomaly Engine")

    # Local LLM Configuration (llama-server, OpenAI-compatible API)
    LLM_BASE_URL: str = Field(default="http://localhost:8080", description="Base URL for llama-server")
    LLM_MODEL: str = Field(default="qwen3.5-9b-instruct", description="Model name reported in explanations")
    LLM_TIMEOUT_SECONDS: float = Field(default=180.0, description="Upper bound for one LLM generation")
    LLM_INLINE_WAIT_SECONDS: float = Field(
        default=10.0,
        description="How long an investigation request waits for a pending LLM narrative before "
                    "returning the rule-based explanation (the narrative keeps generating in background)",
    )

    # Detection tuning
    DATA_DIR: Path = Field(default=SERVICE_DIR.parent / "data-engine" / "data", description="Canonical CSV directory")
    INSIDER_WINDOW_HOURS: int = Field(default=48, description="Max delay between insider action and transfer")
    REPORTING_THRESHOLD: float = Field(default=10_000.0, description="Illustrative cash reporting threshold")
    STRUCTURING_MARGIN: float = Field(default=1_000.0, description="Band below the threshold treated as structuring")
    STRUCTURING_WINDOW_HOURS: int = Field(default=72, description="Window in which near-threshold transfers cluster")
    STRUCTURING_MIN_COUNT: int = Field(default=3, description="Near-threshold transfers needed inside the window")
    CYCLE_MAX_HOPS: int = Field(default=4, description="Longest money cycle searched (in transfers)")
    CYCLE_MAX_DAYS: int = Field(default=7, description="Max time between first and last transfer in a cycle")
    MAX_GRAPH_ACCOUNTS: int = Field(default=15, description="Accounts shown per investigation graph")

    # FastAPI Gateway Configuration
    API_HOST: str = Field(default="0.0.0.0", description="Host binding address for the gateway")
    API_PORT: int = Field(default=8000, description="Port assignment for the gateway")
    CORS_ORIGINS: list[str] = Field(
        default=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000"],
        description="Browser origins allowed to call the gateway",
    )

    model_config = SettingsConfigDict(
        env_file=SERVICE_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )


# Export an instantiated singleton for cross-module imports
settings = Settings()

NEO4J_URI = settings.NEO4J_URI
NEO4J_USER = settings.NEO4J_USER
NEO4J_PASSWORD = settings.NEO4J_PASSWORD
