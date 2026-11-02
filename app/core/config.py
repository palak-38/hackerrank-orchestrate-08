from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    groq_api_key: SecretStr
    agent_model: str = "openai/gpt-oss-20b"
    agent_max_turns: int = 5
    # AGENT_VERBOSE=true prints a per-message RoutingTrace to stderr.
    agent_verbose: bool = False

    model_config = SettingsConfigDict(
        env_file=ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


settings = Settings()
