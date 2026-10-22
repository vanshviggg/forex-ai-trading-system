"""Environment-backed settings.

Secrets and tunables come from the process environment or a local ``.env`` file.
Nothing in this module hardcodes an API key. The MVP uses Yahoo Finance, which
does not require a key.
"""

import logging
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for scripts and the API."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_env: str = "development"
    log_level: str = "INFO"
    default_symbol: str = "EURUSD"
    default_period: str = "2y"
    default_interval: str = "1d"
    risk_per_trade: float = Field(default=0.01, gt=0, lt=1)
    min_risk_reward: float = Field(default=1.5, gt=0)
    max_risk_per_trade: float = Field(default=0.02, gt=0, lt=1)
    max_drawdown: float = Field(default=0.20, gt=0, lt=1)
    max_leverage: float = Field(default=30.0, gt=0)
    initial_capital: float = Field(default=10_000.0, gt=0)
    transaction_cost_pips: float = Field(default=1.0, ge=0)
    slippage_pips: float = Field(default=0.5, ge=0)
    model_dir: Path = Path("models")
    data_dir: Path = Path("data")
    model_filename: str = "direction_model.joblib"

    @model_validator(mode="after")
    def check_risk_bounds(self) -> "Settings":
        if self.risk_per_trade > self.max_risk_per_trade:
            raise ValueError("RISK_PER_TRADE cannot exceed MAX_RISK_PER_TRADE.")
        return self

    @property
    def model_path(self) -> Path:
        """Path of the joblib bundle written by ``scripts/train_model.py``."""
        return self.model_dir / self.model_filename


def configure_logging(level: str = "INFO") -> None:
    """Configure process logging. Application code should log, not print."""
    numeric = getattr(logging, level.upper(), None)
    if not isinstance(numeric, int):
        numeric = logging.INFO
    logging.basicConfig(
        level=numeric,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        force=True,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Load settings once. Existing environment variables win over ``.env``."""
    load_dotenv(override=False)
    return Settings()
