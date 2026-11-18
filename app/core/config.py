from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    app_name: str = "PathSnap API"
    api_v1_prefix: str = "/api/v1"
    cors_origins: str = "http://localhost:3000,http://localhost:5173"
    model_dir: Path = ROOT / "models"
    default_model: str = "xgboost"

    model_config = SettingsConfigDict(env_file=".env", env_prefix="PATHSNAP_", extra="ignore")

    @property
    def allowed_origins(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]


settings = Settings()
