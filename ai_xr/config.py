from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")
    provider: str = "extractive"
    data_dir: Path = Path("data")
    llm_base_url: str = "http://127.0.0.1:11434/v1"
    llm_model: str = "qwen2.5:0.5b"
    llm_api_key: str = ""
    local_model: str = "models/qwen2.5-0.5b-instruct"
    embedding_model: str = ""
    reranker_model: str = ""
    yolo_model: str = "models/yolo11n.pt"
    search_api_key: str = ""
    app_api_key: str = ""
    max_upload_bytes: int = 10 * 1024 * 1024

    def prepare(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self
