"""Centralized configuration loader with use-case support."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class NodeConfig(BaseModel):
    id: str = "node-001"
    location: str = ""
    description: str = ""


class CameraConfig(BaseModel):
    source: str = "0"
    resolution: list[int] = [1280, 720]
    reconnect_delay: int = 5


class DetectorConfig(BaseModel):
    model: str = "yolov8n.pt"
    confidence: float = 0.45
    iou_threshold: float = 0.5
    device: str = "cpu"
    imgsz: int = 640
    track: bool = True


class TelegramConfig(BaseModel):
    enabled: bool = False
    bot_token: str = ""
    chat_id: str = ""
    notify_on_event: bool = True
    send_snapshot: bool = True
    cooldown: int = 30

    def model_post_init(self, __context):
        import os
        if not self.bot_token:
            self.bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        if not self.chat_id:
            self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")


class CountingLineConfig(BaseModel):
    orientation: str = "horizontal"
    position: int = 360
    direction_in: str = "down"


class ZonesConfig(BaseModel):
    entry: list[int] = [100, 200, 600, 500]
    exit: list[int] = [700, 200, 1200, 500]
    active: list[int] = [100, 100, 1180, 620]
    safe_zone: list[int] = [400, 200, 880, 520]


class UseCaseConfig(BaseModel):
    description: str = ""
    classes: list[int] = [0, 2, 7]
    fps: int = 10
    skip_frames: int = 2
    gpio_pin: int | None = None
    open_duration: int = 30
    cooldown: int = 5
    zones: ZonesConfig = ZonesConfig()
    counting_line: CountingLineConfig = CountingLineConfig()
    scan_region: list[int] = [200, 150, 1080, 570]
    motion_threshold: int = 5000
    alert_after_seconds: int = 3
    events: list[str] = []


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "tinyllama"
    base_url: str = "http://127.0.0.1:11434"
    timeout: int = 30
    system_prompt: str = (
        "Eres un asistente de seguridad para control de andén logístico. "
        "Analiza eventos de cámara y genera alertas contextuales. "
        "Responde SOLO en formato JSON con campos: action, severity, description."
    )


class CloudSyncConfig(BaseModel):
    enabled: bool = False
    provider: str = "s3"
    bucket: str = "centinelacam-backup"
    sync_interval: int = 3600


class StorageConfig(BaseModel):
    local_path: str = "/data/cam-pi"
    db_path: str = "/data/cam-pi/events.db"
    retention_days: int = 30
    cloud_sync: CloudSyncConfig = CloudSyncConfig()


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["*"]


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "/data/cam-pi/logs/app.log"
    max_size_mb: int = 50
    backup_count: int = 5


class Settings(BaseSettings):
    use_case: str = "gate_control"
    node: NodeConfig = NodeConfig()
    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    telegram: TelegramConfig = TelegramConfig()
    use_cases: dict[str, Any] = {}
    llm: LLMConfig = LLMConfig()
    storage: StorageConfig = StorageConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()

    @property
    def active_use_case(self) -> UseCaseConfig:
        """Get the configuration for the active use case."""
        uc_data = self.use_cases.get(self.use_case, {})
        return UseCaseConfig(**uc_data)


def load_settings(config_path: str | Path | None = None) -> Settings:
    """Load settings from YAML file, falling back to defaults."""
    if config_path is None:
        config_path = Path(__file__).parent.parent / "config" / "settings.yaml"

    config_path = Path(config_path)
    if config_path.exists():
        with open(config_path) as f:
            data: dict[str, Any] = yaml.safe_load(f) or {}
        return Settings(**data)

    return Settings()


settings = load_settings()
