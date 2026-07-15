"""Centralized configuration loader."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import BaseSettings


class CameraConfig(BaseModel):
    rtsp_url: str = "rtsp://admin:password@192.168.1.100:554/stream1"
    fps: int = 15
    resolution: list[int] = [1280, 720]
    reconnect_delay: int = 5


class DetectorConfig(BaseModel):
    model: str = "yolov8n.pt"
    confidence: float = 0.45
    iou_threshold: float = 0.5
    device: str = "cpu"
    classes: list[int] = [0, 2, 7]
    imgsz: int = 640
    track: bool = True


class GateZones(BaseModel):
    entry: list[int] = [100, 200, 600, 500]
    exit: list[int] = [700, 200, 1200, 500]


class GateConfig(BaseModel):
    gpio_pin: int = 17
    open_duration: int = 30
    cooldown: int = 5
    require_authorization: bool = True
    authorized_plates: list[str] = []
    zones: GateZones = GateZones()


class LLMConfig(BaseModel):
    provider: str = "ollama"
    model: str = "tinyllama"
    base_url: str = "http://localhost:11434"
    timeout: int = 30
    system_prompt: str = (
        "Eres un asistente de seguridad para control de andén logístico. "
        "Analiza eventos de cámara y genera alertas contextuales. "
        "Responde SOLO en formato JSON con campos: action, severity, description."
    )


class CloudSyncConfig(BaseModel):
    enabled: bool = True
    provider: str = "s3"
    bucket: str = "cam-pi-backup"
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
    camera: CameraConfig = CameraConfig()
    detector: DetectorConfig = DetectorConfig()
    gate: GateConfig = GateConfig()
    llm: LLMConfig = LLMConfig()
    storage: StorageConfig = StorageConfig()
    api: APIConfig = APIConfig()
    logging: LoggingConfig = LoggingConfig()


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
