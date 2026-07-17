"""Thread-safe runtime configuration persisted to disk (no redeploy needed)."""

from __future__ import annotations

import json
import logging
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from src.config import settings

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(settings.storage.local_path) / "runtime_config.json"


class MissionConfig(BaseModel):
    name: str = "default_mission"
    description: str = ""
    prompt: str = (
        "Eres un asistente de visión por computadora en el edge. "
        "Analiza eventos de cámara según la misión activa. "
        "Responde en JSON: action, severity, description."
    )
    mode: str = "inference"  # inference | training


class ProcessingConfig(BaseModel):
    use_case: str = ""
    fps: int = 10
    skip_frames: int = 2


class RuntimeTelegramConfig(BaseModel):
    enabled: bool = True
    cooldown: int = 30
    send_snapshot: bool = True


class RuntimeZonesConfig(BaseModel):
    paper_roi: list[int] = Field(default_factory=lambda: [130, 60, 950, 570])
    safe_zone: list[int] = Field(default_factory=lambda: [340, 180, 740, 500])
    entry: list[int] = Field(default_factory=lambda: [100, 200, 600, 500])
    exit: list[int] = Field(default_factory=lambda: [700, 200, 1200, 500])


class TrainingConfig(BaseModel):
    enabled: bool = False
    destination_url: str = "http://127.0.0.1:8787/api/training/upload"
    current_label: str = ""
    auto_capture: bool = False
    auto_capture_interval_sec: float = 2.0
    jpeg_quality: int = 90


class RuntimeDetectorConfig(BaseModel):
    confidence: float = 0.45
    min_object_area: int = 1200
    saturation_threshold: int = 60


class RuntimeConfigData(BaseModel):
    mission: MissionConfig = Field(default_factory=MissionConfig)
    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    telegram: RuntimeTelegramConfig = Field(default_factory=RuntimeTelegramConfig)
    zones: RuntimeZonesConfig = Field(default_factory=RuntimeZonesConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    detector: RuntimeDetectorConfig = Field(default_factory=RuntimeDetectorConfig)


class RuntimeConfig:
    """Mutable runtime settings with file persistence."""

    def __init__(self, path: Path = CONFIG_PATH):
        self._path = path
        self._lock = threading.RLock()
        self._data = self._build_defaults()
        self._load()

    def _build_defaults(self) -> RuntimeConfigData:
        uc = settings.active_use_case
        return RuntimeConfigData(
            mission=MissionConfig(
                name=settings.use_case,
                description=uc.description,
                prompt=settings.llm.system_prompt,
            ),
            processing=ProcessingConfig(
                use_case=settings.use_case,
                fps=uc.fps,
                skip_frames=uc.skip_frames,
            ),
            telegram=RuntimeTelegramConfig(
                enabled=settings.telegram.enabled,
                cooldown=settings.telegram.cooldown,
                send_snapshot=settings.telegram.send_snapshot,
            ),
            zones=RuntimeZonesConfig(
                paper_roi=uc.zones.paper_roi,
                safe_zone=uc.zones.safe_zone,
                entry=uc.zones.entry,
                exit=uc.zones.exit,
            ),
            detector=RuntimeDetectorConfig(
                confidence=settings.detector.confidence,
                min_object_area=uc.min_object_area,
            ),
        )

    def _load(self):
        if not self._path.exists():
            self._save()
            return
        try:
            with open(self._path) as f:
                raw = json.load(f)
            with self._lock:
                self._data = RuntimeConfigData(**raw)
            logger.info("Runtime config loaded from %s", self._path)
        except Exception as e:
            logger.warning("Failed to load runtime config, using defaults: %s", e)

    def _save(self):
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            payload = self._data.model_dump()
        with open(self._path, "w") as f:
            json.dump(payload, f, indent=2)

    def get(self) -> dict[str, Any]:
        with self._lock:
            return self._data.model_dump()

    def update(self, patch: dict[str, Any]) -> dict[str, Any]:
        with self._lock:
            current = self._data.model_dump()
            merged = _deep_merge(current, patch)
            self._data = RuntimeConfigData(**merged)
        self._save()
        logger.info("Runtime config updated")
        return self.get()

    @property
    def data(self) -> RuntimeConfigData:
        with self._lock:
            return deepcopy(self._data)

    def effective_use_case(self) -> str:
        with self._lock:
            return self._data.processing.use_case or settings.use_case

    def is_training_mode(self) -> bool:
        with self._lock:
            return self._data.mission.mode == "training" or self._data.training.enabled

    def telegram_enabled(self) -> bool:
        with self._lock:
            return self._data.telegram.enabled


def _deep_merge(base: dict, patch: dict) -> dict:
    result = deepcopy(base)
    for key, value in patch.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


runtime_config = RuntimeConfig()
