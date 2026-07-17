"""Upload training images to external destination (no local storage on RPi)."""

from __future__ import annotations

import io
import logging
import time
from threading import Thread

import cv2
import httpx
import numpy as np

from src.config import settings
from src.metrics.collector import metrics_collector
from src.runtime.store import runtime_config

logger = logging.getLogger(__name__)


class TrainingUploader:
    """POST labeled frames to a remote training collector."""

    def __init__(self):
        self._client = httpx.Client(timeout=30)
        self._last_auto_capture = 0.0

    def upload_frame(
        self,
        frame: np.ndarray,
        label: str,
        mission_name: str | None = None,
        source: str = "manual",
    ) -> bool:
        cfg = runtime_config.data.training
        if not label.strip():
            logger.warning("Training upload skipped: empty label")
            return False
        if not cfg.destination_url.strip():
            logger.warning("Training upload skipped: no destination_url")
            return False

        mission = mission_name or runtime_config.data.mission.name
        _, jpeg = cv2.imencode(
            ".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, cfg.jpeg_quality]
        )
        photo = io.BytesIO(jpeg.tobytes())
        photo.name = f"{label}_{int(time.time())}.jpg"

        data = {
            "label": label.strip(),
            "mission": mission,
            "node_id": settings.node.id,
            "location": settings.node.location,
            "source": source,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }

        try:
            resp = self._client.post(
                cfg.destination_url,
                data=data,
                files={"image": ("frame.jpg", photo, "image/jpeg")},
            )
            ok = resp.status_code in (200, 201)
            metrics_collector.record_training_upload(ok)
            if ok:
                logger.info("Training image uploaded: label=%s mission=%s", label, mission)
            else:
                logger.warning(
                    "Training upload failed HTTP %s: %s",
                    resp.status_code,
                    resp.text[:200],
                )
            return ok
        except Exception as e:
            metrics_collector.record_training_upload(False)
            logger.error("Training upload error: %s", e)
            return False

    def upload_async(self, frame: np.ndarray, label: str, source: str = "auto"):
        Thread(
            target=self.upload_frame,
            args=(frame.copy(), label, None, source),
            daemon=True,
        ).start()

    def maybe_auto_capture(self, frame: np.ndarray) -> bool:
        cfg = runtime_config.data.training
        if not cfg.auto_capture or not cfg.current_label.strip():
            return False

        now = time.time()
        if now - self._last_auto_capture < cfg.auto_capture_interval_sec:
            return False

        self._last_auto_capture = now
        self.upload_async(frame, cfg.current_label, source="auto")
        return True

    def shutdown(self):
        self._client.close()
