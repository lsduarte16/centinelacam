"""Upload YOLO-annotated training samples to an external collector (no RPi storage)."""

from __future__ import annotations

import io
import json
import logging
import time
from threading import Thread
from typing import Any

import cv2
import httpx
import numpy as np

from src.config import settings
from src.metrics.collector import metrics_collector
from src.runtime.store import runtime_config

logger = logging.getLogger(__name__)


class TrainingUploader:
    """POST labeled frames + YOLO boxes to a remote training collector."""

    def __init__(self):
        self._client = httpx.Client(timeout=30)
        self._last_auto_capture = 0.0

    def upload_frame(
        self,
        frame: np.ndarray,
        label: str,
        boxes: list[dict[str, Any]] | list[list[float]],
        mission_name: str | None = None,
        source: str = "manual",
    ) -> dict[str, Any] | None:
        """Upload one sample.

        boxes: list of {x1,y1,x2,y2} in pixel coords (preferred) or [x1,y1,x2,y2].
        """
        cfg = runtime_config.data.training
        if not label.strip():
            logger.warning("Training upload skipped: empty label")
            return None
        if not cfg.destination_url.strip():
            logger.warning("Training upload skipped: no destination_url")
            return None
        if not boxes:
            logger.warning("Training upload skipped: no bounding boxes")
            return None

        mission = mission_name or runtime_config.data.mission.name
        h, w = frame.shape[:2]
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
            "annotations": json.dumps(boxes),
            "image_width": str(w),
            "image_height": str(h),
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
                payload = resp.json()
                logger.info(
                    "YOLO sample uploaded: label=%s class_id=%s boxes=%s",
                    label,
                    payload.get("class_id"),
                    payload.get("boxes"),
                )
                return payload
            logger.warning(
                "Training upload failed HTTP %s: %s",
                resp.status_code,
                resp.text[:300],
            )
            return None
        except Exception as e:
            metrics_collector.record_training_upload(False)
            logger.error("Training upload error: %s", e)
            return None

    def upload_async(
        self,
        frame: np.ndarray,
        label: str,
        boxes: list[dict[str, Any]] | list[list[float]],
        source: str = "auto",
    ):
        Thread(
            target=self.upload_frame,
            args=(frame.copy(), label, boxes, None, source),
            daemon=True,
        ).start()

    def maybe_auto_capture(self, frame: np.ndarray, detections: list | None = None) -> bool:
        """Auto-capture only when there are detections to use as boxes."""
        cfg = runtime_config.data.training
        if not cfg.auto_capture or not cfg.current_label.strip():
            return False
        if not detections:
            return False

        now = time.time()
        if now - self._last_auto_capture < cfg.auto_capture_interval_sec:
            return False

        boxes = []
        for det in detections:
            bbox = det.get("bbox") if isinstance(det, dict) else getattr(det, "bbox", None)
            if not bbox or len(bbox) != 4:
                continue
            x1, y1, x2, y2 = bbox
            boxes.append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2)})

        if not boxes:
            return False

        self._last_auto_capture = now
        self.upload_async(frame, cfg.current_label, boxes, source="auto")
        return True

    def shutdown(self):
        self._client.close()
