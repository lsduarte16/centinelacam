"""YOLOv8 detector optimized for Raspberry Pi 5."""

import logging
from pathlib import Path

import numpy as np
from ultralytics import YOLO

from src.config import settings

from .models import Detection, TrackingResult

logger = logging.getLogger(__name__)

COCO_NAMES = {
    0: "persona",
    1: "bicicleta",
    2: "auto",
    3: "moto",
    5: "bus",
    7: "camion",
}


class YOLODetector:
    """YOLOv8 object detector with ByteTrack tracking."""

    def __init__(self):
        self.model_path = settings.detector.model
        self.confidence = settings.detector.confidence
        self.iou_threshold = settings.detector.iou_threshold
        self.device = settings.detector.device
        self.classes = settings.active_use_case.classes
        self.imgsz = settings.detector.imgsz
        self.use_tracking = settings.detector.track
        self._model: YOLO | None = None
        self._frame_count = 0

    def load(self) -> "YOLODetector":
        """Load the YOLO model. Downloads if not present."""
        model_file = Path(self.model_path)

        if not model_file.exists():
            logger.info("Downloading model %s...", self.model_path)

        self._model = YOLO(self.model_path)
        logger.info("Model loaded: %s on device=%s", self.model_path, self.device)
        return self

    def detect(self, frame: np.ndarray) -> TrackingResult:
        """Run detection (with optional tracking) on a frame."""
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        self._frame_count += 1

        if self.use_tracking:
            results = self._model.track(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                classes=self.classes,
                imgsz=self.imgsz,
                persist=True,
                verbose=False,
            )
        else:
            results = self._model(
                frame,
                conf=self.confidence,
                iou=self.iou_threshold,
                device=self.device,
                classes=self.classes,
                imgsz=self.imgsz,
                verbose=False,
            )

        detections = self._parse_results(results)

        return TrackingResult(
            frame_id=self._frame_count,
            detections=detections,
        )

    def _parse_results(self, results) -> list[Detection]:
        """Parse YOLO results into Detection objects."""
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            boxes = result.boxes
            for i in range(len(boxes)):
                cls_id = int(boxes.cls[i])
                conf = float(boxes.conf[i])
                x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy().astype(int)

                track_id = None
                if boxes.id is not None:
                    track_id = int(boxes.id[i])

                detections.append(
                    Detection(
                        class_id=cls_id,
                        class_name=COCO_NAMES.get(cls_id, f"class_{cls_id}"),
                        confidence=conf,
                        bbox=(x1, y1, x2, y2),
                        track_id=track_id,
                    )
                )

        return detections

    def warmup(self):
        """Run inference on a dummy frame to warm up the model."""
        if self._model is None:
            self.load()
        dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        self.detect(dummy)
        logger.info("Model warmup complete")
