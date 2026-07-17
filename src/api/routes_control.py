"""API routes for runtime config, metrics and training."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from src.metrics.collector import metrics_collector
from src.runtime.store import runtime_config

router = APIRouter(prefix="/api")


class ConfigPatch(BaseModel):
    mission: dict[str, Any] | None = None
    processing: dict[str, Any] | None = None
    telegram: dict[str, Any] | None = None
    zones: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    detector: dict[str, Any] | None = None


class TrainingCaptureRequest(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    # Pixel boxes: [{x1,y1,x2,y2}, ...] — required for YOLO detection training
    boxes: list[dict[str, float]] = Field(..., min_length=1)
    # Optional JPEG base64 (without data: prefix). If omitted, uses live camera frame.
    image_b64: str | None = None


def create_control_router(pipeline=None, camera=None, training_uploader=None) -> APIRouter:
    @router.get("/config")
    async def get_config():
        return runtime_config.get()

    @router.patch("/config")
    async def patch_config(body: ConfigPatch):
        patch = body.model_dump(exclude_none=True)
        if not patch:
            raise HTTPException(400, "No fields to update")
        return {"status": "ok", "config": runtime_config.update(patch)}

    @router.get("/metrics")
    async def get_metrics():
        data = metrics_collector.snapshot()
        data["mode"] = runtime_config.data.mission.mode
        data["use_case"] = runtime_config.effective_use_case()
        data["telegram_enabled"] = runtime_config.telegram_enabled()
        data["training_enabled"] = runtime_config.is_training_mode()
        return data

    @router.get("/training/status")
    async def training_status():
        cfg = runtime_config.data.training
        m = metrics_collector.snapshot()
        return {
            "mode": runtime_config.data.mission.mode,
            "training_enabled": cfg.enabled,
            "auto_capture": cfg.auto_capture,
            "current_label": cfg.current_label,
            "destination_url": cfg.destination_url,
            "uploads_ok": m["training_uploads_ok"],
            "uploads_fail": m["training_uploads_fail"],
        }

    @router.post("/training/capture")
    async def training_capture(req: TrainingCaptureRequest):
        if not training_uploader:
            raise HTTPException(503, "Training uploader not initialized")
        if not req.boxes:
            raise HTTPException(400, "At least one bounding box is required")

        for box in req.boxes:
            if not all(k in box for k in ("x1", "y1", "x2", "y2")):
                raise HTTPException(400, "Each box needs x1,y1,x2,y2")

        if req.image_b64:
            import base64
            try:
                raw = base64.b64decode(req.image_b64)
                arr = np.frombuffer(raw, dtype=np.uint8)
                frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception as e:
                raise HTTPException(400, f"Invalid image_b64: {e}") from e
            if frame is None:
                raise HTTPException(400, "Could not decode image_b64")
        else:
            if not camera or camera.frame is None:
                raise HTTPException(503, "No frame available")
            frame = camera.frame.copy()

        result = training_uploader.upload_frame(
            frame,
            req.label,
            boxes=req.boxes,
            source="manual",
        )
        if not result:
            raise HTTPException(502, "Upload to destination failed")
        return {"status": "ok", "label": req.label, "yolo": result}

    @router.get("/training/snapshot")
    async def training_snapshot():
        """Raw JPEG for annotation UI (no overlays)."""
        if not camera or camera.frame is None:
            raise HTTPException(503, "No frame available")
        frame = camera.frame.copy()
        h, w = frame.shape[:2]
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
        from fastapi.responses import Response

        return Response(
            content=jpeg.tobytes(),
            media_type="image/jpeg",
            headers={"X-Image-Width": str(w), "X-Image-Height": str(h)},
        )

    @router.post("/training/mode/{mode}")
    async def set_training_mode(mode: str):
        if mode not in ("inference", "training"):
            raise HTTPException(400, "mode must be inference or training")
        patch = {
            "mission": {"mode": mode},
            "training": {"enabled": mode == "training"},
        }
        runtime_config.update(patch)
        return {"status": "ok", "mode": mode}

    @router.get("/classes")
    async def list_classes():
        from src.detector.coco_classes import COCO_GROUPS, class_catalog

        return {"classes": class_catalog(), "groups": COCO_GROUPS}

    return router
