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
        if not camera or camera.frame is None:
            raise HTTPException(503, "No frame available")
        if not training_uploader:
            raise HTTPException(503, "Training uploader not initialized")

        ok = training_uploader.upload_frame(camera.frame.copy(), req.label, source="manual")
        if not ok:
            raise HTTPException(502, "Upload to destination failed")
        return {"status": "ok", "label": req.label}

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
