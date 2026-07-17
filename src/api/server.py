"""FastAPI server for monitoring, control and training."""

import time
from datetime import datetime, timedelta
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse

from src.api.routes_control import create_control_router
from src.config import settings
from src.runtime.store import runtime_config
from src.storage.database import EventDatabase


class GateCommand:
    action: str
    reason: str = ""

    def __init__(self, action: str, reason: str = ""):
        self.action = action
        self.reason = reason


def create_app(
    db: EventDatabase | None = None,
    gate_controller=None,
    camera=None,
    detector=None,
    pipeline=None,
    training_uploader=None,
) -> FastAPI:
    from pydantic import BaseModel

    class GateCmd(BaseModel):
        action: str
        reason: str = ""

    app = FastAPI(
        title="CentinelaCam",
        description="Edge AI vision control system",
        version="0.2.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.db = db
    app.state.gate = gate_controller
    app.include_router(create_control_router(pipeline, camera, training_uploader))

    static_dir = Path(__file__).parent.parent / "static"

    @app.get("/health")
    async def health():
        from src.metrics.collector import metrics_collector

        m = metrics_collector.snapshot()
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "gate_open": gate_controller.is_open if gate_controller else None,
            "camera_connected": camera.frame is not None if camera else False,
            "mode": runtime_config.data.mission.mode,
            "telegram_enabled": runtime_config.telegram_enabled(),
            "pipeline_fps": m["pipeline_fps"],
        }

    @app.get("/events")
    async def get_events(since_hours: int = 1, event_type: str | None = None, limit: int = 50):
        if not db:
            raise HTTPException(503, "Database not initialized")
        since = datetime.now() - timedelta(hours=since_hours)
        events = db.get_events(since=since, event_type=event_type, limit=limit)
        return {"events": events, "count": len(events)}

    @app.get("/summary")
    async def daily_summary(date: str | None = None):
        if not db:
            raise HTTPException(503, "Database not initialized")
        summary = db.get_daily_summary(date)
        return {"date": date or datetime.now().strftime("%Y-%m-%d"), **summary}

    @app.post("/gate")
    async def control_gate(cmd: GateCmd):
        if not gate_controller:
            raise HTTPException(503, "Gate controller not initialized")
        if cmd.action == "open":
            gate_controller.open_gate(reason=cmd.reason or "API request")
        elif cmd.action == "close":
            gate_controller.close_gate()
        else:
            raise HTTPException(400, f"Invalid action: {cmd.action}")
        return {"status": "ok", "gate_open": gate_controller.is_open}

    @app.get("/stats")
    async def stats():
        if not db:
            raise HTTPException(503, "Database not initialized")
        today = datetime.now().strftime("%Y-%m-%d")
        return {
            "today": db.get_daily_summary(today),
            "gate_open": gate_controller.is_open if gate_controller else None,
            "mode": runtime_config.data.mission.mode,
        }

    @app.get("/video")
    async def video_feed():
        if not camera:
            raise HTTPException(503, "Camera not available")
        return StreamingResponse(
            _generate_frames(camera, detector, gate_controller, pipeline),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/snapshot")
    async def snapshot():
        if not camera or camera.frame is None:
            raise HTTPException(503, "No frame available")
        frame = _annotate_frame(camera.frame.copy(), detector, gate_controller, pipeline)
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return StreamingResponse(iter([jpeg.tobytes()]), media_type="image/jpeg")

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        dashboard_path = static_dir / "dashboard.html"
        if dashboard_path.exists():
            return FileResponse(dashboard_path)
        return HTMLResponse("<h1>CentinelaCam</h1><p>Dashboard not found</p>")

    return app


def _generate_frames(camera, detector, gate_controller, pipeline=None):
    while True:
        frame = camera.frame
        if frame is None:
            time.sleep(0.1)
            continue
        annotated = _annotate_frame(frame.copy(), detector, gate_controller, pipeline)
        _, jpeg = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 70])
        yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + jpeg.tobytes() + b"\r\n"
        time.sleep(0.1)


def _annotate_frame(frame: np.ndarray, detector, gate_controller, pipeline=None) -> np.ndarray:
    from src.runtime.store import runtime_config

    rc = runtime_config.data
    use_case = runtime_config.effective_use_case()

    if use_case == "zone_violation":
        roi = rc.zones.paper_roi
        cv2.rectangle(frame, (roi[0], roi[1]), (roi[2], roi[3]), (200, 200, 200), 1)
        sz = rc.zones.safe_zone
        cv2.rectangle(frame, (sz[0], sz[1]), (sz[2], sz[3]), (0, 255, 0), 2)
        cv2.putText(
            frame, "ZONA SEGURA", (sz[0] + 5, sz[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        if pipeline and hasattr(pipeline, "_zone_detections"):
            for (x, y, bw, bh, inside) in pipeline._zone_detections:
                color = (0, 255, 0) if inside else (0, 0, 255)
                label = "DENTRO" if inside else "FUERA!"
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    elif use_case == "object_watch":
        dets = getattr(pipeline, "_yolo_detections", []) if pipeline else []
        for det in dets:
            x1, y1, x2, y2 = det["bbox"]
            color = (0, 200, 255)
            label = f"{det['class_name']} {det['confidence']:.0%}"
            if det.get("track_id"):
                label += f" #{det['track_id']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    else:
        if detector:
            result = detector.detect(frame)
            for det in result.detections:
                x1, y1, x2, y2 = det.bbox
                color = (0, 255, 0) if det.class_id == 0 else (0, 200, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                label = f"{det.class_name} {det.confidence:.0%}"
                if det.track_id:
                    label += f" #{det.track_id}"
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    if rc.mission.mode == "training":
        cv2.putText(frame, f"TRAINING: {rc.training.current_label or 'sin label'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
    else:
        cv2.putText(frame, use_case.upper().replace("_", " "), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)

    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    return frame
