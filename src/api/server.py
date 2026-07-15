"""FastAPI server for monitoring and control."""

import time
from datetime import datetime, timedelta

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

from src.config import settings
from src.storage.database import EventDatabase


class GateCommand(BaseModel):
    action: str  # "open" | "close"
    reason: str = ""


def create_app(db: EventDatabase | None = None, gate_controller=None, camera=None, detector=None, pipeline=None) -> FastAPI:
    app = FastAPI(
        title="CAM-PI Gate Controller",
        description="Edge AI gate control system for loading docks",
        version="0.1.0",
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

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "timestamp": datetime.now().isoformat(),
            "gate_open": gate_controller.is_open if gate_controller else None,
            "camera_connected": camera.frame is not None if camera else False,
        }

    @app.get("/events")
    async def get_events(
        since_hours: int = 1, event_type: str | None = None, limit: int = 50
    ):
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
    async def control_gate(cmd: GateCommand):
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
        }

    @app.get("/video")
    async def video_feed():
        """MJPEG stream with detection overlays."""
        if not camera:
            raise HTTPException(503, "Camera not available")
        return StreamingResponse(
            _generate_frames(camera, detector, gate_controller, pipeline),
            media_type="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/snapshot")
    async def snapshot():
        """Single JPEG frame with detections."""
        if not camera or camera.frame is None:
            raise HTTPException(503, "No frame available")
        frame = _annotate_frame(camera.frame.copy(), detector, gate_controller, pipeline)
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return StreamingResponse(
            iter([jpeg.tobytes()]),
            media_type="image/jpeg",
        )

    @app.get("/", response_class=HTMLResponse)
    async def dashboard():
        """Simple live dashboard."""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>CentinelaCam - Live View</title>
            <style>
                body { margin:0; background:#1a1a2e; color:#eee; font-family:system-ui; }
                .container { max-width:1200px; margin:0 auto; padding:20px; }
                h1 { color:#f5a623; margin-bottom:5px; }
                .subtitle { color:#888; margin-bottom:20px; }
                .video-box { background:#000; border-radius:8px; overflow:hidden; margin-bottom:20px; }
                .video-box img { width:100%; display:block; }
                .stats { display:grid; grid-template-columns:repeat(auto-fit,minmax(150px,1fr)); gap:12px; }
                .stat-card { background:#16213e; padding:16px; border-radius:8px; text-align:center; }
                .stat-value { font-size:2em; font-weight:bold; color:#f5a623; }
                .stat-label { color:#888; font-size:0.85em; margin-top:4px; }
                .status { display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }
                .status.on { background:#4caf50; }
                .status.off { background:#f44336; }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>CentinelaCam</h1>
                <p class="subtitle">{settings.node.location} | {settings.active_use_case.description}</p>
                <div class="video-box">
                    <img src="/video" alt="Live Feed">
                </div>
                <div class="stats" id="stats"></div>
            </div>
            <script>
                async function updateStats() {
                    try {
                        const r = await fetch('/stats');
                        const d = await r.json();
                        const t = d.today || {};
                        document.getElementById('stats').innerHTML = `
                            <div class="stat-card">
                                <div class="stat-value">${t.total_person_in || 0}</div>
                                <div class="stat-label">Personas Entrada</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${t.total_person_out || 0}</div>
                                <div class="stat-label">Personas Salida</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${t.total_vehicle_in || 0}</div>
                                <div class="stat-label">Vehiculos Entrada</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value">${t.total_vehicle_out || 0}</div>
                                <div class="stat-label">Vehiculos Salida</div>
                            </div>
                            <div class="stat-card">
                                <div class="stat-value"><span class="status ${d.gate_open?'on':'off'}"></span>${d.gate_open?'ABIERTA':'CERRADA'}</div>
                                <div class="stat-label">Compuerta</div>
                            </div>
                        `;
                    } catch(e) {}
                }
                updateStats();
                setInterval(updateStats, 3000);
            </script>
        </body>
        </html>
        """

    return app


def _generate_frames(camera, detector, gate_controller, pipeline=None):
    """Generate MJPEG frames with detection overlays."""
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
    """Draw detections, zones, and status on frame."""
    uc = settings.active_use_case

    if settings.use_case == "zone_violation":
        sz = uc.zones.safe_zone
        cv2.rectangle(frame, (sz[0], sz[1]), (sz[2], sz[3]), (0, 255, 0), 2)
        cv2.putText(frame, "ZONA SEGURA", (sz[0] + 5, sz[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw bounding boxes from contour-based detection
        if pipeline and hasattr(pipeline, "_zone_detections"):
            for (x, y, bw, bh, inside) in pipeline._zone_detections:
                color = (0, 255, 0) if inside else (0, 0, 255)
                label = "OK" if inside else "FUERA!"
                cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
                cv2.putText(frame, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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

        entry_zone = uc.zones.entry
        exit_zone = uc.zones.exit
        cv2.rectangle(frame, (entry_zone[0], entry_zone[1]), (entry_zone[2], entry_zone[3]), (255, 200, 0), 2)
        cv2.putText(frame, "ENTRADA", (entry_zone[0] + 5, entry_zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
        cv2.rectangle(frame, (exit_zone[0], exit_zone[1]), (exit_zone[2], exit_zone[3]), (0, 150, 255), 2)
        cv2.putText(frame, "SALIDA", (exit_zone[0] + 5, exit_zone[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 255), 2)

    status_text = settings.use_case.upper().replace("_", " ")
    cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(frame, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    return frame
