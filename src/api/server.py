"""FastAPI server for monitoring and control."""

from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.config import settings
from src.storage.database import EventDatabase


class GateCommand(BaseModel):
    action: str  # "open" | "close"
    reason: str = ""


def create_app(db: EventDatabase | None = None, gate_controller=None) -> FastAPI:
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

    return app
