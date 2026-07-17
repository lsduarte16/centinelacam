#!/usr/bin/env python3
"""Local training image receiver — run on dev PC to collect labeled frames from RPi.

Usage:
    python scripts/training_receiver.py
    # listens on http://0.0.0.0:8787/api/training/upload

Images saved to: ./training_data/<mission>/<label>/
"""

from __future__ import annotations

import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
app = FastAPI(title="CentinelaCam Training Receiver")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w\-.]", "_", value.strip())[:80] or "unknown"


@app.post("/api/training/upload")
async def upload_training_image(
    image: UploadFile = File(...),
    label: str = Form(...),
    mission: str = Form("default"),
    node_id: str = Form(""),
    location: str = Form(""),
    source: str = Form("manual"),
    timestamp: str = Form(""),
):
    safe_label = _safe_name(label)
    safe_mission = _safe_name(mission)
    dest_dir = DATA_DIR / safe_mission / safe_label
    dest_dir.mkdir(parents=True, exist_ok=True)

    ts = timestamp.replace(":", "-") if timestamp else time.strftime("%Y%m%d-%H%M%S")
    filename = f"{ts}_{node_id or 'node'}_{source}.jpg"
    filepath = dest_dir / filename

    content = await image.read()
    filepath.write_bytes(content)

    return {
        "status": "ok",
        "saved_to": str(filepath),
        "label": safe_label,
        "mission": safe_mission,
        "bytes": len(content),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "data_dir": str(DATA_DIR)}


@app.get("/api/training/stats")
async def stats():
    total = 0
    by_mission: dict[str, int] = {}
    if DATA_DIR.exists():
        for mission_dir in DATA_DIR.iterdir():
            if not mission_dir.is_dir():
                continue
            count = sum(1 for _ in mission_dir.rglob("*.jpg"))
            by_mission[mission_dir.name] = count
            total += count
    return {"total_images": total, "by_mission": by_mission}


if __name__ == "__main__":
    print(f"Training receiver → saving to {DATA_DIR}")
    uvicorn.run(app, host="0.0.0.0", port=8787, log_level="info")
