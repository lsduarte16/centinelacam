#!/usr/bin/env python3
"""YOLO-ready training dataset receiver.

Stores samples in Ultralytics/YOLO layout:

    training_data/<mission>/
      data.yaml
      classes.json
      images/train/<stem>.jpg
      labels/train/<stem>.txt

Each label line: class_id x_center y_center width height  (normalized 0-1)

Usage:
    .venv/bin/python scripts/training_receiver.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

DATA_DIR = Path(__file__).resolve().parent.parent / "training_data"
app = FastAPI(title="CentinelaCam YOLO Training Receiver", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _safe_name(value: str) -> str:
    return re.sub(r"[^\w\-.]", "_", value.strip())[:80] or "unknown"


def _mission_root(mission: str) -> Path:
    root = DATA_DIR / _safe_name(mission)
    (root / "images" / "train").mkdir(parents=True, exist_ok=True)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    return root


def _load_classes(root: Path) -> list[str]:
    path = root / "classes.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text())
        return list(data.get("names", []))
    except Exception:
        return []


def _save_classes(root: Path, names: list[str]) -> None:
    path = root / "classes.json"
    path.write_text(json.dumps({"names": names}, indent=2, ensure_ascii=False) + "\n")
    _write_data_yaml(root, names)


def _write_data_yaml(root: Path, names: list[str]) -> None:
    """Ultralytics-compatible dataset descriptor (no PyYAML dependency)."""
    lines = [
        f"path: {root.resolve()}",
        "train: images/train",
        "val: images/train",
        f"nc: {len(names)}",
        "names:",
    ]
    for i, n in enumerate(names):
        lines.append(f"  {i}: {n}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n")


def _ensure_class_id(root: Path, label: str) -> int:
    names = _load_classes(root)
    if label not in names:
        names.append(label)
        _save_classes(root, names)
    return names.index(label)


def _parse_boxes(raw: str, img_w: int, img_h: int) -> list[tuple[float, float, float, float]]:
    """Parse boxes JSON into normalized YOLO boxes (xc, yc, w, h).

    Accepted formats per box:
      {"x1":..,"y1":..,"x2":..,"y2":..}  pixel corners
      {"xc":..,"yc":..,"w":..,"h":..}    already normalized 0-1
      [x1,y1,x2,y2]                      pixel corners
    """
    if not raw or not raw.strip():
        raise HTTPException(400, "annotations required (bbox JSON)")

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise HTTPException(400, f"invalid annotations JSON: {e}") from e

    if isinstance(data, dict) and "boxes" in data:
        data = data["boxes"]
    if not isinstance(data, list) or not data:
        raise HTTPException(400, "annotations must be a non-empty list of boxes")

    if img_w <= 0 or img_h <= 0:
        raise HTTPException(400, "image_width/image_height required")

    out: list[tuple[float, float, float, float]] = []
    for box in data:
        if isinstance(box, dict) and all(k in box for k in ("xc", "yc", "w", "h")):
            xc, yc, w, h = float(box["xc"]), float(box["yc"]), float(box["w"]), float(box["h"])
        elif isinstance(box, dict) and all(k in box for k in ("x1", "y1", "x2", "y2")):
            x1, y1, x2, y2 = float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"])
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
        elif isinstance(box, (list, tuple)) and len(box) == 4:
            x1, y1, x2, y2 = map(float, box)
            xc = ((x1 + x2) / 2) / img_w
            yc = ((y1 + y2) / 2) / img_h
            w = abs(x2 - x1) / img_w
            h = abs(y2 - y1) / img_h
        else:
            raise HTTPException(400, f"unsupported box format: {box}")

        # clamp
        w = max(1e-6, min(w, 1.0))
        h = max(1e-6, min(h, 1.0))
        xc = max(w / 2, min(xc, 1.0 - w / 2))
        yc = max(h / 2, min(yc, 1.0 - h / 2))
        out.append((xc, yc, w, h))

    return out


@app.post("/api/training/upload")
async def upload_training_image(
    image: UploadFile = File(...),
    label: str = Form(...),
    mission: str = Form("default"),
    node_id: str = Form(""),
    location: str = Form(""),
    source: str = Form("manual"),
    timestamp: str = Form(""),
    annotations: str = Form(...),  # JSON boxes — required for YOLO
    image_width: int = Form(...),
    image_height: int = Form(...),
):
    safe_label = _safe_name(label)
    root = _mission_root(mission)
    class_id = _ensure_class_id(root, safe_label)
    boxes = _parse_boxes(annotations, image_width, image_height)

    ts = timestamp.replace(":", "-") if timestamp else time.strftime("%Y%m%d-%H%M%S")
    stem = f"{ts}_{_safe_name(node_id) or 'node'}_{_safe_name(source)}_{safe_label}"
    img_path = root / "images" / "train" / f"{stem}.jpg"
    lbl_path = root / "labels" / "train" / f"{stem}.txt"

    content = await image.read()
    if not content:
        raise HTTPException(400, "empty image")
    img_path.write_bytes(content)

    lines = [f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}" for xc, yc, w, h in boxes]
    lbl_path.write_text("\n".join(lines) + "\n")

    # sidecar metadata for audit / future tooling
    meta = {
        "mission": _safe_name(mission),
        "label": safe_label,
        "class_id": class_id,
        "node_id": node_id,
        "location": location,
        "source": source,
        "timestamp": timestamp or time.strftime("%Y-%m-%dT%H:%M:%S"),
        "image_width": image_width,
        "image_height": image_height,
        "boxes_count": len(boxes),
        "image": str(img_path.relative_to(root)),
        "label_file": str(lbl_path.relative_to(root)),
    }
    (root / "images" / "train" / f"{stem}.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n"
    )

    names = _load_classes(root)
    return {
        "status": "ok",
        "format": "yolo",
        "mission": _safe_name(mission),
        "label": safe_label,
        "class_id": class_id,
        "classes": names,
        "image": str(img_path),
        "label_file": str(lbl_path),
        "boxes": len(boxes),
        "data_yaml": str(root / "data.yaml"),
        "bytes": len(content),
    }


@app.get("/health")
async def health():
    return {"status": "ok", "data_dir": str(DATA_DIR), "format": "yolo"}


@app.get("/api/training/stats")
async def stats():
    result: dict = {"total_images": 0, "missions": {}}
    if not DATA_DIR.exists():
        return result

    for mission_dir in sorted(DATA_DIR.iterdir()):
        if not mission_dir.is_dir():
            continue
        images = list((mission_dir / "images" / "train").glob("*.jpg"))
        labels = list((mission_dir / "labels" / "train").glob("*.txt"))
        names = _load_classes(mission_dir)
        by_class = {n: 0 for n in names}
        for lbl in labels:
            for line in lbl.read_text().splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                cid = int(parts[0])
                if 0 <= cid < len(names):
                    by_class[names[cid]] += 1
        result["missions"][mission_dir.name] = {
            "images": len(images),
            "labels": len(labels),
            "classes": names,
            "boxes_by_class": by_class,
            "data_yaml": str(mission_dir / "data.yaml") if (mission_dir / "data.yaml").exists() else None,
        }
        result["total_images"] += len(images)

    # legacy flat folders (pre-YOLO) counted separately
    legacy = 0
    for mission_dir in DATA_DIR.iterdir():
        if not mission_dir.is_dir():
            continue
        if (mission_dir / "images").exists():
            continue
        legacy += sum(1 for _ in mission_dir.rglob("*.jpg"))
    result["legacy_images"] = legacy
    return result


@app.get("/api/training/missions/{mission}/classes")
async def mission_classes(mission: str):
    root = DATA_DIR / _safe_name(mission)
    if not root.exists():
        return {"mission": _safe_name(mission), "names": []}
    return {"mission": _safe_name(mission), "names": _load_classes(root)}


if __name__ == "__main__":
    print(f"YOLO training receiver → {DATA_DIR}")
    print("Layout: <mission>/images/train + labels/train + data.yaml")
    uvicorn.run(app, host="0.0.0.0", port=8787, log_level="info")
