"""System and pipeline performance metrics."""

from __future__ import annotations

import time
from collections import deque
from threading import Lock

try:
    import psutil
except ImportError:
    psutil = None  # type: ignore


class MetricsCollector:
    def __init__(self, window: int = 60):
        self._lock = Lock()
        self._frame_times: deque[float] = deque(maxlen=window)
        self._process_times: deque[float] = deque(maxlen=window)
        self._last_frame_ts: float | None = None
        self._frames_processed = 0
        self._training_uploads_ok = 0
        self._training_uploads_fail = 0

    def record_frame(self, process_ms: float):
        now = time.time()
        with self._lock:
            self._frame_times.append(now)
            self._process_times.append(process_ms)
            self._last_frame_ts = now
            self._frames_processed += 1

    def record_training_upload(self, success: bool):
        with self._lock:
            if success:
                self._training_uploads_ok += 1
            else:
                self._training_uploads_fail += 1

    def _fps(self) -> float:
        with self._lock:
            if len(self._frame_times) < 2:
                return 0.0
            span = self._frame_times[-1] - self._frame_times[0]
            if span <= 0:
                return 0.0
            return round((len(self._frame_times) - 1) / span, 2)

    def _avg_process_ms(self) -> float:
        with self._lock:
            if not self._process_times:
                return 0.0
            return round(sum(self._process_times) / len(self._process_times), 1)

    def snapshot(self) -> dict:
        cpu_pct = mem_pct = mem_used_mb = mem_total_mb = 0.0
        if psutil:
            cpu_pct = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            mem_pct = mem.percent
            mem_used_mb = round(mem.used / (1024 * 1024), 1)
            mem_total_mb = round(mem.total / (1024 * 1024), 1)

        with self._lock:
            last_frame_age = (
                round(time.time() - self._last_frame_ts, 2) if self._last_frame_ts else None
            )
            return {
                "cpu_percent": cpu_pct,
                "memory_percent": mem_pct,
                "memory_used_mb": mem_used_mb,
                "memory_total_mb": mem_total_mb,
                "pipeline_fps": self._fps(),
                "avg_process_ms": self._avg_process_ms(),
                "frames_processed": self._frames_processed,
                "last_frame_age_sec": last_frame_age,
                "training_uploads_ok": self._training_uploads_ok,
                "training_uploads_fail": self._training_uploads_fail,
            }


metrics_collector = MetricsCollector()
