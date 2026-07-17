"""Main processing pipeline - orchestrates all components based on active use case."""

import logging
import logging.handlers
import signal
import sys
import time
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import uvicorn

from src.api.server import create_app
from src.camera.capture import CameraStream
from src.config import settings
from src.detector.yolo_detector import YOLODetector
from src.gate_logic.controller import GateController
from src.gate_logic.events import EventType, GateEvent, Severity
from src.llm_engine.analyzer import LLMAnalyzer
from src.metrics.collector import metrics_collector
from src.notifications.telegram import TelegramNotifier
from src.runtime.store import runtime_config
from src.storage.database import EventDatabase
from src.storage.sync import CloudSync
from src.training.uploader import TrainingUploader

logger = logging.getLogger(__name__)


class Pipeline:
    """Main application pipeline driven by runtime config."""

    def __init__(self):
        self.use_case = settings.use_case
        self.uc_config = settings.active_use_case
        self.camera = CameraStream()
        self.detector = YOLODetector()
        self.gate = GateController()
        self.llm = LLMAnalyzer()
        self.db = EventDatabase()
        self.cloud = CloudSync()
        self.telegram = TelegramNotifier()
        self.training = TrainingUploader()
        self._running = False
        self._last_violation_time = 0.0
        self._zone_detections: list = []
        self._yolo_detections: list = []
        self._last_watch_notify: dict[str, float] = {}

    def initialize(self):
        Path(settings.storage.local_path).mkdir(parents=True, exist_ok=True)
        Path(settings.storage.local_path, "logs").mkdir(parents=True, exist_ok=True)

        self._setup_logging()
        runtime_config.get()  # ensure file exists

        logger.info(
            "Initializing CentinelaCam v0.2.0 | node=%s | use_case=%s",
            settings.node.id,
            self.use_case,
        )

        self.db.initialize()
        self.detector.classes = self.uc_config.classes
        self.detector.load()
        self.detector.warmup()

        if self.use_case == "gate_control":
            if self.uc_config.gpio_pin:
                self.gate.gpio_pin = self.uc_config.gpio_pin
            self.gate.open_duration = self.uc_config.open_duration
            self.gate.cooldown = self.uc_config.cooldown
            self.gate.entry_zone = self.uc_config.zones.entry
            self.gate.exit_zone = self.uc_config.zones.exit
            self.gate.initialize()
            self.gate.on_event(self._on_gate_event)

        self.telegram.set_camera(self.camera)
        if self.telegram.health_check():
            logger.info("Telegram bot connected")
        elif settings.telegram.enabled:
            logger.warning("Telegram bot token invalid or not configured")

        if self.llm.health_check():
            logger.info("Ollama LLM available: %s", settings.llm.model)
        else:
            logger.warning("Ollama not reachable - LLM analysis disabled")

        self.cloud.start_periodic()
        rc = runtime_config.data
        logger.info(
            "Ready | mode=%s fps=%d skip=%d telegram=%s",
            rc.mission.mode,
            rc.processing.fps,
            rc.processing.skip_frames,
            rc.telegram.enabled,
        )

    def _setup_logging(self):
        log_file = Path(settings.logging.file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        logging.basicConfig(
            level=getattr(logging, settings.logging.level),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=settings.logging.max_size_mb * 1024 * 1024,
                    backupCount=settings.logging.backup_count,
                ),
            ],
        )

    def _on_gate_event(self, event):
        self.db.insert_event(event)
        if event.event_type.value in self.uc_config.events:
            self.telegram.notify(
                event_type=event.event_type.value,
                message=event.description,
                frame=self.camera.frame.copy() if self.camera.frame is not None else None,
            )
        analysis = self.llm.analyze_event_sync(event)
        if analysis:
            logger.info("LLM Analysis: %s", analysis)

    def run(self):
        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        self.camera.start()

        api_app = create_app(
            db=self.db,
            gate_controller=self.gate,
            camera=self.camera,
            detector=self.detector,
            pipeline=self,
            training_uploader=self.training,
        )
        config = uvicorn.Config(
            api_app,
            host=settings.api.host,
            port=settings.api.port,
            log_level="warning",
            # uvloop/httptools hang on this ARM64 build; use portable stack
            loop="asyncio",
            http="h11",
        )
        server = uvicorn.Server(config)
        # Disable signal handlers so uvicorn can serve from a worker thread
        server.install_signal_handlers = lambda: None
        api_thread = Thread(target=server.run, daemon=True)
        api_thread.start()
        logger.info("API server started on %s:%d", settings.api.host, settings.api.port)

        try:
            self._process_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _process_loop(self):
        frame_count = 0
        for frame in self.camera.frames():
            if not self._running:
                break

            rc = runtime_config.data
            skip_frames = rc.processing.skip_frames
            target_fps = rc.processing.fps
            frame_interval = 1.0 / max(target_fps, 1)

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            start = time.time()
            use_case = runtime_config.effective_use_case()

            if runtime_config.is_training_mode():
                self._process_training(frame)
            elif use_case == "gate_control":
                self._process_gate_control(frame, frame_count)
            elif use_case == "people_counter":
                self._process_people_counter(frame, frame_count)
            elif use_case == "package_counter":
                self._process_package_counter(frame, frame_count)
            elif use_case == "barcode_reader":
                self._process_barcode(frame, frame_count)
            elif use_case == "sorter_monitor":
                self._process_sorter(frame, frame_count)
            elif use_case == "zone_violation":
                self._process_zone_violation(frame, frame_count)
            elif use_case == "object_watch":
                self._process_object_watch(frame, frame_count)

            elapsed_ms = (time.time() - start) * 1000
            metrics_collector.record_frame(elapsed_ms)

            sleep_time = max(0, frame_interval - (time.time() - start))
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_training(self, frame):
        """Training mode: only capture and upload labeled frames."""
        self._zone_detections = []
        self._yolo_detections = []
        self.training.maybe_auto_capture(frame)

    def _process_object_watch(self, frame, frame_count):
        """Detect selected YOLO/COCO classes and optionally notify."""
        rc = runtime_config.data
        watch = rc.detector.watch_classes or [0]
        self.detector.classes = watch
        self._zone_detections = []

        result = self.detector.detect(frame)
        self._yolo_detections = [
            {
                "bbox": det.bbox,
                "class_name": det.class_name,
                "confidence": det.confidence,
                "track_id": det.track_id,
            }
            for det in result.detections
        ]

        if not rc.detector.notify_on_detect or not result.detections:
            return

        cooldown = rc.telegram.cooldown
        now = time.time()
        for det in result.detections:
            key = f"{det.class_name}:{det.track_id or 'na'}"
            last = self._last_watch_notify.get(key, 0)
            if now - last < cooldown:
                continue
            self._last_watch_notify[key] = now
            logger.info(
                "OBJECT WATCH: %s conf=%.0f%% track=%s",
                det.class_name, det.confidence * 100, det.track_id,
            )
            self.telegram.notify(
                event_type="object_detected",
                message=(
                    f"Detectado: {det.class_name} "
                    f"({det.confidence:.0%})"
                    + (f" #{det.track_id}" if det.track_id else "")
                ),
                frame=frame,
            )
            self.db.insert_event(
                GateEvent(
                    event_type=EventType.ANOMALY_DETECTED,
                    severity=Severity.MEDIUM,
                    description=f"Detectado {det.class_name}",
                    track_id=det.track_id or 0,
                    frame_id=frame_count,
                    metadata={
                        "class": det.class_name,
                        "class_id": det.class_id,
                        "confidence": round(det.confidence, 3),
                    },
                )
            )

    def _process_gate_control(self, frame, frame_count):
        result = self.detector.detect(frame)
        if result.detections:
            self.gate.process_frame(result)

    def _process_people_counter(self, frame, frame_count):
        result = self.detector.detect(frame)
        if result.detections:
            events = self.gate.process_frame(result)
            for event in events:
                if event.event_type.value in self.uc_config.events:
                    self.telegram.notify(event_type=event.event_type.value, message=event.description)

    def _process_package_counter(self, frame, frame_count):
        result = self.detector.detect(frame)
        if result.detections:
            events = self.gate.process_frame(result)
            for event in events:
                if event.event_type.value in self.uc_config.events:
                    self.telegram.notify(
                        event_type=event.event_type.value,
                        message=f"Paquete detectado (track={event.track_id})",
                    )

    def _process_barcode(self, frame, frame_count):
        try:
            from pyzbar.pyzbar import decode
        except ImportError:
            return
        region = self.uc_config.scan_region
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]
        for barcode in decode(crop):
            data = barcode.data.decode("utf-8")
            self.telegram.notify(
                event_type="barcode_read",
                message=f"Código: `{data}` ({barcode.type})",
                frame=frame,
            )

    def _process_sorter(self, frame, frame_count):
        result = self.detector.detect(frame)
        if result.detections:
            events = self.gate.process_frame(result)
            for event in events:
                if event.event_type.value in self.uc_config.events:
                    self.telegram.notify(
                        event_type=event.event_type.value,
                        message=event.description,
                        frame=frame,
                    )

    def _process_zone_violation(self, frame, frame_count):
        rc = runtime_config.data
        safe = rc.zones.safe_zone
        sx1, sy1, sx2, sy2 = safe
        roi = rc.zones.paper_roi
        rx1, ry1, rx2, ry2 = roi

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]
        roi_mask = np.zeros(saturation.shape, dtype=np.uint8)
        roi_mask[ry1:ry2, rx1:rx2] = 255

        thresh = rc.detector.saturation_threshold
        _, sat_mask = cv2.threshold(saturation, thresh, 255, cv2.THRESH_BINARY)
        sat_mask = cv2.bitwise_and(sat_mask, roi_mask)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        min_area = rc.detector.min_object_area
        cooldown = rc.telegram.cooldown
        now = time.time()
        self._zone_detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = x + bw // 2, y + bh // 2
            inside = sx1 <= cx <= sx2 and sy1 <= cy <= sy2
            self._zone_detections.append((x, y, bw, bh, inside))

            if not inside and (now - self._last_violation_time >= cooldown):
                self._last_violation_time = now
                logger.info("ZONE VIOLATION at (%d,%d) area=%d", cx, cy, area)
                self.telegram.notify(
                    event_type="zone_violation",
                    message=f"Objeto FUERA de zona segura (pos: {cx},{cy})",
                    frame=frame,
                )
                self.db.insert_event(
                    GateEvent(
                        event_type=EventType.ANOMALY_DETECTED,
                        severity=Severity.HIGH,
                        description="Objeto fuera de zona segura",
                        track_id=0,
                        frame_id=frame_count,
                        metadata={"cx": cx, "cy": cy, "area": int(area)},
                    )
                )

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def shutdown(self):
        logger.info("Shutting down pipeline...")
        self._running = False
        self.camera.stop()
        self.gate.shutdown()
        self.llm.shutdown()
        self.telegram.shutdown()
        self.training.shutdown()
        self.cloud.stop()
        self.db.cleanup_old_records()
        self.db.close()
        logger.info("Pipeline shutdown complete")


def main():
    pipeline = Pipeline()
    pipeline.initialize()
    pipeline.run()


if __name__ == "__main__":
    main()
