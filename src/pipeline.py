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
from src.notifications.telegram import TelegramNotifier
from src.storage.database import EventDatabase
from src.storage.sync import CloudSync

logger = logging.getLogger(__name__)


class Pipeline:
    """Main application pipeline driven by the active use case."""

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
        self._running = False
        self._last_violation_time = 0.0
        self._violation_cooldown = 10.0  # seconds between notifications
        self._zone_detections = []  # shared with video overlay

    def initialize(self):
        """Initialize all components based on active use case."""
        Path(settings.storage.local_path).mkdir(parents=True, exist_ok=True)
        Path(settings.storage.local_path, "logs").mkdir(parents=True, exist_ok=True)
        Path(settings.storage.local_path, "snapshots").mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        logger.info(
            "Initializing CentinelaCam v0.1.0 | node=%s | use_case=%s",
            settings.node.id,
            self.use_case,
        )
        logger.info("Use case: %s", self.uc_config.description)

        self.db.initialize()

        # Configure detector with use-case specific classes and resolution
        self.detector.classes = self.uc_config.classes
        self.detector.load()
        self.detector.warmup()

        # Gate controller only for gate_control use case
        if self.use_case == "gate_control":
            if self.uc_config.gpio_pin:
                self.gate.gpio_pin = self.uc_config.gpio_pin
            self.gate.open_duration = self.uc_config.open_duration
            self.gate.cooldown = self.uc_config.cooldown
            self.gate.entry_zone = self.uc_config.zones.entry
            self.gate.exit_zone = self.uc_config.zones.exit
            self.gate.initialize()
            self.gate.on_event(self._on_gate_event)

        # Telegram notifications
        self.telegram.set_camera(self.camera)
        if self.telegram.health_check():
            logger.info("Telegram bot connected")
        elif settings.telegram.enabled:
            logger.warning("Telegram bot token invalid or not configured")

        # LLM
        if self.llm.health_check():
            logger.info("Ollama LLM available: %s", settings.llm.model)
        else:
            logger.warning("Ollama not reachable - LLM analysis disabled")

        self.cloud.start_periodic()
        logger.info(
            "All components initialized | fps=%d skip=%d imgsz=%d",
            self.uc_config.fps,
            self.uc_config.skip_frames,
            settings.detector.imgsz,
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
        """Handle gate events - store, analyze, notify."""
        self.db.insert_event(event)

        # Notify via Telegram for configured event types
        if event.event_type.value in self.uc_config.events:
            self.telegram.notify(
                event_type=event.event_type.value,
                message=event.description,
                frame=self.camera.frame.copy() if self.camera.frame is not None else None,
            )

        # LLM contextual analysis
        analysis = self.llm.analyze_event_sync(event)
        if analysis:
            logger.info("LLM Analysis: %s", analysis)

    def run(self):
        """Start the main processing loop."""
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
        )
        api_thread = Thread(
            target=uvicorn.run,
            args=(api_app,),
            kwargs={"host": settings.api.host, "port": settings.api.port, "log_level": "warning"},
            daemon=True,
        )
        api_thread.start()
        logger.info("API server started on %s:%d", settings.api.host, settings.api.port)
        logger.info("Processing pipeline started - waiting for frames...")

        try:
            self._process_loop()
        except KeyboardInterrupt:
            pass
        finally:
            self.shutdown()

    def _process_loop(self):
        """Main frame processing loop with adaptive FPS."""
        frame_count = 0
        skip_frames = self.uc_config.skip_frames
        target_fps = self.uc_config.fps
        frame_interval = 1.0 / target_fps

        for frame in self.camera.frames():
            if not self._running:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            start = time.time()

            if self.use_case == "gate_control":
                self._process_gate_control(frame, frame_count)
            elif self.use_case == "people_counter":
                self._process_people_counter(frame, frame_count)
            elif self.use_case == "package_counter":
                self._process_package_counter(frame, frame_count)
            elif self.use_case == "barcode_reader":
                self._process_barcode(frame, frame_count)
            elif self.use_case == "sorter_monitor":
                self._process_sorter(frame, frame_count)
            elif self.use_case == "zone_violation":
                self._process_zone_violation(frame, frame_count)

            elapsed = time.time() - start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _process_gate_control(self, frame, frame_count):
        """Gate control: detect vehicles/persons and control relay."""
        result = self.detector.detect(frame)
        if result.detections:
            self.gate.process_frame(result)

    def _process_people_counter(self, frame, frame_count):
        """People counter: track persons crossing a line."""
        result = self.detector.detect(frame)
        if result.detections:
            events = self.gate.process_frame(result)
            for event in events:
                if event.event_type.value in self.uc_config.events:
                    self.telegram.notify(
                        event_type=event.event_type.value,
                        message=event.description,
                    )

    def _process_package_counter(self, frame, frame_count):
        """Package counter: count objects crossing detection zone."""
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
        """Barcode reader: scan region for barcodes/QR."""
        try:
            from pyzbar.pyzbar import decode
        except ImportError:
            return

        region = self.uc_config.scan_region
        x1, y1, x2, y2 = region
        crop = frame[y1:y2, x1:x2]

        barcodes = decode(crop)
        for barcode in barcodes:
            data = barcode.data.decode("utf-8")
            logger.info("Barcode read: %s (%s)", data, barcode.type)
            self.telegram.notify(
                event_type="barcode_read",
                message=f"Código: `{data}` ({barcode.type})",
                frame=frame,
            )

    def _process_sorter(self, frame, frame_count):
        """Sorter monitor: detect jams or fallen objects."""
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
        """Zone violation: detect COLORED objects outside safe zone.

        Uses HSV saturation channel to find colored objects (high saturation)
        while ignoring black pen lines (zero saturation) and white paper.
        """
        safe = self.uc_config.zones.safe_zone
        sx1, sy1, sx2, sy2 = safe

        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1]

        # High saturation = colored object (Stitch is blue = high saturation)
        # Black pen lines and white paper both have LOW saturation
        _, sat_mask = cv2.threshold(saturation, 60, 255, cv2.THRESH_BINARY)

        # Clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_OPEN, kernel)
        sat_mask = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, kernel)

        # Find all colored objects in the full frame
        contours, _ = cv2.findContours(
            sat_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        min_area = 500
        now = time.time()
        self._zone_detections = []  # store for video overlay

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            cx, cy = x + bw // 2, y + bh // 2
            inside = sx1 <= cx <= sx2 and sy1 <= cy <= sy2

            self._zone_detections.append((x, y, bw, bh, inside))

            if not inside and (now - self._last_violation_time >= self._violation_cooldown):
                self._last_violation_time = now
                logger.info(
                    "ZONE VIOLATION: colored object at (%d,%d) area=%d outside safe zone",
                    cx, cy, area,
                )
                self.telegram.notify(
                    event_type="zone_violation",
                    message=(
                        f"Objeto detectado FUERA de la zona segura "
                        f"(posición: {cx},{cy}, área: {area}px)"
                    ),
                    frame=frame,
                )
                self.db.insert_event(
                    GateEvent(
                        event_type=EventType.ANOMALY_DETECTED,
                        severity=Severity.HIGH,
                        description="Objeto fuera de zona segura",
                        track_id=0,
                        frame_id=frame_count,
                        metadata={"cx": cx, "cy": cy, "area": area},
                    )
                )

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self._running = False

    def shutdown(self):
        """Graceful shutdown of all components."""
        logger.info("Shutting down pipeline...")
        self._running = False
        self.camera.stop()
        self.gate.shutdown()
        self.llm.shutdown()
        self.telegram.shutdown()
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
