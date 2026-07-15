"""Main processing pipeline - orchestrates all components."""

import logging
import logging.handlers
import signal
import sys
import time
from pathlib import Path
from threading import Thread

import uvicorn

from src.api.server import create_app
from src.camera.capture import CameraStream
from src.config import settings
from src.detector.yolo_detector import YOLODetector
from src.gate_logic.controller import GateController
from src.llm_engine.analyzer import LLMAnalyzer
from src.storage.database import EventDatabase
from src.storage.sync import CloudSync

logger = logging.getLogger(__name__)


class Pipeline:
    """Main application pipeline that ties all components together."""

    def __init__(self):
        self.camera = CameraStream()
        self.detector = YOLODetector()
        self.gate = GateController()
        self.llm = LLMAnalyzer()
        self.db = EventDatabase()
        self.cloud = CloudSync()
        self._running = False

    def initialize(self):
        """Initialize all components."""
        Path(settings.storage.local_path).mkdir(parents=True, exist_ok=True)
        Path(settings.storage.local_path, "logs").mkdir(parents=True, exist_ok=True)
        Path(settings.storage.local_path, "snapshots").mkdir(parents=True, exist_ok=True)

        self._setup_logging()

        logger.info("Initializing CAM-PI Gate Controller v0.1.0")

        self.db.initialize()
        self.detector.load()
        self.detector.warmup()
        self.gate.initialize()

        self.gate.on_event(self._on_gate_event)

        if self.llm.health_check():
            logger.info("Ollama LLM available: %s", settings.llm.model)
        else:
            logger.warning("Ollama not reachable - LLM analysis disabled")

        self.cloud.start_periodic()

        logger.info("All components initialized")

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
        """Handle gate events - store and analyze."""
        self.db.insert_event(event)

        analysis = self.llm.analyze_event_sync(event)
        if analysis:
            logger.info("LLM Analysis: %s", analysis)

    def run(self):
        """Start the main processing loop."""
        self._running = True

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        self.camera.start()

        api_app = create_app(db=self.db, gate_controller=self.gate)
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
        """Main frame processing loop."""
        frame_count = 0
        skip_frames = 2  # Process every Nth frame for performance

        for frame in self.camera.frames():
            if not self._running:
                break

            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            result = self.detector.detect(frame)

            if result.detections:
                events = self.gate.process_frame(result)
                if events:
                    logger.debug(
                        "Frame %d: %d detections, %d events",
                        frame_count,
                        len(result.detections),
                        len(events),
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
