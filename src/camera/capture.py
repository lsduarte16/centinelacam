"""RTSP camera stream capture with reconnection logic."""

import logging
import time
from collections.abc import Generator
from threading import Event, Thread

import cv2
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages RTSP connection to dome camera with auto-reconnect."""

    def __init__(self):
        self.url = settings.camera.rtsp_url
        self.fps = settings.camera.fps
        self.resolution = tuple(settings.camera.resolution)
        self._cap: cv2.VideoCapture | None = None
        self._frame: np.ndarray | None = None
        self._running = Event()
        self._thread: Thread | None = None

    def start(self) -> "CameraStream":
        """Start the capture thread."""
        self._running.set()
        self._thread = Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        logger.info("Camera stream started: %s", self.url)
        return self

    def stop(self):
        """Stop the capture thread."""
        self._running.clear()
        if self._thread:
            self._thread.join(timeout=5)
        if self._cap:
            self._cap.release()
        logger.info("Camera stream stopped")

    def _connect(self) -> bool:
        """Establish connection to camera."""
        if self._cap:
            self._cap.release()

        self._cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if not self._cap.isOpened():
            logger.warning("Failed to connect to camera: %s", self.url)
            return False

        logger.info("Connected to camera: %s", self.url)
        return True

    def _capture_loop(self):
        """Main capture loop with reconnection."""
        frame_interval = 1.0 / self.fps

        while self._running.is_set():
            if self._cap is None or not self._cap.isOpened():
                if not self._connect():
                    time.sleep(settings.camera.reconnect_delay)
                    continue

            start = time.time()
            ret, frame = self._cap.read()

            if not ret:
                logger.warning("Frame read failed, reconnecting...")
                time.sleep(settings.camera.reconnect_delay)
                self._connect()
                continue

            if self.resolution:
                frame = cv2.resize(frame, self.resolution)

            self._frame = frame

            elapsed = time.time() - start
            sleep_time = max(0, frame_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)

    @property
    def frame(self) -> np.ndarray | None:
        """Get the latest frame."""
        return self._frame

    def frames(self) -> Generator[np.ndarray, None, None]:
        """Generator that yields frames continuously."""
        while self._running.is_set():
            if self._frame is not None:
                yield self._frame.copy()
            else:
                time.sleep(0.01)

    @property
    def is_running(self) -> bool:
        return self._running.is_set()
