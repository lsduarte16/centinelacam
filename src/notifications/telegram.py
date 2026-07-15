"""Telegram bot notifications with snapshot evidence."""

import io
import logging
import time
from threading import Thread

import cv2
import httpx
import numpy as np

from src.config import settings

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Sends event notifications to Telegram with optional snapshot."""

    def __init__(self):
        self.enabled = settings.telegram.enabled
        self.bot_token = settings.telegram.bot_token
        self.chat_id = settings.telegram.chat_id
        self.send_snapshot = settings.telegram.send_snapshot
        self.cooldown = settings.telegram.cooldown
        self._base_url = f"https://api.telegram.org/bot{self.bot_token}"
        self._client = httpx.Client(timeout=15)
        self._last_sent: dict[str, float] = {}
        self._camera = None

    def set_camera(self, camera):
        """Set camera reference for snapshot capture."""
        self._camera = camera

    def notify(self, event_type: str, message: str, frame: np.ndarray | None = None):
        """Send notification if cooldown has passed."""
        if not self.enabled or not self.bot_token or not self.chat_id:
            return

        now = time.time()
        last = self._last_sent.get(event_type, 0)
        if now - last < self.cooldown:
            return

        self._last_sent[event_type] = now

        Thread(target=self._send, args=(event_type, message, frame), daemon=True).start()

    def _send(self, event_type: str, message: str, frame: np.ndarray | None):
        """Send message (and optional photo) to Telegram."""
        node = settings.node
        header = f"🚨 *{event_type.upper().replace('_', ' ')}*\n"
        body = (
            f"📍 {node.location} (`{node.id}`)\n"
            f"📋 {message}\n"
            f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')}"
        )
        full_text = header + body

        try:
            if self.send_snapshot and frame is not None:
                self._send_photo(full_text, frame)
            elif self.send_snapshot and self._camera and self._camera.frame is not None:
                self._send_photo(full_text, self._camera.frame.copy())
            else:
                self._send_text(full_text)
        except Exception as e:
            logger.error("Telegram notification failed: %s", e)

    def _send_text(self, text: str):
        resp = self._client.post(
            f"{self._base_url}/sendMessage",
            json={"chat_id": self.chat_id, "text": text, "parse_mode": "Markdown"},
        )
        if resp.status_code != 200:
            logger.warning("Telegram sendMessage failed: %s", resp.text)

    def _send_photo(self, caption: str, frame: np.ndarray):
        _, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        photo_bytes = io.BytesIO(jpeg.tobytes())
        photo_bytes.name = "snapshot.jpg"

        resp = self._client.post(
            f"{self._base_url}/sendPhoto",
            data={"chat_id": self.chat_id, "caption": caption, "parse_mode": "Markdown"},
            files={"photo": photo_bytes},
        )
        if resp.status_code != 200:
            logger.warning("Telegram sendPhoto failed: %s", resp.text)

    def health_check(self) -> bool:
        """Verify bot token is valid."""
        if not self.bot_token:
            return False
        try:
            resp = self._client.get(f"{self._base_url}/getMe")
            return resp.status_code == 200
        except Exception:
            return False

    def shutdown(self):
        self._client.close()
