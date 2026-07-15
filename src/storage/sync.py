"""Cloud sync using rclone for backup."""

import logging
import subprocess
from pathlib import Path
from threading import Thread

from src.config import settings

logger = logging.getLogger(__name__)


class CloudSync:
    """Syncs local storage to cloud using rclone."""

    def __init__(self):
        self.enabled = settings.storage.cloud_sync.enabled
        self.provider = settings.storage.cloud_sync.provider
        self.bucket = settings.storage.cloud_sync.bucket
        self.local_path = Path(settings.storage.local_path)
        self.interval = settings.storage.cloud_sync.sync_interval
        self._running = False

    def sync_now(self) -> bool:
        """Run a sync operation."""
        if not self.enabled:
            logger.debug("Cloud sync disabled")
            return False

        remote = f"{self.provider}:{self.bucket}"
        cmd = [
            "rclone",
            "sync",
            str(self.local_path),
            remote,
            "--transfers", "4",
            "--checkers", "8",
            "--log-level", "INFO",
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                logger.info("Cloud sync completed: %s -> %s", self.local_path, remote)
                return True
            else:
                logger.error("Cloud sync failed: %s", result.stderr)
                return False
        except subprocess.TimeoutExpired:
            logger.error("Cloud sync timed out")
            return False
        except FileNotFoundError:
            logger.error("rclone not installed")
            return False

    def start_periodic(self):
        """Start periodic sync in background."""
        if not self.enabled:
            return

        import schedule
        import time

        self._running = True
        schedule.every(self.interval).seconds.do(self.sync_now)

        def _run():
            while self._running:
                schedule.run_pending()
                time.sleep(60)

        thread = Thread(target=_run, daemon=True)
        thread.start()
        logger.info("Periodic cloud sync started (interval=%ds)", self.interval)

    def stop(self):
        self._running = False
