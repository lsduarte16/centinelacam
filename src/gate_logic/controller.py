"""Gate controller with GPIO relay management and zone-based logic."""

import logging
import time
from collections import defaultdict
from threading import Lock, Timer

from src.config import settings
from src.detector.models import Detection, TrackingResult

from .events import EventType, GateEvent, Severity

logger = logging.getLogger(__name__)

try:
    from gpiozero import OutputDevice

    GPIO_AVAILABLE = True
except ImportError:
    GPIO_AVAILABLE = False
    logger.warning("gpiozero not available - running in simulation mode")


class GateController:
    """Controls physical gate relay based on detection events."""

    def __init__(self):
        uc = settings.active_use_case
        self.gpio_pin = uc.gpio_pin or 17
        self.open_duration = uc.open_duration
        self.cooldown = uc.cooldown
        self.entry_zone = uc.zones.entry
        self.exit_zone = uc.zones.exit

        self._relay = None
        self._is_open = False
        self._last_activation = 0.0
        self._lock = Lock()
        self._close_timer: Timer | None = None
        self._track_history: dict[int, list[str]] = defaultdict(list)
        self._event_callbacks: list = []

    def initialize(self):
        """Initialize GPIO relay."""
        if GPIO_AVAILABLE:
            self._relay = OutputDevice(self.gpio_pin, active_high=True, initial_value=False)
            logger.info("GPIO relay initialized on pin %d", self.gpio_pin)
        else:
            logger.info("Simulated relay on pin %d", self.gpio_pin)

    def process_frame(self, result: TrackingResult) -> list[GateEvent]:
        """Process a tracking result and generate gate events."""
        events = []

        for detection in result.detections:
            zone_events = self._check_zones(detection, result.frame_id)
            events.extend(zone_events)

        if result.person_count > 5:
            events.append(
                GateEvent(
                    event_type=EventType.ZONE_CROWDED,
                    severity=Severity.MEDIUM,
                    description=f"{result.person_count} personas detectadas en zona",
                    frame_id=result.frame_id,
                    metadata={"count": result.person_count},
                )
            )

        for event in events:
            self._notify(event)

        return events

    def _check_zones(self, detection: Detection, frame_id: int) -> list[GateEvent]:
        """Check if detection triggers zone-based events."""
        events = []
        track_id = detection.track_id

        if track_id is None:
            return events

        in_entry = detection.is_in_zone(self.entry_zone)
        in_exit = detection.is_in_zone(self.exit_zone)

        history = self._track_history[track_id]

        if in_entry and (not history or history[-1] != "entry"):
            history.append("entry")

            if detection.class_id == 0:
                event_type = EventType.PERSON_ENTRY
            else:
                event_type = EventType.VEHICLE_ENTRY

            events.append(
                GateEvent(
                    event_type=event_type,
                    severity=Severity.LOW,
                    description=f"{detection.class_name} ingresando (track={track_id})",
                    track_id=track_id,
                    zone="entry",
                    frame_id=frame_id,
                )
            )

            if event_type == EventType.VEHICLE_ENTRY:
                self.open_gate(reason=f"vehicle_entry track={track_id}")

        elif in_exit and (not history or history[-1] != "exit"):
            history.append("exit")

            if detection.class_id == 0:
                event_type = EventType.PERSON_EXIT
            else:
                event_type = EventType.VEHICLE_EXIT

            events.append(
                GateEvent(
                    event_type=event_type,
                    severity=Severity.LOW,
                    description=f"{detection.class_name} saliendo (track={track_id})",
                    track_id=track_id,
                    zone="exit",
                    frame_id=frame_id,
                )
            )

        # Cleanup old tracks
        if len(self._track_history) > 200:
            oldest = list(self._track_history.keys())[:100]
            for k in oldest:
                del self._track_history[k]

        return events

    def open_gate(self, reason: str = ""):
        """Open the gate relay."""
        with self._lock:
            now = time.time()
            if now - self._last_activation < self.cooldown:
                logger.debug("Gate open request ignored (cooldown)")
                return

            self._is_open = True
            self._last_activation = now

            if self._relay:
                self._relay.on()

            logger.info("Gate OPENED: %s", reason)

            if self._close_timer:
                self._close_timer.cancel()

            self._close_timer = Timer(self.open_duration, self.close_gate)
            self._close_timer.daemon = True
            self._close_timer.start()

            self._notify(
                GateEvent(
                    event_type=EventType.GATE_OPENED,
                    severity=Severity.LOW,
                    description=reason,
                )
            )

    def close_gate(self):
        """Close the gate relay."""
        with self._lock:
            self._is_open = False
            if self._relay:
                self._relay.off()
            logger.info("Gate CLOSED")
            self._notify(
                GateEvent(
                    event_type=EventType.GATE_CLOSED,
                    severity=Severity.LOW,
                )
            )

    def on_event(self, callback):
        """Register an event callback."""
        self._event_callbacks.append(callback)

    def _notify(self, event: GateEvent):
        """Notify all registered callbacks."""
        for cb in self._event_callbacks:
            try:
                cb(event)
            except Exception as e:
                logger.error("Event callback error: %s", e)

    @property
    def is_open(self) -> bool:
        return self._is_open

    def shutdown(self):
        """Clean shutdown."""
        if self._close_timer:
            self._close_timer.cancel()
        self.close_gate()
        if self._relay:
            self._relay.close()
