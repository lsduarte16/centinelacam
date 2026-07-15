"""Event definitions for gate control system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    PERSON_ENTRY = "person_entry"
    PERSON_EXIT = "person_exit"
    VEHICLE_ENTRY = "vehicle_entry"
    VEHICLE_EXIT = "vehicle_exit"
    GATE_OPENED = "gate_opened"
    GATE_CLOSED = "gate_closed"
    UNAUTHORIZED_ACCESS = "unauthorized_access"
    ZONE_CROWDED = "zone_crowded"
    ANOMALY_DETECTED = "anomaly_detected"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class GateEvent:
    event_type: EventType
    severity: Severity = Severity.LOW
    description: str = ""
    track_id: int | None = None
    zone: str = ""
    frame_id: int = 0
    metadata: dict = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "track_id": self.track_id,
            "zone": self.zone,
            "frame_id": self.frame_id,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }
