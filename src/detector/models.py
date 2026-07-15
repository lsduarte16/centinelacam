"""Data models for detection results."""

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    track_id: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def center(self) -> tuple[int, int]:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.bbox
        return (x2 - x1) * (y2 - y1)

    def is_in_zone(self, zone: list[int]) -> bool:
        """Check if detection center is within a zone [x1, y1, x2, y2]."""
        cx, cy = self.center
        zx1, zy1, zx2, zy2 = zone
        return zx1 <= cx <= zx2 and zy1 <= cy <= zy2


@dataclass
class TrackingResult:
    frame_id: int
    detections: list[Detection]
    person_count: int = 0
    vehicle_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def __post_init__(self):
        self.person_count = sum(1 for d in self.detections if d.class_id == 0)
        self.vehicle_count = sum(1 for d in self.detections if d.class_id in (2, 7))
