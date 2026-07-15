"""Tests for gate logic controller."""

from src.detector.models import Detection, TrackingResult
from src.gate_logic.controller import GateController
from src.gate_logic.events import EventType


def test_person_entry_detection():
    """Test that a person in entry zone generates an event."""
    controller = GateController()
    controller.initialize()

    detection = Detection(
        class_id=0,
        class_name="persona",
        confidence=0.85,
        bbox=(200, 300, 400, 450),
        track_id=1,
    )

    result = TrackingResult(frame_id=1, detections=[detection])
    events = controller.process_frame(result)

    assert len(events) >= 1
    assert any(e.event_type == EventType.PERSON_ENTRY for e in events)

    controller.shutdown()


def test_vehicle_entry_opens_gate():
    """Test that a vehicle in entry zone opens the gate."""
    controller = GateController()
    controller.initialize()

    detection = Detection(
        class_id=2,
        class_name="auto",
        confidence=0.90,
        bbox=(250, 300, 500, 450),
        track_id=10,
    )

    result = TrackingResult(frame_id=1, detections=[detection])
    controller.process_frame(result)

    assert controller.is_open

    controller.shutdown()


def test_zone_crowded_event():
    """Test crowded zone detection with >5 people."""
    controller = GateController()
    controller.initialize()

    detections = [
        Detection(
            class_id=0,
            class_name="persona",
            confidence=0.8,
            bbox=(i * 50, 300, i * 50 + 40, 400),
            track_id=i + 100,
        )
        for i in range(6)
    ]

    result = TrackingResult(frame_id=1, detections=detections)
    events = controller.process_frame(result)

    assert any(e.event_type == EventType.ZONE_CROWDED for e in events)

    controller.shutdown()
