from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from src.live_recognition import LiveRecognizer, PredictionStability


ROOT = Path(__file__).resolve().parents[1]


def test_prediction_stability_requires_consecutive_predictions_and_resets():
    stable = PredictionStability(required_frames=3)
    assert stable.update("HELLO") is None
    assert stable.update("HELLO") is None
    assert stable.update("GOOD MORNING") is None
    assert stable.update("GOOD MORNING") is None
    assert stable.update("GOOD MORNING") == "GOOD MORNING"
    assert stable.update("UNKNOWN") == "UNKNOWN"
    assert stable.update("GOOD MORNING") is None


def test_saved_recognizer_artifacts_and_existing_video_frame():
    recognizer = LiveRecognizer(ROOT)
    assert recognizer.method == "robust_scaled"
    assert recognizer.prototype_method == "mean"
    assert recognizer.max_hands == 2
    assert recognizer.threshold == recognizer.metadata["rejection_threshold"]
    assert len(recognizer.class_labels) == 11

    video_path = ROOT / "0" / "0.MOV"
    capture = cv2.VideoCapture(str(video_path))
    try:
        assert capture.isOpened(), f"Sample video could not be opened: {video_path}"
        outcome = None
        for _ in range(180):
            ok, frame = capture.read()
            if not ok:
                break
            outcome = recognizer.process_frame(frame)
            if outcome["status"] != "no_hand":
                break
        assert outcome is not None
        assert outcome["status"] not in {"no_hand", "invalid_hand"}, "Sample video had no usable hand in inspected frames"
        assert outcome["frame"].shape == frame.shape
        assert outcome["latency_ms"] >= 0
        if outcome["detected_hand_count"] == 1:
            assert outcome["raw_label"] in recognizer.class_labels
            assert outcome["distance"] >= 0
        else:
            assert outcome["status"] == "two_hand_unsupported"
            assert len(outcome["hands"]) == 2
    finally:
        capture.release()
        recognizer.close()


def test_translation_lookup_uses_csv_and_rejects_unsupported_labels():
    recognizer = LiveRecognizer(ROOT)
    try:
        mapping = recognizer.translation_for("GOOD MORNING")
        assert mapping == {"english": "GOOD MORNING", "cebuano": "REVIEW REQUIRED",
                           "review_status": "English from labels.csv; Cebuano unreviewed"}
        assert recognizer.translation_for("NOT A MODEL CLASS") is None
    finally:
        recognizer.close()


def test_missing_hand_status_and_resets_stability():
    recognizer = LiveRecognizer(ROOT)
    try:
        recognizer.stability.update("HELLO")
        recognizer.landmarker.close()
        recognizer.landmarker = SimpleNamespace(
            detect=lambda image: SimpleNamespace(hand_landmarks=[]), close=lambda: None)
        result = recognizer.process_frame(np.zeros((64, 64, 3), dtype=np.uint8))
        assert result["status"] == "no_hand"
        assert result["detection_status"] == "No hand detected"
        assert result["label"] == ""
        assert recognizer.stability.update("HELLO") is None
    finally:
        recognizer.close()


def test_unknown_is_immediate_and_does_not_show_a_translation(monkeypatch):
    recognizer = LiveRecognizer(ROOT, stability_frames=5)
    try:
        hand = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        recognizer.landmarker.close()
        recognizer.landmarker = SimpleNamespace(
            detect=lambda image: SimpleNamespace(hand_landmarks=[hand]), close=lambda: None)
        monkeypatch.setattr("src.live_recognition.make_fingerprint", lambda landmarks: np.full(70, 1e6))
        result = recognizer.process_frame(np.zeros((64, 64, 3), dtype=np.uint8))
        assert result["status"] == "unknown"
        assert result["label"] == "UNKNOWN"
        assert result["detection_status"].endswith("within threshold")
        assert recognizer.translation_for(result["label"]) is None
    finally:
        recognizer.close()


def test_two_hands_are_both_extracted_and_not_misrepresented_as_joint_recognition(monkeypatch):
    recognizer = LiveRecognizer(ROOT)
    try:
        hand = [SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(21)]
        handedness = [[SimpleNamespace(category_name="Left")],
                      [SimpleNamespace(category_name="Right")]]
        recognizer.landmarker.close()
        recognizer.landmarker = SimpleNamespace(
            detect=lambda image: SimpleNamespace(hand_landmarks=[hand, hand], handedness=handedness),
            close=lambda: None)
        extracted = []

        def fake_fingerprint(landmarks):
            extracted.append(landmarks)
            return recognizer.prototypes[0].copy() if len(extracted) == 1 else np.full(70, 1e6)

        monkeypatch.setattr("src.live_recognition.make_fingerprint", fake_fingerprint)
        result = recognizer.process_frame(np.zeros((96, 96, 3), dtype=np.uint8))
        assert len(extracted) == 2
        assert result["detected_hand_count"] == 2
        assert [item["side"] for item in result["hands"]] == ["Left", "Right"]
        assert result["status"] == "two_hand_unsupported"
        assert result["label"] == "UNKNOWN"
        assert "joint recognition is unsupported" in result["detection_status"]
    finally:
        recognizer.close()


def test_open_webcam_releases_unavailable_devices(monkeypatch):
    from src.webcam_app import open_webcam

    created = []

    class FakeCapture:
        def __init__(self, index, backend):
            self.index = index
            self.backend = backend
            self.released = False
            created.append(self)

        def isOpened(self):
            return self.index == 1

        def release(self):
            self.released = True

    monkeypatch.setattr("src.webcam_app.cv2.VideoCapture", FakeCapture)
    monkeypatch.setattr("src.webcam_app.sys.platform", "linux")
    capture, index = open_webcam(indexes=(0, 1))
    assert capture is created[-1]
    assert index == 1
    assert created[0].released


def test_phone_stream_url_opens_supported_stream(monkeypatch):
    from src.webcam_app import open_phone_stream

    class FakeCapture:
        opened = True
        released = False

        def __init__(self, url, backend):
            self.url = url
            self.backend = backend

        def isOpened(self):
            return self.opened

        def release(self):
            self.released = True

    captures = []

    def fake_capture(url, backend):
        item = FakeCapture(url, backend)
        captures.append(item)
        return item

    monkeypatch.setattr("src.webcam_app.cv2.VideoCapture", fake_capture)
    opened = open_phone_stream("  http://192.168.1.20:8080/video  ")
    assert opened is captures[0]
    assert captures[0].url == "http://192.168.1.20:8080/video"


def test_phone_stream_rejects_invalid_or_unavailable_url(monkeypatch):
    from src.webcam_app import open_phone_stream

    with pytest.raises(ValueError, match="URL first"):
        open_phone_stream(" ")
    with pytest.raises(ValueError, match="must start with"):
        open_phone_stream("192.168.1.20/video")

    class FailedCapture:
        released = False

        def isOpened(self):
            return False

        def release(self):
            self.released = True

    failed = FailedCapture()
    monkeypatch.setattr("src.webcam_app.cv2.VideoCapture", lambda url, backend: failed)
    assert open_phone_stream("rtsp://192.168.1.20/live") is None
    assert failed.released

