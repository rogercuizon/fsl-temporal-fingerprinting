"""MediaPipe Hands wrapper. Imports MediaPipe lazily so non-video tools work without it."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import cv2


@dataclass
class VideoExtraction:
    landmarks: list[list[tuple[float, float, float]]]
    sampled_frames: int
    processed_frames: int
    no_hand_frames: int
    error: str = ""
    timeline: list[dict] = field(default_factory=list)


class LandmarkExtractor:
    def __init__(self, max_hands: int = 1, sample_every: int = 5, min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5, visualize: bool = False,
                 model_path: str | Path | None = None):
        if sample_every < 1:
            raise ValueError("sample_every must be at least 1")
        self.max_hands, self.sample_every = max_hands, sample_every
        self.min_detection_confidence, self.min_tracking_confidence = min_detection_confidence, min_tracking_confidence
        self.visualize = visualize
        self.model_path = Path(model_path) if model_path else Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"

    def extract_video(self, path: str | Path, visualization_dir: str | Path | None = None) -> VideoExtraction:
        """Sample frames and return the first detected hand in each sampled frame."""
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is unavailable. Install requirements in a supported Python environment.") from exc
        if not self.model_path.is_file():
            raise RuntimeError(f"MediaPipe Hand Landmarker model is missing: {self.model_path}")
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            capture.release()
            return VideoExtraction([], 0, 0, 0, "Could not open video")
        found, sampled, processed, no_hand, frame_no = [], 0, 0, 0, 0
        timeline = []
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not fps or fps <= 0:
            fps = 30.0
        try:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO,
                num_hands=self.max_hands,
                min_hand_detection_confidence=self.min_detection_confidence,
                min_hand_presence_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence)
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as hands:
                while True:
                    ok, frame = capture.read()
                    if not ok:
                        break
                    current = frame_no
                    frame_no += 1
                    if current % self.sample_every:
                        continue
                    sampled += 1
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
                    timestamp_ms = round(current * 1000 / fps)
                    result = hands.detect_for_video(image, timestamp_ms)
                    hand_count = len(result.hand_landmarks or [])
                    timeline.append({"sample_index": sampled, "source_frame_number": current,
                                     "timestamp_ms": timestamp_ms, "detected_hand_count": hand_count,
                                     "hand_detected": bool(hand_count)})
                    if not hand_count:
                        no_hand += 1
                        continue
                    processed += 1
                    hand = result.hand_landmarks[0]
                    found.append([(p.x, p.y, p.z) for p in hand])
                    if self.visualize and visualization_dir:
                        outdir = Path(visualization_dir)
                        outdir.mkdir(parents=True, exist_ok=True)
                        height, width = frame.shape[:2]
                        points = [(round(p.x * width), round(p.y * height)) for p in hand]
                        for edge in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                            cv2.line(frame, points[edge.start], points[edge.end], (0, 255, 0), 2)
                        for x, y in points:
                            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)
                        cv2.imwrite(str(outdir / f"{Path(path).stem}_frame_{current:06d}.jpg"), frame)
        except Exception as exc:
            return VideoExtraction(found, sampled, processed, no_hand, f"Extraction error: {exc}", timeline)
        finally:
            capture.release()
        return VideoExtraction(found, sampled, processed, no_hand, timeline=timeline)
