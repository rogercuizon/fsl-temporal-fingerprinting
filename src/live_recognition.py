"""Live webcam inference using the project's frozen landmark matcher artifacts."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from .fingerprint import make_fingerprint
from .matching import fingerprint_distance


class LiveRecognizer:
    """Extract a 70-feature vector from each frame and match frozen prototypes."""

    def __init__(self, project_root: str | Path, stability_frames: int = 3):
        if stability_frames < 1:
            raise ValueError("stability_frames must be at least 1")
        self.root = Path(project_root).resolve()
        model_dir = self.root / "output" / "matching"
        metadata_path = model_dir / "selected_model_metadata.json"
        artifact_path = model_dir / "selected_model_artifacts.npz"
        landmark_model = self.root / "models" / "hand_landmarker.task"
        translations_path = self.root / "translations.csv"
        for path, description in ((metadata_path, "selected model metadata"),
                                  (artifact_path, "selected model artifacts"),
                                  (landmark_model, "MediaPipe hand model"),
                                  (translations_path, "translations.csv")):
            if not path.is_file():
                raise FileNotFoundError(f"Missing {description}: {path}")

        try:
            self.metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig"))
            self.artifacts = np.load(artifact_path, allow_pickle=False)
            self.class_labels = [str(v) for v in self.artifacts["class_labels"].tolist()]
            self.prototypes = self.artifacts["prototypes"]
            self.scale = self.artifacts["scaler_scale"]
            self.method = str(self.artifacts["distance_method"].item())
            self.prototype_method = str(self.artifacts["prototype_method"].item())
            self.threshold = float(self.artifacts["rejection_threshold"].item())
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(f"Could not load saved recognition model: {exc}") from exc
        expected_threshold = float(self.metadata["rejection_threshold"])
        if (self.method != "robust_scaled" or self.prototype_method != "mean"
                or not np.isclose(self.threshold, expected_threshold, rtol=0, atol=1e-10)
                or not np.isclose(self.threshold, 9.222295068261808, rtol=0, atol=1e-10)
                or self.prototypes.shape != (len(self.class_labels), 70)
                or self.scale.shape != (70,) or not np.isfinite(self.prototypes).all()
                or not np.isfinite(self.scale).all() or np.any(self.scale <= 0)):
            raise ValueError("Saved model does not match the approved mean / robust_scaled model and frozen threshold")

        self.translations = {}
        with translations_path.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream):
                label = row.get("gesture_label", "").strip()
                if label:
                    self.translations[label] = {
                        "english": row.get("english_output", "").strip(),
                        "cebuano": row.get("cebuano_output", "").strip(),
                        "review_status": row.get("review_status", "").strip(),
                    }
        missing = set(self.class_labels) - set(self.translations)
        if missing:
            raise ValueError(f"translations.csv is missing model classes: {', '.join(sorted(missing))}")

        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is unavailable. Activate the project .venv and install requirements.txt.") from exc
        if not hasattr(mp, "tasks"):
            raise RuntimeError("Installed MediaPipe does not provide the Tasks HandLandmarker API.")
        self.mp = mp
        self.max_hands = 2
        try:
            options = mp.tasks.vision.HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(landmark_model)),
                running_mode=mp.tasks.vision.RunningMode.IMAGE,
                num_hands=self.max_hands,
                min_hand_detection_confidence=0.5,
                min_hand_presence_confidence=0.5,
                min_tracking_confidence=0.5,
            )
            self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(options)
        except Exception as exc:
            raise RuntimeError(f"Could not initialize MediaPipe Hand Landmarker: {exc}") from exc
        self.stability = PredictionStability(stability_frames)

    def close(self):
        if getattr(self, "landmarker", None) is not None:
            self.landmarker.close()
            self.landmarker = None
        if getattr(self, "artifacts", None) is not None:
            self.artifacts.close()

    def translation_for(self, label: str) -> dict | None:
        """Return the predefined mapping only for supported gesture labels."""
        if label not in self.class_labels:
            return None
        return self.translations.get(label)

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        started = time.perf_counter()
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        result = self.landmarker.detect(image)
        hands = result.hand_landmarks or []
        annotated = frame_bgr.copy()
        if not hands:
            self.stability.reset()
            return {"status": "no_hand", "detection_status": "No hand detected",
                    "label": "", "raw_label": "", "distance": None,
                    "detected_hand_count": 0, "hands": [],
                    "latency_ms": (time.perf_counter() - started) * 1000.0, "frame": annotated}
        h, w = annotated.shape[:2]
        connections = self.mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
        hand_results = []
        colors = ((0, 220, 0), (255, 160, 0))
        for index, hand in enumerate(hands):
            side = self._hand_side(result, index)
            color = colors[index % len(colors)]
            for edge in connections:
                a, b = hand[edge.start], hand[edge.end]
                cv2.line(annotated, (round(a.x * w), round(a.y * h)),
                         (round(b.x * w), round(b.y * h)), color, 2)
            for point in hand:
                cv2.circle(annotated, (round(point.x * w), round(point.y * h)), 3, (0, 80, 255), -1)
            try:
                fingerprint = make_fingerprint([(p.x, p.y, p.z) for p in hand])
            except (ValueError, FloatingPointError) as exc:
                hand_results.append({"side": side, "status": "invalid", "error": str(exc),
                                     "candidate_label": "UNKNOWN", "distance": None})
                continue
            scores = [(label, fingerprint_distance(fingerprint, self.prototypes[i], self.method, self.scale))
                      for i, label in enumerate(self.class_labels)]
            candidate, distance = min(scores, key=lambda item: (item[1], item[0]))
            accepted = distance <= self.threshold
            hand_results.append({"side": side, "status": "matched" if accepted else "unknown",
                                 "candidate_label": candidate if accepted else "UNKNOWN",
                                 "nearest_label": candidate, "distance": float(distance), "accepted": accepted})

        hand_count = len(hands)
        if hand_count >= 2:
            # The saved prototypes describe one 70-value hand at a time. Per-hand
            # matches are diagnostic only; they cannot identify the joint gesture.
            self.stability.reset()
            side_text = ", ".join(item["side"] for item in hand_results)
            return {"status": "two_hand_unsupported", "detection_status": f"2 hands detected ({side_text}); joint recognition is unsupported by the single-hand model",
                    "label": "UNKNOWN", "raw_label": "", "accepted": False, "distance": None,
                    "detected_hand_count": hand_count, "hands": hand_results,
                    "latency_ms": (time.perf_counter() - started) * 1000.0, "frame": annotated}

        hand_result = hand_results[0]
        if hand_result["status"] == "invalid":
            self.stability.reset()
            return {"status": "invalid_hand", "detection_status": "1 hand detected; landmarks unusable",
                    "label": "", "raw_label": "", "error": hand_result["error"],
                    "distance": None, "detected_hand_count": 1, "hands": hand_results,
                    "latency_ms": (time.perf_counter() - started) * 1000.0, "frame": annotated}
        raw_label = hand_result["nearest_label"]
        accepted = hand_result["accepted"]
        stable_label = self.stability.update(hand_result["candidate_label"] if accepted else "UNKNOWN")
        if stable_label == "UNKNOWN":
            status, label = "unknown", "UNKNOWN"
            detection_status = f"1 hand detected ({hand_result['side']}); no supported class within threshold"
        elif stable_label is None:
            status, label = "stabilizing", ""
            detection_status = f"1 hand detected ({hand_result['side']}); stabilizing prediction"
        else:
            status, label = "recognized", stable_label
            detection_status = f"1 hand detected ({hand_result['side']})"
        return {"status": status, "detection_status": detection_status,
                "label": label, "raw_label": raw_label, "accepted": accepted,
                "distance": hand_result["distance"], "detected_hand_count": 1, "hands": hand_results,
                "latency_ms": (time.perf_counter() - started) * 1000.0, "frame": annotated}

    @staticmethod
    def _hand_side(result, index: int) -> str:
        handedness = getattr(result, "handedness", None) or []
        if index < len(handedness) and handedness[index]:
            category = handedness[index][0]
            name = getattr(category, "category_name", "")
            if name in {"Left", "Right"}:
                return name
        return f"Hand {index + 1}"


class PredictionStability:
    """Require the same accepted label for N successive detected frames."""

    def __init__(self, required_frames: int = 3):
        if required_frames < 1:
            raise ValueError("required_frames must be at least 1")
        self.required_frames = required_frames
        self._last = None
        self._count = 0

    def update(self, label: str) -> str | None:
        if label == "UNKNOWN":
            self.reset()
            return "UNKNOWN"
        if label == self._last:
            self._count += 1
        else:
            self._last, self._count = label, 1
        return label if self._count >= self.required_frames else None

    def reset(self):
        self._last, self._count = None, 0
