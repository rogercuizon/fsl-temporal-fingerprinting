"""Video-level two-hand temporal landmark fingerprints and DTW matching.

This is a separate research path from the existing 70-feature static model.
It uses MediaPipe hand landmarks only; no body pose landmarks are used.
"""
from __future__ import annotations

import csv
import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .mobile_sequence_extractor import (
    SEQUENCE_FEATURE_COUNT,
    encode_two_hand_frame,
)

FRAME_FEATURE_COUNT = SEQUENCE_FEATURE_COUNT  # 151: two hand slots, presence, inter-hand wrist offset.
TEMPORAL_FEATURE_COUNT = FRAME_FEATURE_COUNT * 3  # geometry/position, first difference, second difference.


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def _canonical_path(path: str) -> str:
    return path.strip().replace("/", "\\").casefold()


def load_official_manifests(
    train_manifest: str | Path,
    test_manifest: str | Path,
    labels_csv: str | Path,
) -> tuple[list[dict], list[dict], dict]:
    """Load split records and resolve canonical labels by numeric folder ID.

    Working manifests preserve source membership. Where their text labels differ
    from labels.csv, the numeric ID is authoritative and the discrepancy is kept
    in ``label_mismatch`` for reporting.
    """
    label_rows = _read_csv(labels_csv)
    if not {"id", "label"}.issubset(label_rows[0] if label_rows else {}):
        raise ValueError("labels.csv must have id and label columns")
    labels_raw = {str(row["id"]).strip(): str(row["label"]) for row in label_rows}
    labels = {key: value.strip() for key, value in labels_raw.items()}
    if len(labels) != len(label_rows):
        raise ValueError("labels.csv contains duplicate class IDs")

    splits: dict[str, list[dict]] = {"train": [], "test": []}
    seen: dict[str, str] = {}
    source_mismatches = []
    labels_csv_mismatches = []
    for split, manifest_path in (("train", train_manifest), ("test", test_manifest)):
        rows = _read_csv(manifest_path)
        required = {"video_path", "class_id", "gesture_label"}
        if rows and not required.issubset(rows[0]):
            raise ValueError(f"{manifest_path} must include {sorted(required)}")
        for source in rows:
            path = str(source["video_path"]).strip()
            class_id = str(source["class_id"]).strip()
            if class_id not in labels:
                raise ValueError(f"Unknown class_id={class_id!r} in {manifest_path}")
            key = _canonical_path(path)
            if key in seen:
                raise ValueError(f"Video path appears in both/same split manifests: {path}")
            seen[key] = split
            # `gesture_label` in the audited working manifest is the upstream
            # dataset's ID-normalized label; keep it for the experiment. Separately
            # report disagreements with this workspace's older labels.csv.
            canonical_label = str(source["gesture_label"]).strip()
            source_label = str(source.get("source_manifest_label") or source["gesture_label"])
            labels_csv_label = labels[class_id]
            if source_label != str(source["gesture_label"]):
                source_mismatches.append({"video_path": path, "class_id": class_id,
                    "source_manifest_label": source_label,
                    "normalized_manifest_label": str(source["gesture_label"]),
                    "official_split": split})
            if canonical_label.casefold() != labels_csv_label.casefold():
                labels_csv_mismatches.append({"video_path": path, "class_id": class_id,
                    "normalized_manifest_label": canonical_label,
                    "workspace_labels_csv_label": labels_csv_label,
                    "official_split": split})
            record = dict(source)
            record.update({"class_id": class_id, "gesture_label": canonical_label,
                           "official_split": split,
                           "label_mismatch": canonical_label.casefold() != labels_csv_label.casefold(),
                           "source_manifest_label": source_label})
            splits[split].append(record)
    if not splits["train"] or not splits["test"]:
        raise ValueError("Both official train and test manifests must contain rows")
    return splits["train"], splits["test"], {
        "labels": labels,
        "source_manifest_text_mismatches": source_mismatches,
        "workspace_labels_csv_mismatches": labels_csv_mismatches,
    }


def resolve_video_path(dataset_root: str | Path, record: dict) -> Path:
    """Resolve manifest paths against either a `clips/` tree or numbered folders."""
    root = Path(dataset_root)
    parts = Path(str(record["video_path"]).replace("\\", "/")).parts
    candidates = [root.joinpath(*parts)]
    if parts and parts[0].casefold() == "clips":
        candidates.append(root.joinpath(*parts[1:]))
    candidates.append(root / str(record["class_id"]) / Path(parts[-1]).name)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Video for manifest row not found under {root}: {record['video_path']}")


def verify_split_files(
    train_rows: Sequence[dict], test_rows: Sequence[dict], dataset_root: str | Path,
    *, check_content_hashes: bool = False, progress_every: int = 250,
) -> dict:
    """Check path separation and file existence without opening test media by default.

    Keep ``check_content_hashes=False`` during model development. Hash-based
    cross-split duplicate verification should run only at the final test stage.
    """
    paths: dict[str, str] = {}
    resolved: list[tuple[dict, Path, str]] = []
    for split, rows in (("train", train_rows), ("test", test_rows)):
        for row in rows:
            path = resolve_video_path(dataset_root, row)
            key = str(path.resolve()).casefold()
            if key in paths:
                raise ValueError(f"Same resolved video is assigned to {paths[key]} and {split}: {path}")
            paths[key] = split
            resolved.append((row, path, split))

    duplicate_hashes = []
    if check_content_hashes:
        hashes: dict[str, tuple[str, str]] = {}
        for index, (row, path, split) in enumerate(resolved, 1):
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            value = digest.hexdigest()
            previous = hashes.get(value)
            if previous:
                duplicate_hashes.append({"sha256": value,
                                         "first_path": previous[0], "first_split": previous[1],
                                         "duplicate_path": str(path), "duplicate_split": split})
            else:
                hashes[value] = (str(path), split)
            if progress_every and index % progress_every == 0:
                print(f"Hashed {index}/{len(resolved)} manifest videos")
    cross_split_duplicates = [d for d in duplicate_hashes if d["first_split"] != d["duplicate_split"]]
    if cross_split_duplicates:
        raise ValueError(f"Exact duplicate video content crosses official splits: {cross_split_duplicates[:5]}")
    return {"train_count": len(train_rows), "test_count": len(test_rows),
            "resolved_count": len(resolved), "path_overlap": 0,
            "content_hash_check": check_content_hashes,
            "exact_duplicate_file_count": len(duplicate_hashes),
            "cross_split_duplicate_count": len(cross_split_duplicates),
            "duplicates": duplicate_hashes}


def hash_training_files(train_rows: Sequence[dict], dataset_root: str | Path,
                        progress_every: int = 250) -> dict[str, str]:
    """Hash official training files only, for a final-stage duplicate check."""
    hashes = {}
    for index, row in enumerate(train_rows, 1):
        path = resolve_video_path(dataset_root, row)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        hashes[digest.hexdigest()] = str(row["video_path"])
        if progress_every and index % progress_every == 0:
            print(f"Hashed {index}/{len(train_rows)} training videos")
    return hashes


def reject_cross_split_duplicate_test_files(
    training_hashes: dict[str, str], test_rows: Sequence[dict], dataset_root: str | Path,
    progress_every: int = 100,
) -> list[dict]:
    """At final evaluation time, reject exact test duplicates of training clips.

    This function reads test file bytes and therefore must not be called until
    model structure, scaling, prototypes, and rejection threshold are frozen.
    """
    duplicates = []
    for index, row in enumerate(test_rows, 1):
        path = resolve_video_path(dataset_root, row)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        train_path = training_hashes.get(digest.hexdigest())
        if train_path:
            duplicates.append({"training_video_path": train_path,
                               "test_video_path": str(row["video_path"]),
                               "sha256": digest.hexdigest()})
        if progress_every and index % progress_every == 0:
            print(f"Checked {index}/{len(test_rows)} test files for training duplicates")
    return duplicates


def split_training_rows(
    train_rows: Sequence[dict], validation_fraction: float = 0.20, seed: int = 42,
) -> tuple[list[dict], list[dict]]:
    """Create a deterministic, class-stratified video-level split from official train only."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    from sklearn.model_selection import train_test_split

    labels = np.asarray([str(row["gesture_label"]) for row in train_rows])
    counts = {label: int(np.sum(labels == label)) for label in set(labels)}
    if any(count < 2 for count in counts.values()):
        raise ValueError("Every class needs at least two official training videos for a stratified split")
    indices = np.arange(len(train_rows))
    fit_idx, val_idx = train_test_split(indices, test_size=validation_fraction,
                                        random_state=seed, stratify=labels)
    fit = [dict(train_rows[i], internal_partition="fit") for i in fit_idx]
    val = [dict(train_rows[i], internal_partition="validation") for i in val_idx]
    if {_canonical_path(r["video_path"]) for r in fit} & {_canonical_path(r["video_path"]) for r in val}:
        raise AssertionError("Internal fit/validation path overlap")
    return fit, val


@dataclass
class VideoSequence:
    video_path: str
    class_id: str
    gesture_label: str
    official_split: str
    features: np.ndarray
    sampled_frames: int
    hand_count_zero: int
    hand_count_one: int
    hand_count_two: int
    processing_seconds: float
    error: str = ""


class TwoHandVideoExtractor:
    """MediaPipe Tasks image-mode extractor that keeps zero/one/two-hand frames."""

    def __init__(self, model_path: str | Path, sample_every: int = 5,
                 min_detection_confidence: float = 0.5, max_frames: int | None = None):
        if sample_every < 1:
            raise ValueError("sample_every must be >= 1")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"MediaPipe Hand Landmarker model is missing: {self.model_path}")
        self.sample_every = sample_every
        self.max_frames = max_frames
        self.min_detection_confidence = min_detection_confidence
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("MediaPipe is required; install mediapipe in the notebook runtime") from exc
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=self.min_detection_confidence,
            min_hand_presence_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_detection_confidence)
        self.mp = mp
        self.detector = mp.tasks.vision.HandLandmarker.create_from_options(options)

    def close(self):
        if self.detector is not None:
            self.detector.close()
            self.detector = None

    def __enter__(self):
        if self.detector is None:
            raise RuntimeError("This extractor is already closed")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def extract(self, record: dict, video_path: str | Path) -> VideoSequence:
        start = time.perf_counter()
        frame_features: list[np.ndarray] = []
        zero = one = two = sampled = 0
        error = ""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            return VideoSequence(str(record["video_path"]), str(record["class_id"]),
                                 str(record["gesture_label"]), str(record["official_split"]),
                                 np.empty((0, FRAME_FEATURE_COUNT), np.float32), 0, 0, 0, 0,
                                 time.perf_counter() - start, "Could not open video")
        try:
            if self.detector is None:
                raise RuntimeError("Extractor is closed")
            frame_index = 0
            while self.max_frames is None or sampled < self.max_frames:
                ok, frame = capture.read()
                if not ok:
                    break
                current = frame_index
                frame_index += 1
                if current % self.sample_every:
                    continue
                sampled += 1
                image = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                                      data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = self.detector.detect(image)
                hands = result.hand_landmarks or []
                sides = result.handedness or []
                count = min(len(hands), 2)
                if count == 0:
                    zero += 1
                elif count == 1:
                    one += 1
                else:
                    two += 1
                features = encode_two_hand_frame(hands, sides, frame.shape[1], frame.shape[0])
                if features.shape != (FRAME_FEATURE_COUNT,) or not np.isfinite(features).all():
                    raise ValueError(f"Invalid two-hand frame feature at source frame {current}")
                frame_features.append(features.astype(np.float32, copy=False))
            if sampled == 0:
                error = "Video contains no readable frames"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            capture.release()
        matrix = (np.vstack(frame_features).astype(np.float32, copy=False) if frame_features
                  else np.empty((0, FRAME_FEATURE_COUNT), dtype=np.float32))
        return VideoSequence(str(record["video_path"]), str(record["class_id"]),
                             str(record["gesture_label"]), str(record["official_split"]),
                             matrix, sampled, zero, one, two,
                             time.perf_counter() - start, error)


def extract_sequences(
    rows: Sequence[dict], dataset_root: str | Path, extractor: TwoHandVideoExtractor,
    *, expected_split: str, progress_every: int = 50,
) -> list[VideoSequence]:
    """Extract only the explicitly selected split and retain every failure row."""
    if any(row.get("official_split") != expected_split for row in rows):
        raise ValueError(f"Refusing to extract rows outside requested split={expected_split!r}")
    sequences = []
    for i, row in enumerate(rows, 1):
        try:
            path = resolve_video_path(dataset_root, row)
            seq = extractor.extract(row, path)
        except Exception as exc:
            seq = VideoSequence(str(row["video_path"]), str(row["class_id"]),
                                 str(row["gesture_label"]), str(row["official_split"]),
                                 np.empty((0, FRAME_FEATURE_COUNT), np.float32), 0, 0, 0, 0, 0.0,
                                 f"{type(exc).__name__}: {exc}")
        sequences.append(seq)
        if progress_every and i % progress_every == 0:
            print(f"Processed {i}/{len(rows)} videos ({expected_split})", flush=True)
    return sequences


def write_sequence_archive(sequences: Sequence[VideoSequence], npz_path: str | Path,
                           report_csv: str | Path) -> None:
    """Write reusable concatenated arrays plus offsets and an all-video report."""
    target = Path(npz_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    offsets = [0]
    matrices = []
    for seq in sequences:
        matrix = np.asarray(seq.features, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != FRAME_FEATURE_COUNT:
            raise ValueError(f"Bad frame feature shape for {seq.video_path}: {matrix.shape}")
        matrices.append(matrix)
        offsets.append(offsets[-1] + len(matrix))
    all_features = np.vstack(matrices) if offsets[-1] else np.empty((0, FRAME_FEATURE_COUNT), np.float32)
    np.savez_compressed(target, frame_features=all_features,
                        offsets=np.asarray(offsets, dtype=np.int64),
                        video_path=np.asarray([s.video_path for s in sequences], dtype=str),
                        class_id=np.asarray([s.class_id for s in sequences], dtype=str),
                        gesture_label=np.asarray([s.gesture_label for s in sequences], dtype=str),
                        official_split=np.asarray([s.official_split for s in sequences], dtype=str))
    report = Path(report_csv)
    report.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video_path", "class_id", "gesture_label", "official_split", "sampled_frames",
              "zero_hand_frames", "one_hand_frames", "two_hand_frames", "valid_frame_features",
              "processing_seconds", "error"]
    with report.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for s in sequences:
            writer.writerow({"video_path": s.video_path, "class_id": s.class_id,
                             "gesture_label": s.gesture_label, "official_split": s.official_split,
                             "sampled_frames": s.sampled_frames, "zero_hand_frames": s.hand_count_zero,
                             "one_hand_frames": s.hand_count_one, "two_hand_frames": s.hand_count_two,
                             "valid_frame_features": len(s.features),
                             "processing_seconds": f"{s.processing_seconds:.6f}", "error": s.error})


def load_sequence_archive(npz_path: str | Path) -> list[dict]:
    """Read the sequence archive without pickle; an empty sequence is preserved."""
    with np.load(npz_path, allow_pickle=False) as data:
        x = data["frame_features"]
        offsets = data["offsets"]
        count = len(data["video_path"])
        if len(offsets) != count + 1 or offsets[0] != 0 or offsets[-1] != len(x):
            raise ValueError("Malformed sequence offsets")
        return [{"video_path": str(data["video_path"][i]),
                 "class_id": str(data["class_id"][i]),
                 "gesture_label": str(data["gesture_label"][i]),
                 "official_split": str(data["official_split"][i]),
                 "features": x[int(offsets[i]):int(offsets[i + 1])].copy()}
                for i in range(count)]


def temporal_features(frame_sequence: np.ndarray) -> np.ndarray:
    """Append first and second temporal differences to geometry/position frames."""
    frames = np.asarray(frame_sequence, dtype=np.float64)
    if frames.ndim != 2 or frames.shape[1] != FRAME_FEATURE_COUNT:
        raise ValueError(f"Expected (frames, {FRAME_FEATURE_COUNT}) array, got {frames.shape}")
    if not np.isfinite(frames).all():
        raise ValueError("Sequence contains non-finite features")
    if len(frames) == 0:
        return np.empty((0, TEMPORAL_FEATURE_COUNT), dtype=np.float64)
    velocity = np.vstack([np.zeros((1, FRAME_FEATURE_COUNT)), np.diff(frames, axis=0)])
    acceleration = np.vstack([np.zeros((1, FRAME_FEATURE_COUNT)), np.diff(velocity, axis=0)])
    return np.concatenate([frames, velocity, acceleration], axis=1)


def resample_sequence(sequence: np.ndarray, length: int = 24) -> np.ndarray:
    """Linearly resample a non-empty sequence to a fixed length for medoid selection."""
    x = np.asarray(sequence, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != TEMPORAL_FEATURE_COUNT:
        raise ValueError(f"Expected a temporal sequence of width {TEMPORAL_FEATURE_COUNT}")
    if len(x) == 0 or length < 1:
        raise ValueError("Cannot resample empty input or to a non-positive length")
    if len(x) == 1:
        return np.repeat(x, length, axis=0)
    positions = np.linspace(0, len(x) - 1, length)
    left = np.floor(positions).astype(int)
    right = np.minimum(left + 1, len(x) - 1)
    weight = (positions - left)[:, None]
    return x[left] * (1 - weight) + x[right] * weight


def dtw_distance(a: np.ndarray, b: np.ndarray, window_ratio: float = 0.20) -> float:
    """Return length-normalized DTW Euclidean cost; this is not a probability."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2 or x.shape[1] != y.shape[1]:
        raise ValueError("DTW inputs must be 2D sequences with equal feature width")
    if len(x) == 0 or len(y) == 0:
        return math.inf
    if not 0 <= window_ratio <= 1:
        raise ValueError("window_ratio must be between 0 and 1")
    # Euclidean pairwise frame costs via a matrix product, then banded dynamic programming.
    x2 = np.sum(x * x, axis=1)[:, None]
    y2 = np.sum(y * y, axis=1)[None, :]
    costs = np.sqrt(np.maximum(x2 + y2 - 2 * (x @ y.T), 0.0))
    n, m = costs.shape
    band = max(abs(n - m), int(math.ceil(window_ratio * max(n, m))))
    table = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((n + 1, m + 1), dtype=np.int32)
    table[0, 0] = 0.0
    for i in range(1, n + 1):
        start, end = max(1, i - band), min(m, i + band)
        for j in range(start, end + 1):
            candidates = (table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
            choice = int(np.argmin(candidates))
            if choice == 0:
                previous_steps = steps[i - 1, j]
            elif choice == 1:
                previous_steps = steps[i, j - 1]
            else:
                previous_steps = steps[i - 1, j - 1]
            table[i, j] = costs[i - 1, j - 1] + candidates[choice]
            steps[i, j] = previous_steps + 1
    count = int(steps[n, m])
    return float(table[n, m] / count) if count else math.inf


class TemporalDTWMatcher:
    """Robust-scaled frame-sequence medoids with nearest-prototype DTW matching."""

    def __init__(self, prototype_length: int = 24, window_ratio: float = 0.20):
        if prototype_length < 2:
            raise ValueError("prototype_length must be at least 2")
        self.prototype_length = prototype_length
        self.window_ratio = window_ratio
        self.threshold_: float | None = None
        self.prototypes_: dict[str, np.ndarray] = {}
        self.prototype_counts_: dict[str, int] = {}

    def fit(self, fit_sequences: Sequence[dict]) -> "TemporalDTWMatcher":
        valid = [r for r in fit_sequences if has_detected_hand(r["features"])]
        if not valid:
            raise ValueError("No valid fit sequences with detected frames")
        augmented = [temporal_features(r["features"]) for r in valid]
        all_frames = np.vstack(augmented)
        self.center_ = np.median(all_frames, axis=0)
        q25, q75 = np.quantile(all_frames, [0.25, 0.75], axis=0)
        self.scale_ = q75 - q25
        self.scale_[self.scale_ < 1e-9] = 1.0
        transformed = [(x - self.center_) / self.scale_ for x in augmented]
        labels = sorted({str(r["gesture_label"]) for r in valid})
        self.prototypes_, self.prototype_counts_ = {}, {}
        for label in labels:
            indices = [i for i, row in enumerate(valid) if str(row["gesture_label"]) == label]
            if not indices:
                continue
            compressed = np.stack([resample_sequence(transformed[i], self.prototype_length).ravel()
                                   for i in indices])
            norms = np.sum(compressed * compressed, axis=1)
            pairwise_sq = np.maximum(norms[:, None] + norms[None, :] - 2 * compressed @ compressed.T, 0)
            medoid_index = int(np.argmin(np.mean(np.sqrt(pairwise_sq), axis=1)))
            self.prototypes_[label] = resample_sequence(transformed[indices[medoid_index]],
                                                        self.prototype_length)
            self.prototype_counts_[label] = len(indices)
        self.class_labels_ = tuple(self.prototypes_)
        if not self.class_labels_:
            raise ValueError("No class prototypes were constructed")
        return self

    def _transform(self, sequence: np.ndarray) -> np.ndarray:
        return (temporal_features(sequence) - self.center_) / self.scale_

    def predict_one(self, sequence: np.ndarray) -> dict:
        if not self.prototypes_:
            raise RuntimeError("Call fit() before prediction")
        if not has_detected_hand(sequence):
            return {"predicted_label": "UNKNOWN", "nearest_label": "UNKNOWN",
                    "nearest_distance": math.inf, "distances": {}}
        transformed = resample_sequence(self._transform(sequence), self.prototype_length)
        distances = {label: dtw_distance(transformed, prototype, self.window_ratio)
                     for label, prototype in self.prototypes_.items()}
        label = min(distances, key=distances.get)
        distance = float(distances[label])
        result = "UNKNOWN" if self.threshold_ is not None and distance > self.threshold_ else label
        return {"predicted_label": result, "nearest_label": label,
                "nearest_distance": distance, "distances": distances}

    def calibrate_rejection(self, validation_sequences: Sequence[dict], target_correct_coverage: float = 0.95) -> float:
        """Calibrate on correct known validation clips only; no probability is estimated."""
        if not 0 < target_correct_coverage <= 1:
            raise ValueError("target_correct_coverage must be in (0, 1]")
        correct_distances = []
        for row in validation_sequences:
            result = self.predict_one(row["features"])
            if result["nearest_label"] == str(row["gesture_label"]):
                correct_distances.append(float(result["nearest_distance"]))
        if not correct_distances:
            raise ValueError("No correctly predicted validation sequences; cannot calibrate threshold")
        ordered = np.sort(correct_distances)
        index = max(0, math.ceil(target_correct_coverage * len(ordered)) - 1)
        self.threshold_ = float(ordered[min(index, len(ordered) - 1)])
        return self.threshold_


def write_manual_review_template(rows: Sequence[dict], hand_summary: dict[str, dict], output_csv: str | Path) -> None:
    """Create a human annotation template; labels are intentionally left blank."""
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["video_path", "class_id", "gesture_label", "official_split", "temporal_type",
              "hand_requirement", "reviewer", "notes"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({"video_path": row["video_path"], "class_id": row["class_id"],
                             "gesture_label": row["gesture_label"],
                             "official_split": row["official_split"],
                             "temporal_type": "", "hand_requirement": "",
                             "reviewer": "", "notes": ""})


def has_detected_hand(sequence: np.ndarray) -> bool:
    """Presence bits follow the two 73-value hand slots at indices 146 and 147."""
    x = np.asarray(sequence)
    return bool(x.ndim == 2 and x.shape[1] == FRAME_FEATURE_COUNT and len(x)
                and np.any(x[:, 2 * 73:2 * 73 + 2] > 0.5))


def evaluate_sequence_predictions(rows: Sequence[dict], matcher: TemporalDTWMatcher) -> dict:
    """Calculate known-class metrics with UNKNOWN rejects counted as incorrect."""
    from sklearn.metrics import (accuracy_score, confusion_matrix,
                                 precision_recall_fscore_support)

    labels = list(matcher.class_labels_)
    truth, predicted, nearest, distances, latency_ms = [], [], [], [], []
    for row in rows:
        start = time.perf_counter()
        result = matcher.predict_one(row["features"])
        latency_ms.append((time.perf_counter() - start) * 1000.0)
        truth.append(str(row["gesture_label"]))
        predicted.append(result["predicted_label"])
        nearest.append(result["nearest_label"])
        distances.append(result["nearest_distance"])
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predicted, labels=labels, average=None, zero_division=0)
    accepted = np.asarray([p != "UNKNOWN" for p in predicted], dtype=bool)
    cm = confusion_matrix(truth, predicted, labels=labels + ["UNKNOWN"])
    rows_out = [{"gesture_label": label, "precision": float(precision[i]),
                 "recall": float(recall[i]), "f1_score": float(f1[i]),
                 "support": int(support[i])} for i, label in enumerate(labels)]
    return {"sample_count": len(rows), "accepted_count": int(accepted.sum()),
            "coverage": float(accepted.mean()) if len(accepted) else 0.0,
            "rejection_rate": float((~accepted).mean()) if len(accepted) else 0.0,
            "accuracy": float(accuracy_score(truth, predicted)) if truth else 0.0,
            "precision_macro": float(np.mean(precision)) if len(precision) else 0.0,
            "recall_macro": float(np.mean(recall)) if len(recall) else 0.0,
            "f1_macro": float(np.mean(f1)) if len(f1) else 0.0,
            "accuracy_accepted": float(accuracy_score(
                [y for y, ok in zip(truth, accepted) if ok],
                [y for y, ok in zip(predicted, accepted) if ok])) if accepted.any() else 0.0,
            "confusion_labels": labels + ["UNKNOWN"], "confusion_matrix": cm.tolist(),
            "per_class": rows_out, "truth": truth, "predicted": predicted,
            "nearest_label": nearest, "nearest_distance": distances,
            "matching_latency_ms": latency_ms,
            "matching_latency_mean_ms": float(np.mean(latency_ms)) if latency_ms else 0.0,
            "matching_latency_p95_ms": float(np.percentile(latency_ms, 95)) if latency_ms else 0.0}
