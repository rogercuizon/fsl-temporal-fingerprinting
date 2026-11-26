"""Dataset scanning, video-level extraction, reports, and CSV writing."""
from __future__ import annotations
import csv
from pathlib import Path

import numpy as np

from .fingerprint import FEATURE_COUNT, FEATURE_NAMES, aggregate_fingerprints, make_fingerprint
from .landmark_extractor import LandmarkExtractor

VIDEO_EXTENSIONS = {".mov", ".mp4", ".avi", ".m4v", ".mkv"}


def load_label_mapping(root: str | Path) -> dict[str, str]:
    """Read dataset ID-to-English-label mapping from labels.csv when present."""
    path = Path(root) / "labels.csv"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return {row["id"].strip(): row["label"].strip() for row in csv.DictReader(stream)
                if row.get("id") and row.get("label")}


def scan_videos(root: str | Path) -> list[tuple[Path, str]]:
    root = Path(root)
    labels = load_label_mapping(root)
    return sorted(((p, labels.get(p.parent.name, p.parent.name)) for p in root.rglob("*") if p.is_file() and p.suffix.lower() in VIDEO_EXTENSIONS),
                  key=lambda item: str(item[0]).casefold())


def process_dataset(root: str | Path, output: str | Path, sample_every: int = 5, max_hands: int = 1,
                    min_valid_frames: int = 1, visualize: bool = False) -> None:
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if visualize:
        (output / "visualizations").mkdir(exist_ok=True)
    extractor = LandmarkExtractor(max_hands=max_hands, sample_every=sample_every, visualize=visualize)
    rows, reports, failures = [], [], []
    videos = scan_videos(root)
    for path, label in videos:
        result = extractor.extract_video(path, output / "visualizations" if visualize else None)
        fingerprints, invalid = [], 0
        for points in result.landmarks:
            try:
                fingerprints.append(make_fingerprint(points))
            except ValueError:
                invalid += 1
        valid = len(fingerprints)
        rate = result.processed_frames / result.sampled_frames if result.sampled_frames else 0.0
        status = "ok" if valid >= min_valid_frames else "insufficient_valid_frames"
        reports.append({"video_path": str(path), "gesture_label": label, "sampled_frames": result.sampled_frames,
                        "processed_frames": result.processed_frames, "no_hand_frames": result.no_hand_frames,
                        "valid_frames": valid, "invalid_landmark_frames": invalid, "detection_rate": rate,
                        "status": status, "error": result.error})
        if status == "ok":
            row = {"video_path": str(path), "gesture_label": label, "detected_frames": valid, "detection_rate": rate}
            row.update({f"feature_{i+1}": float(x) for i, x in enumerate(aggregate_fingerprints(fingerprints))})
            rows.append(row)
        else:
            failures.append({"video_path": str(path), "gesture_label": label, "reason": result.error or status,
                             "sampled_frames": result.sampled_frames, "detected_frames": valid})
    feature_rows = [{"index": i+1, "name": name,
                     "description": ("Wrist-relative normalized x/y/z coordinate" if i < 60 else
                                     "Joint interior angle divided by pi" if i < 65 else
                                     "Normalized wrist-to-fingertip Euclidean distance")}
                    for i, name in enumerate(FEATURE_NAMES)]
    _write(output / "fingerprints.csv", rows, ["video_path", "gesture_label", "detected_frames", "detection_rate"] + [f"feature_{i}" for i in range(1, FEATURE_COUNT+1)])
    _write(output / "extraction_report.csv", reports, ["video_path", "gesture_label", "sampled_frames", "processed_frames", "no_hand_frames", "valid_frames", "invalid_landmark_frames", "detection_rate", "status", "error"])
    _write(output / "failed_videos.csv", failures, ["video_path", "gesture_label", "reason", "sampled_frames", "detected_frames"])
    _write(output / "feature_description.csv", feature_rows, ["index", "name", "description"])


def _write(path: Path, rows: list[dict], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
