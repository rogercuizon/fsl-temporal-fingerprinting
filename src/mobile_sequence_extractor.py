"""Training-split-only two-hand temporal feature extraction for the Android model."""
from __future__ import annotations

import csv
import json
import tempfile
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np

from .fingerprint import FEATURE_COUNT, make_fingerprint

HAND_SLOT_FEATURES = FEATURE_COUNT + 3  # 70 shape features + normalized wrist position
SEQUENCE_FEATURE_COUNT = 2 * HAND_SLOT_FEATURES + 2 + 3  # slots, presence bits, relative wrist vector


def encode_two_hand_frame(hand_landmarks, handedness, image_width: int, image_height: int) -> np.ndarray:
    """Encode left/right hand shape and position without changing the saved 70-feature function."""
    del image_width, image_height  # MediaPipe coordinates are already image-normalized.
    slots = [None, None]
    wrists = [None, None]
    for index, hand in enumerate(hand_landmarks[:2]):
        category = handedness[index][0] if index < len(handedness) and handedness[index] else None
        side = getattr(category, "category_name", "")
        slot = 0 if side == "Left" else 1 if side == "Right" else None
        if slot is None or slots[slot] is not None:
            slot = 0 if slots[0] is None else 1 if slots[1] is None else None
        if slot is None:
            continue
        values = [(point.x, point.y, point.z) for point in hand]
        try:
            shape = make_fingerprint(values)
        except ValueError:
            continue
        slots[slot] = shape
        wrists[slot] = np.asarray(values[0], dtype=np.float32)

    output = np.zeros(SEQUENCE_FEATURE_COUNT, dtype=np.float32)
    for slot in range(2):
        if slots[slot] is None:
            continue
        start = slot * HAND_SLOT_FEATURES
        output[start:start + FEATURE_COUNT] = slots[slot]
        output[start + FEATURE_COUNT:start + HAND_SLOT_FEATURES] = wrists[slot]
        output[2 * HAND_SLOT_FEATURES + slot] = 1.0
    if wrists[0] is not None and wrists[1] is not None:
        output[2 * HAND_SLOT_FEATURES + 2:2 * HAND_SLOT_FEATURES + 5] = wrists[1] - wrists[0]
    return output


def extract_official_training_sequences(source_dir: str | Path, manifest_csv: str | Path,
                                        output_dir: str | Path, sample_every: int = 5,
                                        progress_every: int = 100,
                                        max_videos: int | None = None) -> dict:
    """Extract frame sequences only from rows marked official_split=train.

    Test paths are neither read from the video archive nor included in output.
    Every training row is retained, including rows with no detected hands.
    """
    if sample_every < 1:
        raise ValueError("sample_every must be positive")
    source = Path(source_dir).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with Path(manifest_csv).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["official_split"] == "train"]
    if max_videos is not None:
        if max_videos < 1:
            raise ValueError("max_videos must be positive when set")
        rows = rows[:max_videos]
    if not rows:
        raise ValueError("No official training rows found")

    model_path = source / "models" / "hand_landmarker.task"
    if not model_path.is_file():
        model_path = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"
    if not model_path.is_file():
        raise FileNotFoundError(f"MediaPipe hand model missing: {model_path}")

    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("MediaPipe is required for sequence extraction") from exc

    all_frames: list[np.ndarray] = []
    offsets = [0]
    video_paths, class_ids, labels, categories = [], [], [], []
    sampled_counts, detected_counts, valid_counts, error_messages = [], [], [], []
    zip_path = source / "clips.zip"
    started = time.perf_counter()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.IMAGE,
        num_hands=2,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    with zipfile.ZipFile(zip_path) as archive, mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        member_by_key = {name.replace("\\", "/").casefold(): name
                         for name in archive.namelist() if not name.endswith("/")}
        for video_index, row in enumerate(rows, 1):
            relative = row["video_path"].replace("\\", "/")
            member = member_by_key.get(relative.casefold())
            if member is None:
                raise FileNotFoundError(f"Training video missing from archive: {relative}")
            frame_rows = []
            sampled = detected = valid = 0
            errors = []
            with tempfile.TemporaryDirectory(prefix="fsl105_clip_") as temp_dir:
                clip_path = Path(temp_dir) / Path(relative).name
                clip_path.write_bytes(archive.read(member))
                capture = cv2.VideoCapture(str(clip_path))
                if not capture.isOpened():
                    capture.release()
                    errors.append("Could not open video")
                else:
                    frame_index = 0
                    try:
                        while True:
                            okay, frame = capture.read()
                            if not okay:
                                break
                            current = frame_index
                            frame_index += 1
                            if current % sample_every:
                                continue
                            sampled += 1
                            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                            result = detector.detect(image)
                            hand_list = result.hand_landmarks or []
                            if hand_list:
                                detected += 1
                            encoded = encode_two_hand_frame(hand_list, result.handedness or [], frame.shape[1], frame.shape[0])
                            if encoded[2 * HAND_SLOT_FEATURES] or encoded[2 * HAND_SLOT_FEATURES + 1]:
                                valid += 1
                            frame_rows.append(encoded)
                    except Exception as exc:
                        errors.append(f"Frame extraction error: {exc}")
                    finally:
                        capture.release()

            if not frame_rows:
                frame_rows = [np.zeros(SEQUENCE_FEATURE_COUNT, dtype=np.float32)]
            all_frames.extend(frame_rows)
            offsets.append(len(all_frames))
            video_paths.append(row["video_path"])
            class_ids.append(row["class_id"])
            labels.append(row["gesture_label"])
            categories.append(row["category"])
            sampled_counts.append(sampled)
            detected_counts.append(detected)
            valid_counts.append(valid)
            error_messages.append(" | ".join(errors))
            if progress_every and video_index % progress_every == 0:
                elapsed = time.perf_counter() - started
                print(f"Processed {video_index}/{len(rows)} official training videos ({elapsed:.1f}s)", flush=True)

    frame_matrix = np.vstack(all_frames).astype(np.float32)
    np.savez_compressed(output / "training_sequences.npz",
                        frame_features=frame_matrix,
                        offsets=np.asarray(offsets, dtype=np.int64),
                        video_paths=np.asarray(video_paths, dtype=str),
                        class_ids=np.asarray(class_ids, dtype=str),
                        labels=np.asarray(labels, dtype=str),
                        categories=np.asarray(categories, dtype=str))
    report_fields = ["video_path", "class_id", "gesture_label", "sampled_frames", "detected_frames",
                     "valid_feature_frames", "detection_rate", "error"]
    with (output / "training_extraction_report.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=report_fields)
        writer.writeheader()
        for path, class_id, label, sampled, detected, valid, error in zip(
                video_paths, class_ids, labels, sampled_counts, detected_counts, valid_counts, error_messages):
            writer.writerow({"video_path": path, "class_id": class_id, "gesture_label": label,
                             "sampled_frames": sampled, "detected_frames": detected,
                             "valid_feature_frames": valid,
                             "detection_rate": detected / sampled if sampled else 0.0, "error": error})
    metadata = {"video_count": len(video_paths), "frame_count": len(all_frames),
                "feature_count_per_frame": SEQUENCE_FEATURE_COUNT, "sample_every": sample_every,
                "official_split_used": "train only", "test_media_read": False,
                "test_features_extracted": False, "low_detection_rows_excluded": 0,
                "original_zip_modified": False, "elapsed_seconds": time.perf_counter() - started}
    (output / "training_extraction_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata
