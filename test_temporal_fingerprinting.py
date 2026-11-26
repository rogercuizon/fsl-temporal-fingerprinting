import csv
from pathlib import Path

import numpy as np
import pytest

from src.temporal_fingerprinting import (
    FRAME_FEATURE_COUNT,
    TEMPORAL_FEATURE_COUNT,
    TemporalDTWMatcher,
    VideoSequence,
    dtw_distance,
    evaluate_sequence_predictions,
    has_detected_hand,
    load_official_manifests,
    load_sequence_archive,
    reject_cross_split_duplicate_test_files,
    split_training_rows,
    temporal_features,
    verify_split_files,
    write_sequence_archive,
)


def _row(path, class_id="0", label="A", split="train"):
    return {"video_path": path, "class_id": class_id, "gesture_label": label,
            "official_split": split}


def _sequence(label, center, split="train", offset=0.0):
    x = np.zeros((8, FRAME_FEATURE_COUNT), dtype=np.float32)
    x[:, 0] = np.linspace(center, center + 0.2, len(x)) + offset
    x[:, 146] = 1.0
    return {"video_path": f"{label}_{offset}.MOV", "class_id": "0",
            "gesture_label": label, "official_split": split, "features": x}


def test_temporal_features_include_geometry_velocity_and_acceleration():
    x = np.zeros((3, FRAME_FEATURE_COUNT), dtype=float)
    x[:, 0] = [0, 1, 3]
    result = temporal_features(x)
    assert result.shape == (3, TEMPORAL_FEATURE_COUNT)
    np.testing.assert_allclose(result[:, FRAME_FEATURE_COUNT], [0, 1, 2])
    np.testing.assert_allclose(result[:, 2 * FRAME_FEATURE_COUNT], [0, 1, 1])


def test_dtw_identical_and_symmetric_sequences():
    a = np.arange(24, dtype=float).reshape(6, 4)
    b = np.vstack([a[0], a[0], a[1:]])
    assert dtw_distance(a, a) == pytest.approx(0.0)
    assert dtw_distance(a, b) == pytest.approx(dtw_distance(b, a))


def test_two_hand_temporal_matcher_and_unknown_rejection():
    fit = [_sequence("A", 0.1, offset=o) for o in (0, .01, -.01)]
    fit += [_sequence("B", -0.5, offset=o) for o in (0, .01, -.01)]
    matcher = TemporalDTWMatcher(prototype_length=8).fit(fit)
    validation = [_sequence("A", 0.1, "validation"), _sequence("B", -0.5, "validation")]
    threshold = matcher.calibrate_rejection(validation, target_correct_coverage=1.0)
    assert np.isfinite(threshold)
    assert matcher.predict_one(validation[0]["features"])["predicted_label"] == "A"
    no_hands = np.zeros((8, FRAME_FEATURE_COUNT), dtype=np.float32)
    assert matcher.predict_one(no_hands)["predicted_label"] == "UNKNOWN"
    metrics = evaluate_sequence_predictions(validation, matcher)
    assert metrics["sample_count"] == 2
    assert metrics["accepted_count"] == 2
    assert metrics["accuracy"] == pytest.approx(1.0)


def test_has_detected_hand_uses_presence_bits():
    assert not has_detected_hand(np.zeros((2, FRAME_FEATURE_COUNT)))
    x = np.zeros((2, FRAME_FEATURE_COUNT)); x[1, 147] = 1
    assert has_detected_hand(x)
    assert not has_detected_hand(np.empty((0, FRAME_FEATURE_COUNT)))


def test_load_manifests_uses_id_label_and_reports_text_mismatch(tmp_path):
    labels = tmp_path / "labels.csv"
    train = tmp_path / "train.csv"
    test = tmp_path / "test.csv"
    labels.write_text("id,label,category\n0,Canonical,GREETING\n", encoding="utf-8")
    train.write_text("video_path,class_id,gesture_label,source_manifest_label\n"
                     "clips/0/0.MOV,0,Old Name,Old Name\n", encoding="utf-8")
    test.write_text("video_path,class_id,gesture_label,source_manifest_label\n"
                    "clips/0/1.MOV,0,Canonical,Canonical\n", encoding="utf-8")
    tr, te, audit = load_official_manifests(train, test, labels)
    assert tr[0]["gesture_label"] == "Old Name"
    assert tr[0]["official_split"] == "train"
    assert te[0]["official_split"] == "test"
    assert len(audit["workspace_labels_csv_mismatches"]) == 1
    assert len(audit["source_manifest_text_mismatches"]) == 0


def test_split_validation_and_path_integrity_do_not_cross_split(tmp_path):
    train_rows = [_row(f"clips/0/{i}.MOV", label="A") for i in range(4)]
    fit, val = split_training_rows(train_rows, validation_fraction=.25, seed=9)
    assert len(fit) == 3 and len(val) == 1
    assert not {r["video_path"] for r in fit} & {r["video_path"] for r in val}
    for name in ("0.MOV", "1.MOV"):
        path = tmp_path / "0" / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
    result = verify_split_files([_row("clips/0/0.MOV")],
                                [_row("clips/0/1.MOV", split="test")], tmp_path)
    assert result["path_overlap"] == 0


def test_duplicate_content_detected_at_final_test_stage(tmp_path):
    (tmp_path / "0").mkdir()
    (tmp_path / "0" / "a.MOV").write_bytes(b"same clip")
    (tmp_path / "0" / "b.MOV").write_bytes(b"same clip")
    train = [_row("clips/0/a.MOV")]
    test = [_row("clips/0/b.MOV", split="test")]
    # Only training content is hashed before model selection.
    hashes = verify_split_files(train, test, tmp_path, check_content_hashes=False)
    assert hashes["content_hash_check"] is False
    from src.temporal_fingerprinting import hash_training_files
    train_hashes = hash_training_files(train, tmp_path, progress_every=0)
    duplicates = reject_cross_split_duplicate_test_files(train_hashes, test, tmp_path,
                                                          progress_every=0)
    assert len(duplicates) == 1


def test_sequence_archive_preserves_failed_and_empty_rows(tmp_path):
    seq = VideoSequence("0/0.MOV", "0", "A", "train",
                        np.zeros((2, FRAME_FEATURE_COUNT), dtype=np.float32),
                        2, 2, 0, 0, .1, "")
    failed = VideoSequence("0/1.MOV", "0", "A", "train",
                           np.empty((0, FRAME_FEATURE_COUNT), dtype=np.float32),
                           0, 0, 0, 0, .2, "Could not open video")
    archive, report = tmp_path / "seq.npz", tmp_path / "report.csv"
    write_sequence_archive([seq, failed], archive, report)
    rows = load_sequence_archive(archive)
    assert len(rows) == 2
    assert rows[0]["features"].shape == (2, FRAME_FEATURE_COUNT)
    assert rows[1]["features"].shape == (0, FRAME_FEATURE_COUNT)
    with report.open(encoding="utf-8-sig", newline="") as f:
        report_rows = list(csv.DictReader(f))
    assert report_rows[1]["error"] == "Could not open video"
