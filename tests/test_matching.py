import csv
import numpy as np
import pytest

from src.fingerprint import FEATURE_COUNT
from src.matching import (FeatureScaler, apply_rejection, assert_no_official_test_leakage,
                          fingerprint_distance, load_official_training_data, make_prototype,
                          split_official_training)


def test_distance_formulas():
    a = np.ones(FEATURE_COUNT)
    b = np.zeros(FEATURE_COUNT)
    ones = np.ones(FEATURE_COUNT)
    assert fingerprint_distance(a, b, "euclidean") == pytest.approx(np.sqrt(70))
    assert fingerprint_distance(a, b, "training_scaled", ones) == pytest.approx(np.sqrt(70))
    assert fingerprint_distance(a, b, "robust_scaled", ones) == pytest.approx(np.sqrt(70))
    assert fingerprint_distance(a, b, "group_balanced", ones) == pytest.approx(1.0)


def test_feature_scaling_is_fitted_from_training_only():
    train = np.zeros((2, FEATURE_COUNT))
    train[1, 0] = 2.0
    standard = FeatureScaler("standard").fit(train)
    robust = FeatureScaler("robust").fit(train)
    assert standard.scale_[0] == pytest.approx(1.0)
    assert robust.scale_[0] == pytest.approx(1.0)
    assert np.all(standard.scale_[1:] == 1.0)  # safe zero-variance handling


def test_mean_and_median_prototypes():
    rows = np.zeros((3, FEATURE_COUNT))
    rows[:, 0] = [0.0, 2.0, 100.0]
    assert make_prototype(rows, "mean")[0] == pytest.approx(34.0)
    assert make_prototype(rows, "median")[0] == pytest.approx(2.0)


def test_unknown_rejection():
    assert apply_rejection("HELLO", 0.5, 1.0) == ("HELLO", True)
    assert apply_rejection("HELLO", 1.01, 1.0) == ("UNKNOWN", False)


def _synthetic_train_rows():
    rows = []
    for label, offset in (("A", 0.0), ("B", 5.0), ("C", 10.0)):
        for i in range(5):
            x = np.zeros(FEATURE_COUNT); x[0] = offset + i * 0.01
            rows.append({"video_path":f"{label}/{i}.MOV", "gesture_label":label, "features":x})
    return rows


def test_official_test_paths_never_enter_fit_or_validation():
    train_rows = _synthetic_train_rows()
    test_paths = {r"a/test.MOV", r"b/test.MOV"}
    fit_rows, validation_rows = split_official_training(train_rows, test_paths, seed=42)
    assert_no_official_test_leakage(fit_rows, validation_rows, test_paths)
    assert not ({r["video_path"].lower() for r in fit_rows+validation_rows} & {p.lower() for p in test_paths})
    with pytest.raises(ValueError, match="Official test"):
        split_official_training(train_rows + [{"video_path":"A/test.MOV","gesture_label":"A","features":np.zeros(FEATURE_COUNT)}],
                                test_paths, seed=42)


def test_test_fingerprint_features_are_not_parsed_during_preparation(tmp_path):
    fingerprints = tmp_path / "fingerprints.csv"
    mapping = tmp_path / "split.csv"
    fields = ["video_path", "gesture_label"] + [f"feature_{i}" for i in range(1, FEATURE_COUNT+1)]
    with fingerprints.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        train = {"video_path":"0/train.MOV", "gesture_label":"A"}
        test = {"video_path":"0/test.MOV", "gesture_label":"A"}
        train.update({f"feature_{i}":0 for i in range(1, FEATURE_COUNT+1)})
        test.update({f"feature_{i}":"must-not-parse" for i in range(1, FEATURE_COUNT+1)})
        writer.writerow(train); writer.writerow(test)
    with mapping.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_path","gesture_label","official_split"]);writer.writeheader()
        writer.writerow({"video_path":"0/train.MOV","gesture_label":"A","official_split":"train"})
        writer.writerow({"video_path":"0/test.MOV","gesture_label":"A","official_split":"test"})
    training, test_paths, assignments = load_official_training_data(fingerprints, mapping)
    assert len(training) == 1
    assert test_paths == {r"0\test.mov"}
    assert sum(a["official_split"] == "test" for a in assignments) == 1
