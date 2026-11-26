import numpy as np
import pytest
from src.fingerprint import (FEATURE_COUNT, aggregate_fingerprints, euclidean_distance,
                             joint_angle, make_fingerprint, normalize_landmarks)


def sample_hand():
    return np.array([(i * 0.01, i * 0.02, i * 0.005) for i in range(21)], dtype=float) + [0, 0.1, 0]


def test_distance():
    assert euclidean_distance([0, 0], [3, 4]) == 5


def test_angle_is_normalized():
    assert joint_angle([1, 0, 0], [0, 0, 0], [0, 1, 0]) == pytest.approx(0.5)


def test_normalization_places_wrist_at_origin_and_scale_one():
    points = normalize_landmarks(sample_hand())
    assert np.allclose(points[0], 0)
    assert np.linalg.norm(points[9]) == pytest.approx(1)


def test_fingerprint_length_and_median():
    fingerprint = make_fingerprint(sample_hand())
    assert fingerprint.shape == (FEATURE_COUNT,)
    assert aggregate_fingerprints([fingerprint, fingerprint]).shape == (FEATURE_COUNT,)


def test_missing_or_invalid_landmarks_rejected():
    with pytest.raises(ValueError): make_fingerprint([[0, 0, 0]] * 20)
    points = sample_hand(); points[1, 0] = np.nan
    with pytest.raises(ValueError): make_fingerprint(points)


def test_degenerate_scale_rejected():
    with pytest.raises(ValueError): normalize_landmarks(np.zeros((21, 3)))
