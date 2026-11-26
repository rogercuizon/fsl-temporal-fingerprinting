import numpy as np
from src.fingerprint_matcher import FingerprintMatcher


def test_matching_and_unknown_rejection():
    x = np.array([[0.0, 0], [0.1, 0], [4, 4], [4.1, 4]])
    matcher = FingerprintMatcher(metric="euclidean", rejection_threshold=0.5).fit(x, ["a", "a", "b", "b"])
    assert matcher.predict_one([0.05, 0]).label == "a"
    assert matcher.predict_one([50, 50]).label == "UNKNOWN"


def test_threshold_calibrated_from_validation():
    x = np.array([[0.0, 0], [0.1, 0], [4, 4], [4.1, 4]])
    matcher = FingerprintMatcher(metric="euclidean").fit(x, ["a", "a", "b", "b"])
    threshold = matcher.calibrate_threshold([[0.1, 0], [4.1, 4]], ["a", "b"])
    assert threshold >= 0
