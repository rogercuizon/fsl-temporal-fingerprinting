"""Reference-only feature scaling and nearest-class-prototype matching."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class Match:
    label: str
    distance: float
    accepted: bool


class FingerprintMatcher:
    def __init__(self, metric: str = "standardized", rejection_threshold: float = float("inf")):
        if metric not in {"euclidean", "standardized"}:
            raise ValueError("metric must be 'euclidean' or 'standardized'")
        self.metric, self.rejection_threshold = metric, float(rejection_threshold)
        self.labels = None

    def fit(self, features, labels):
        x = np.asarray(features, dtype=float)
        y = np.asarray(labels, dtype=str)
        if x.ndim != 2 or len(x) != len(y) or not len(y) or not np.isfinite(x).all():
            raise ValueError("Expected finite 2D features and one label per row")
        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-12] = 1.0
        self.prototypes_ = {label: x[y == label].mean(axis=0) for label in sorted(set(y))}
        self.labels = tuple(self.prototypes_)
        return self

    def _distance(self, a, b):
        delta = np.asarray(a) - np.asarray(b)
        if self.metric == "standardized":
            delta = delta / self.scale_
        return float(np.linalg.norm(delta))

    def predict_one(self, features) -> Match:
        if self.labels is None:
            raise RuntimeError("Call fit with reference/training data first")
        x = np.asarray(features, dtype=float)
        if x.shape != (next(iter(self.prototypes_.values())).shape[0],) or not np.isfinite(x).all():
            raise ValueError("Input must be a finite fingerprint matching the training feature count")
        label, distance = min(((k, self._distance(x, v)) for k, v in self.prototypes_.items()), key=lambda pair: pair[1])
        accepted = distance <= self.rejection_threshold
        return Match(label if accepted else "UNKNOWN", distance, accepted)

    def calibrate_threshold(self, validation_features, validation_labels, target_coverage: float = 0.95):
        """Set threshold from correctly classified validation distances; never use final test data."""
        if not 0 < target_coverage <= 1:
            raise ValueError("target_coverage must be in (0, 1]")
        matches = [self.predict_one(x) for x in validation_features]
        accepted_correct = [m.distance for m, truth in zip(matches, validation_labels)
                            if m.label == str(truth) and m.accepted]
        if not accepted_correct:
            raise ValueError("No correctly classified validation rows to calibrate threshold")
        self.rejection_threshold = float(np.quantile(accepted_correct, target_coverage))
        return self.rejection_threshold
