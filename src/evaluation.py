"""Video-level train/validation/test splits and metrics."""
from __future__ import annotations
import numpy as np
from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, confusion_matrix,
                             classification_report)
from .fingerprint_matcher import FingerprintMatcher


def split_video_rows(rows, seed=42, validation_fraction=0.2, test_fraction=0.2, signer_column=None):
    """Rows are video-level records. Optional signer column groups all a signer's videos."""
    if signer_column and all(signer_column in row for row in rows):
        # Group-level split keeps signer identities disjoint.
        import pandas as pd
        from sklearn.model_selection import GroupShuffleSplit
        frame = pd.DataFrame(rows)
        outer = GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed)
        trainval_i, test_i = next(outer.split(frame, groups=frame[signer_column]))
        trainval = frame.iloc[trainval_i]
        inner = GroupShuffleSplit(n_splits=1, test_size=validation_fraction/(1-test_fraction), random_state=seed)
        train_i, val_i = next(inner.split(trainval, groups=trainval[signer_column]))
        return trainval.iloc[train_i].to_dict("records"), trainval.iloc[val_i].to_dict("records"), frame.iloc[test_i].to_dict("records"), "signer-independent"
    trainval, test = train_test_split(rows, test_size=test_fraction, random_state=seed,
                                      stratify=[row["gesture_label"] for row in rows])
    train, validation = train_test_split(trainval, test_size=validation_fraction/(1-test_fraction), random_state=seed,
                                         stratify=[row["gesture_label"] for row in trainval])
    return train, validation, test, "video-independent; signer metadata unavailable"


def evaluate_rows(train, validation, test, metric="standardized", target_coverage=0.95, similar_pairs=None):
    def xy(rows):
        return (np.array([[float(r[f"feature_{i}"]) for i in range(1, 71)] for r in rows]),
                np.array([r["gesture_label"] for r in rows], dtype=str))
    x_train, y_train = xy(train); x_val, y_val = xy(validation); x_test, y_test = xy(test)
    matcher = FingerprintMatcher(metric=metric).fit(x_train, y_train)
    matcher.calibrate_threshold(x_val, y_val, target_coverage)
    timings = []
    results = []
    for row in x_test:
        started = perf_counter()
        results.append(matcher.predict_one(row))
        timings.append((perf_counter() - started) * 1000.0)
    pred = np.array([r.label for r in results])
    labels = sorted(set(y_train) | set(y_test)) + ["UNKNOWN"]
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, labels=labels[:-1], average="macro", zero_division=0)
    covered = pred != "UNKNOWN"
    pair_errors = {}
    for left, right in similar_pairs or []:
        selected = (y_test == str(left)) | (y_test == str(right))
        pair_errors[f"{left}__{right}"] = {"videos": int(selected.sum()),
            "misclassified_or_rejected": int(np.sum(selected & (pred != y_test))),
            "labels": [str(left), str(right)]}
    return {"accuracy_including_rejects": float(np.mean(pred == y_test)), "precision_macro": float(p),
            "recall_macro": float(r), "f1_macro": float(f), "coverage": float(np.mean(covered)),
            "mean_matching_latency_ms": float(np.mean(timings)), "median_matching_latency_ms": float(np.median(timings)),
            "rejection_rate": float(1-np.mean(covered)), "confusion_matrix": confusion_matrix(y_test, pred, labels=labels).tolist(),
            "confusion_labels": labels, "per_class": classification_report(y_test, pred, labels=labels[:-1], output_dict=True, zero_division=0),
            "selected_pair_errors": pair_errors,
            "threshold_from_validation": matcher.rejection_threshold,
            "signer_limitation": "Signer-independent evaluation unavailable unless verified signer metadata is supplied."}
