"""Experimental V2 masked, group-weighted temporal landmark DTW matcher.

Only hand landmarks and the existing 151-value two-hand frame archive are used.
Scalers and exemplar prototypes must be fitted from the fit partition only.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .temporal_fingerprinting import FRAME_FEATURE_COUNT

GROUP_DIMS = {
    "shape_left": 70, "shape_right": 70,
    "position_relative_left": 3, "position_relative_right": 3,
    "position_absolute_left": 3, "position_absolute_right": 3,
    "interhand_position": 3,
    "shape_velocity_left": 70, "shape_velocity_right": 70,
    "position_velocity_left": 3, "position_velocity_right": 3,
    "interhand_velocity": 3,
    "direction_left": 2, "direction_right": 2, "interhand_direction": 2,
    "position_acceleration_left": 3, "position_acceleration_right": 3,
    "interhand_acceleration": 3,
    "presence": 2,
}

PROFILE_WEIGHTS = {
    "shape_preserving": {
        "shape_left": 3.0, "shape_right": 3.0,
        "position_relative_left": .3, "position_relative_right": .3,
        "position_absolute_left": .15, "position_absolute_right": .15,
        "interhand_position": .4,
        "shape_velocity_left": .25, "shape_velocity_right": .25,
        "position_velocity_left": .35, "position_velocity_right": .35,
        "interhand_velocity": .5,
        "direction_left": .2, "direction_right": .2, "interhand_direction": .3,
        "position_acceleration_left": .1, "position_acceleration_right": .1,
        "interhand_acceleration": .15, "presence": .5,
    },
    "balanced": {
        "shape_left": 1.0, "shape_right": 1.0,
        "position_relative_left": .65, "position_relative_right": .65,
        "position_absolute_left": .20, "position_absolute_right": .20,
        "interhand_position": 1.0,
        "shape_velocity_left": .45, "shape_velocity_right": .45,
        "position_velocity_left": .85, "position_velocity_right": .85,
        "interhand_velocity": 1.0,
        "direction_left": .45, "direction_right": .45, "interhand_direction": .6,
        "position_acceleration_left": .2, "position_acceleration_right": .2,
        "interhand_acceleration": .25, "presence": .75,
    },
    "motion_emphasis": {
        "shape_left": .8, "shape_right": .8,
        "position_relative_left": .8, "position_relative_right": .8,
        "position_absolute_left": .15, "position_absolute_right": .15,
        "interhand_position": 1.0,
        "shape_velocity_left": .65, "shape_velocity_right": .65,
        "position_velocity_left": 1.2, "position_velocity_right": 1.2,
        "interhand_velocity": 1.25,
        "direction_left": .9, "direction_right": .9, "interhand_direction": 1.0,
        "position_acceleration_left": .35, "position_acceleration_right": .35,
        "interhand_acceleration": .4, "presence": .75,
    },
}


@dataclass
class ComponentSequence:
    values: dict[str, np.ndarray]
    valid: dict[str, np.ndarray]


def _difference(values: np.ndarray, valid_now: np.ndarray, valid_previous: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros_like(values, dtype=np.float64)
    active = np.zeros(len(values), dtype=bool)
    if len(values) > 1:
        active[1:] = valid_now[1:] & valid_previous[:-1]
        out[1:] = values[1:] - values[:-1]
        out[~active] = 0.0
    return out, active


def make_components(frame_features: np.ndarray) -> ComponentSequence:
    """Derive shape, absolute and start-relative paths, masked motion, and hand relation.

    Missing-hand frames remain represented by presence masks. Motion values are
    emitted only across adjacent valid samples, preventing zero-fill boundaries
    from becoming artificial movement spikes.
    """
    x = np.asarray(frame_features, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != FRAME_FEATURE_COUNT:
        raise ValueError(f"Expected a (frames, {FRAME_FEATURE_COUNT}) array; got {x.shape}")
    if not np.isfinite(x).all():
        raise ValueError("Frame archive contains non-finite values")
    n = len(x)
    present_l = x[:, 146] > .5
    present_r = x[:, 147] > .5
    both = present_l & present_r
    shapes = [x[:, 0:70].copy(), x[:, 73:143].copy()]
    positions = [x[:, 70:73].copy(), x[:, 143:146].copy()]
    rel = x[:, 148:151].copy()
    valid = {
        "shape_left": present_l.copy(), "shape_right": present_r.copy(),
        "position_relative_left": present_l.copy(), "position_relative_right": present_r.copy(),
        "position_absolute_left": present_l.copy(), "position_absolute_right": present_r.copy(),
        "interhand_position": both.copy(), "presence": np.ones(n, dtype=bool),
    }
    values = {
        "shape_left": shapes[0], "shape_right": shapes[1],
        "position_absolute_left": positions[0].copy(), "position_absolute_right": positions[1].copy(),
        "interhand_position": rel,
        "presence": np.column_stack([present_l, present_r]).astype(np.float64),
    }
    # Remove initial camera placement as a nuisance channel while retaining the
    # complete wrist trajectory relative to the first observed position.
    for name, pos, mask in (("left", positions[0], present_l), ("right", positions[1], present_r)):
        relative = np.zeros_like(pos)
        if np.any(mask):
            first = pos[np.flatnonzero(mask)[0]].copy()
            relative[mask] = pos[mask] - first
        values[f"position_relative_{name}"] = relative

    for name, shape, pos, mask in (("left", shapes[0], positions[0], present_l),
                                   ("right", shapes[1], positions[1], present_r)):
        shape_vel, shape_valid = _difference(shape, mask, mask)
        pos_vel, pos_valid = _difference(pos, mask, mask)
        values[f"shape_velocity_{name}"] = shape_vel
        valid[f"shape_velocity_{name}"] = shape_valid
        values[f"position_velocity_{name}"] = pos_vel
        valid[f"position_velocity_{name}"] = pos_valid
        direction = np.zeros((n, 2), dtype=np.float64)
        speed = np.linalg.norm(pos_vel[:, :2], axis=1)
        direction_valid = pos_valid & (speed > 1e-9)
        direction[direction_valid] = pos_vel[direction_valid, :2] / speed[direction_valid, None]
        values[f"direction_{name}"] = direction
        valid[f"direction_{name}"] = direction_valid

        acceleration, acceleration_valid = _difference(pos_vel, pos_valid, pos_valid)
        values[f"position_acceleration_{name}"] = acceleration
        valid[f"position_acceleration_{name}"] = acceleration_valid

    inter_vel, inter_valid = _difference(rel, both, both)
    values["interhand_velocity"] = inter_vel
    valid["interhand_velocity"] = inter_valid
    inter_direction = np.zeros((n, 2), dtype=np.float64)
    inter_speed = np.linalg.norm(inter_vel[:, :2], axis=1)
    inter_dir_valid = inter_valid & (inter_speed > 1e-9)
    inter_direction[inter_dir_valid] = inter_vel[inter_dir_valid, :2] / inter_speed[inter_dir_valid, None]
    values["interhand_direction"] = inter_direction
    valid["interhand_direction"] = inter_dir_valid
    inter_acc, inter_acc_valid = _difference(inter_vel, inter_valid, inter_valid)
    values["interhand_acceleration"] = inter_acc
    valid["interhand_acceleration"] = inter_acc_valid

    for group, array in values.items():
        if array.shape != (n, GROUP_DIMS[group]):
            raise AssertionError(f"{group} has unexpected shape {array.shape}")
        array[~valid[group]] = 0.0
    return ComponentSequence(values, valid)


def _resample_array(x: np.ndarray, length: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((length, x.shape[1]), dtype=np.float64)
    if len(x) == 1:
        return np.repeat(x, length, axis=0)
    point = np.linspace(0, len(x) - 1, length)
    left = np.floor(point).astype(int)
    right = np.minimum(left + 1, len(x) - 1)
    alpha = (point - left)[:, None]
    return x[left] * (1 - alpha) + x[right] * alpha


def resample_components(sequence: ComponentSequence, length: int = 24) -> ComponentSequence:
    if length < 2:
        raise ValueError("length must be at least 2")
    n = len(next(iter(sequence.values.values())))
    grid = np.linspace(0, max(n - 1, 0), length)
    nearest = np.rint(grid).astype(int) if n else np.zeros(length, dtype=int)
    values, valid = {}, {}
    left = np.floor(grid).astype(int) if n else np.zeros(length, dtype=int)
    right = np.minimum(left + 1, max(n - 1, 0))
    alpha = (grid - left) if n else np.zeros(length)
    nearest = np.rint(grid).astype(int) if n else np.zeros(length, dtype=int)
    for group in GROUP_DIMS:
        mask = sequence.valid[group]
        valid[group] = mask[nearest].copy() if n else np.zeros(length, dtype=bool)
        source = sequence.values[group]
        sampled = np.zeros((length, GROUP_DIMS[group]), dtype=np.float64)
        if n:
            both_valid = mask[left] & mask[right]
            interp = (source[left] * (1 - alpha[:, None]) + source[right] * alpha[:, None])
            sampled[both_valid] = interp[both_valid]
            only_near = valid[group] & ~both_valid
            sampled[only_near] = source[nearest[only_near]]
        values[group] = sampled
        values[group][~valid[group]] = 0.0
    return ComponentSequence(values, valid)


@dataclass
class RobustGroupScaler:
    center: dict[str, np.ndarray]
    scale: dict[str, np.ndarray]

    @classmethod
    def fit(cls, sequences: Sequence[ComponentSequence]) -> "RobustGroupScaler":
        center, scale = {}, {}
        for group, width in GROUP_DIMS.items():
            if group == "presence":
                center[group] = np.zeros(width, dtype=np.float64)
                scale[group] = np.ones(width, dtype=np.float64)
                continue
            rows = [seq.values[group][seq.valid[group]] for seq in sequences if np.any(seq.valid[group])]
            if not rows:
                center[group] = np.zeros(width, dtype=np.float64)
                scale[group] = np.ones(width, dtype=np.float64)
                continue
            data = np.vstack(rows)
            center[group] = np.median(data, axis=0)
            q25, q75 = np.quantile(data, [.25, .75], axis=0)
            spread = q75 - q25
            spread[spread < 1e-8] = 1.0
            scale[group] = spread
        return cls(center, scale)

    def transform(self, sequence: ComponentSequence, length: int = 24) -> ComponentSequence:
        values = {}
        for group in GROUP_DIMS:
            z = np.zeros_like(sequence.values[group], dtype=np.float64)
            active = sequence.valid[group]
            if np.any(active):
                z[active] = (sequence.values[group][active] - self.center[group]) / self.scale[group]
            values[group] = z
        return resample_components(ComponentSequence(values, sequence.valid), length)


def local_weighted_cost(a: ComponentSequence, b: ComponentSequence,
                        weights: dict[str, float]) -> np.ndarray:
    """Pairwise frame costs with equalized group contribution and explicit weights.

    Inactive group values are zero after scaling. Presence remains a separate
    group, so a missing hand is distinguishable from a present zero-valued hand.
    """
    av = weighted_frame_features(a, weights)
    bv = weighted_frame_features(b, weights)
    a2 = np.sum(av * av, axis=1)[:, None]
    b2 = np.sum(bv * bv, axis=1)[None, :]
    return np.sqrt(np.maximum(a2 + b2 - 2 * (av @ bv.T), 0.0))


def weighted_frame_features(sequence: ComponentSequence, weights: dict[str, float]) -> np.ndarray:
    total = sum(weight for weight in weights.values() if weight > 0)
    parts = []
    for group, width in GROUP_DIMS.items():
        weight = weights.get(group, 0.0)
        if weight > 0:
            parts.append(sequence.values[group] * math.sqrt(weight / (width * total)))
    return np.concatenate(parts, axis=1)


def dtw_cost_matrix(costs: np.ndarray, window_ratio: float) -> float:
    """Constrained DTW over a precomputed local-cost matrix; normalized by path length."""
    c = np.asarray(costs, dtype=np.float64)
    if c.ndim != 2 or min(c.shape) == 0:
        return math.inf
    if not 0 <= window_ratio <= 1:
        raise ValueError("window_ratio must be in [0,1]")
    n, m = c.shape
    band = max(abs(n - m), int(math.ceil(window_ratio * max(n, m))))
    cost = np.full((n + 1, m + 1), np.inf, dtype=np.float64)
    length = np.zeros((n + 1, m + 1), dtype=np.int32)
    cost[0, 0] = 0.0
    # Fill anti-diagonals. Every cell on a diagonal depends only on the prior
    # one or two diagonals, allowing vectorized updates with the same recurrence.
    for diagonal in range(2, n + m + 1):
        i = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        j = diagonal - i
        keep = np.abs(i - j) <= band
        i, j = i[keep], j[keep]
        if not len(i):
            continue
        previous = np.stack((cost[i - 1, j], cost[i, j - 1], cost[i - 1, j - 1]), axis=1)
        choice = np.argmin(previous, axis=1)
        best = previous[np.arange(len(i)), choice]
        up_count = length[i - 1, j]
        left_count = length[i, j - 1]
        diag_count = length[i - 1, j - 1]
        prior_count = np.where(choice == 0, up_count, np.where(choice == 1, left_count, diag_count))
        cost[i, j] = c[i - 1, j - 1] + best
        length[i, j] = prior_count + 1
    return float(cost[n, m] / length[n, m]) if length[n, m] else math.inf


def dtw_cost_matrices(costs: np.ndarray, window_ratio: float) -> np.ndarray:
    """Batched DTW for many prototype cost matrices of equal sequence length."""
    c = np.asarray(costs, dtype=np.float64)
    if c.ndim != 3 or min(c.shape[1:]) == 0:
        raise ValueError("Batched costs must have shape (prototypes, frames, frames)")
    if not 0 <= window_ratio <= 1:
        raise ValueError("window_ratio must be in [0,1]")
    p, n, m = c.shape
    band = max(abs(n - m), int(math.ceil(window_ratio * max(n, m))))
    table = np.full((p, n + 1, m + 1), np.inf, dtype=np.float64)
    steps = np.zeros((p, n + 1, m + 1), dtype=np.int32)
    table[:, 0, 0] = 0.0
    for diagonal in range(2, n + m + 1):
        i = np.arange(max(1, diagonal - m), min(n, diagonal - 1) + 1)
        j = diagonal - i
        keep = np.abs(i - j) <= band
        i, j = i[keep], j[keep]
        if not len(i):
            continue
        previous = np.stack((table[:, i - 1, j], table[:, i, j - 1],
                             table[:, i - 1, j - 1]), axis=0)
        choice = np.argmin(previous, axis=0)
        best = np.take_along_axis(previous, choice[None, :, :], axis=0)[0]
        prior_steps = np.stack((steps[:, i - 1, j], steps[:, i, j - 1],
                                steps[:, i - 1, j - 1]), axis=0)
        count = np.take_along_axis(prior_steps, choice[None, :, :], axis=0)[0]
        table[:, i, j] = c[:, i - 1, j - 1] + best
        steps[:, i, j] = count + 1
    final_steps = steps[:, n, m]
    return np.divide(table[:, n, m], final_steps,
                     out=np.full(p, np.inf, dtype=np.float64), where=final_steps > 0)


def batch_local_costs(query: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Compute Euclidean frame-cost matrices for a batch of weighted prototypes."""
    q = np.asarray(query, dtype=np.float64)
    p = np.asarray(prototypes, dtype=np.float64)
    if p.ndim != 3 or q.ndim != 2 or p.shape[1:] != q.shape:
        raise ValueError(f"Expected prototypes=(n,{q.shape[0]},{q.shape[1]}), got {p.shape}")
    n_proto, length, _ = p.shape
    flat = p.reshape(-1, p.shape[-1])
    cross = (flat @ q.T).reshape(n_proto, length, length).transpose(0, 2, 1)
    q_norm = np.sum(q * q, axis=1)[None, :, None]
    p_norm = np.sum(p * p, axis=2)[:, None, :]
    return np.sqrt(np.maximum(q_norm + p_norm - 2 * cross, 0.0))


def _embedding(sequence: ComponentSequence, weights: dict[str, float]) -> np.ndarray:
    # A training-only summary for selecting diverse exemplars; never used to classify.
    parts = []
    for group, weight in weights.items():
        if group == "presence":
            parts.append(sequence.values[group].mean(axis=0) * math.sqrt(weight))
            continue
        mask = sequence.valid[group]
        mean = sequence.values[group][mask].mean(axis=0) if np.any(mask) else np.zeros(GROUP_DIMS[group])
        parts.append(mean * math.sqrt(weight / GROUP_DIMS[group]))
    return np.concatenate(parts)


def select_training_prototypes(records: Sequence[dict], sequences: Sequence[ComponentSequence],
                               count: int, weights: dict[str, float]) -> dict[str, list[ComponentSequence]]:
    by_label: dict[str, list[ComponentSequence]] = {}
    for row, seq in zip(records, sequences):
        if any(np.any(seq.valid[g]) for g in ("shape_left", "shape_right")):
            by_label.setdefault(str(row["gesture_label"]), []).append(seq)
    result = {}
    for label, examples in by_label.items():
        vectors = np.stack([_embedding(seq, weights) for seq in examples])
        center = vectors.mean(axis=0)
        first = int(np.argmin(np.sum((vectors - center) ** 2, axis=1)))
        selected = [first]
        while len(selected) < min(count, len(examples)):
            distances = np.full(len(examples), np.inf)
            for chosen in selected:
                distances = np.minimum(distances, np.sum((vectors - vectors[chosen]) ** 2, axis=1))
            distances[selected] = -np.inf
            selected.append(int(np.argmax(distances)))
        result[label] = [examples[i] for i in selected]
    return result


def score_validation(validation_records: Sequence[dict], validation_sequences: Sequence[ComponentSequence],
                     prototypes: dict[str, list[ComponentSequence]], weights: dict[str, float],
                     window_ratio: float, prototype_count: int) -> dict:
    """Return nearest-class validation scores and per-row prediction latency."""
    labels = sorted(prototypes)
    y_true, nearest, distances, latency = [], [], [], []
    for row, sequence in zip(validation_records, validation_sequences):
        started = time.perf_counter()
        class_distances = {}
        for label in labels:
            values = [dtw_cost_matrix(local_weighted_cost(sequence, proto, weights), window_ratio)
                      for proto in prototypes[label][:prototype_count]]
            class_distances[label] = min(values)
        winner = min(class_distances, key=class_distances.get)
        latency.append((time.perf_counter() - started) * 1000.0)
        y_true.append(str(row["gesture_label"]))
        nearest.append(winner)
        distances.append(float(class_distances[winner]))
    return {"truth": y_true, "nearest": nearest, "distance": distances, "latency_ms": latency}


def score_validation_grid(validation_records: Sequence[dict], validation_sequences: Sequence[ComponentSequence],
                          prototypes: dict[str, list[ComponentSequence]], weights: dict[str, float],
                          windows: Sequence[float], prototype_counts: Sequence[int],
                          preprocessing_ms: Sequence[float] | None = None) -> dict[tuple[int, float], dict]:
    """Score a compact grid while calculating each local frame cost only once."""
    labels = sorted(prototypes)
    max_k = max(prototype_counts)
    configs = [(int(k), float(w)) for k in prototype_counts for w in windows]
    predictions = {cfg: {"nearest": [], "distance": [], "latency_ms": []} for cfg in configs}
    ordered_prototypes = [(label, prototype) for label in labels for prototype in prototypes[label][:max_k]]
    all_proto_features = np.stack([weighted_frame_features(seq, weights)
                                   for _, seq in ordered_prototypes])
    proto_labels = [label for label, _ in ordered_prototypes]
    selected_indices = {
        int(k): np.asarray([i for i, label in enumerate(proto_labels)
                            if sum(1 for prior in proto_labels[:i + 1] if prior == label) <= int(k)], dtype=int)
        for k in prototype_counts
    }
    if preprocessing_ms is None:
        preprocessing_ms = [0.0] * len(validation_records)
    if len(preprocessing_ms) != len(validation_records):
        raise ValueError("preprocessing_ms must align one-to-one with validation records")
    for row_index, (row, sequence) in enumerate(zip(validation_records, validation_sequences)):
        query_started = time.perf_counter()
        query = weighted_frame_features(sequence, weights)
        query_ms = (time.perf_counter() - query_started) * 1000.0
        for k in prototype_counts:
            local_started = time.perf_counter()
            indices = selected_indices[int(k)]
            local = batch_local_costs(query, all_proto_features[indices])
            local_ms = (time.perf_counter() - local_started) * 1000.0
            candidate_labels = [proto_labels[i] for i in indices]
            for window in windows:
                dtw_started = time.perf_counter()
                distances = dtw_cost_matrices(local, float(window))
                dtw_ms = (time.perf_counter() - dtw_started) * 1000.0
                by_class: dict[str, float] = {}
                for label, distance in zip(candidate_labels, distances):
                    by_class[label] = min(by_class.get(label, math.inf), float(distance))
                winner = min(by_class, key=by_class.get)
                cfg = (int(k), float(window))
                predictions[cfg]["nearest"].append(winner)
                predictions[cfg]["distance"].append(float(by_class[winner]))
                predictions[cfg]["latency_ms"].append(
                    float(preprocessing_ms[row_index]) + query_ms + local_ms + dtw_ms)
    truth = [str(row["gesture_label"]) for row in validation_records]
    return {cfg: {"truth": truth, **data} for cfg, data in predictions.items()}


def calibrate_threshold(truth: Sequence[str], nearest: Sequence[str], distances: Sequence[float],
                        correct_target_coverage: float = .95) -> float:
    correct = sorted(float(d) for y, p, d in zip(truth, nearest, distances) if y == p)
    if not correct:
        raise ValueError("No correctly classified validation rows; cannot calibrate threshold")
    index = min(max(math.ceil(correct_target_coverage * len(correct)) - 1, 0), len(correct) - 1)
    return correct[index]


def metric_report(truth: Sequence[str], nearest: Sequence[str], distances: Sequence[float],
                  threshold: float, latency_ms: Sequence[float], class_labels: Sequence[str]) -> dict:
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    predicted = [p if float(d) <= threshold else "UNKNOWN" for p, d in zip(nearest, distances)]
    p, r, f, support = precision_recall_fscore_support(
        truth, predicted, labels=list(class_labels), average=None, zero_division=0)
    accepted = np.asarray([label != "UNKNOWN" for label in predicted])
    return {
        "sample_count": len(truth), "accepted_count": int(accepted.sum()),
        "accuracy": float(accuracy_score(truth, predicted)),
        "precision_macro": float(np.mean(p)), "recall_macro": float(np.mean(r)),
        "f1_macro": float(np.mean(f)), "coverage": float(accepted.mean()),
        "rejection_rate": float((~accepted).mean()),
        "accuracy_accepted": float(accuracy_score(np.asarray(truth)[accepted], np.asarray(predicted)[accepted])) if accepted.any() else 0.0,
        "confusion_labels": list(class_labels) + ["UNKNOWN"],
        "confusion_matrix": confusion_matrix(truth, predicted, labels=list(class_labels) + ["UNKNOWN"]).tolist(),
        "per_class": [{"gesture_label": label, "precision": float(p[i]), "recall": float(r[i]),
                       "f1_score": float(f[i]), "support": int(support[i])}
                      for i, label in enumerate(class_labels)],
        "truth": list(truth), "nearest_label": list(nearest), "predicted": predicted,
        "nearest_distance": [float(x) for x in distances],
        "matching_latency_ms": [float(x) for x in latency_ms],
        "matching_latency_mean_ms": float(np.mean(latency_ms)),
        "matching_latency_p95_ms": float(np.percentile(latency_ms, 95)),
        "threshold_distance_units": float(threshold), "threshold_is_probability": False,
    }


def save_model_artifacts(path: str | Path, labels: Sequence[str],
                         prototypes: dict[str, list[ComponentSequence]], scaler: RobustGroupScaler,
                         prototype_count: int, profile: str, window_ratio: float,
                         threshold: float, sequence_length: int = 24) -> None:
    """Save frozen scaler and selected training prototypes without pickle."""
    labels = list(labels)
    protos = [(label, seq) for label in labels for seq in prototypes[label][:prototype_count]]
    arrays: dict[str, np.ndarray] = {
        "class_labels": np.asarray(labels, dtype=str),
        "prototype_labels": np.asarray([label for label, _ in protos], dtype=str),
        "profile": np.asarray(profile), "window_ratio": np.asarray(window_ratio),
        "prototype_count": np.asarray(prototype_count), "sequence_length": np.asarray(sequence_length),
        "rejection_threshold": np.asarray(threshold),
        "feature_group_names": np.asarray(list(GROUP_DIMS), dtype=str),
        "feature_group_weights": np.asarray([PROFILE_WEIGHTS[profile][group] for group in GROUP_DIMS], dtype=np.float64),
    }
    for group in GROUP_DIMS:
        arrays[f"scaler_center__{group}"] = scaler.center[group]
        arrays[f"scaler_scale__{group}"] = scaler.scale[group]
        arrays[f"prototype_values__{group}"] = np.stack([seq.values[group] for _, seq in protos])
        arrays[f"prototype_valid__{group}"] = np.stack([seq.valid[group] for _, seq in protos]).astype(np.uint8)
    np.savez_compressed(path, **arrays)

