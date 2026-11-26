"""70-value hand fingerprint: wrist-relative, palm-scale normalized geometry."""
from __future__ import annotations
import math
from typing import Sequence

import numpy as np

LANDMARK_COUNT = 21
FEATURE_COUNT = 70
# Landmark triples are in MediaPipe Hands order. Each angle uses the proximal,
# middle, and distal joints (thumb: 1,2,4; fingers: MCP,PIP,tip).
ANGLE_TRIPLETS = ((1, 2, 4), (5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20))
FEATURE_NAMES = ([f"coord_{i}_{axis}" for i in range(1, 21) for axis in "xyz"] +
                 [f"angle_{name}" for name in ("thumb", "index", "middle", "ring", "pinky")] +
                 [f"wrist_tip_distance_{name}" for name in ("thumb", "index", "middle", "ring", "pinky")])


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def joint_angle(a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> float:
    """Interior angle at b, mapped from radians to [0, 1]."""
    u, v = np.asarray(a, float) - b, np.asarray(c, float) - b
    denom = np.linalg.norm(u) * np.linalg.norm(v)
    if denom < 1e-12:
        return 0.0
    cosine = np.clip(np.dot(u, v) / denom, -1.0, 1.0)
    return float(math.acos(cosine) / math.pi)


def normalize_landmarks(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Translate wrist to zero and divide xyz by wrist-to-middle-MCP scale."""
    points = np.asarray(landmarks, dtype=float)
    if points.shape != (LANDMARK_COUNT, 3) or not np.isfinite(points).all():
        raise ValueError("Expected 21 finite landmarks, each with x, y, z")
    relative = points - points[0]
    scale = np.linalg.norm(relative[9])
    if not np.isfinite(scale) or scale < 1e-8:
        raise ValueError("Invalid hand scale: wrist and middle-finger MCP coincide")
    return relative / scale


def make_fingerprint(landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    p = normalize_landmarks(landmarks)
    coords = p[1:].reshape(-1)
    angles = np.array([joint_angle(p[a], p[b], p[c]) for a, b, c in ANGLE_TRIPLETS])
    tips = (4, 8, 12, 16, 20)
    distances = np.array([euclidean_distance(p[0], p[i]) for i in tips])
    result = np.concatenate((coords, angles, distances))
    if result.shape != (FEATURE_COUNT,):
        raise AssertionError(f"Expected {FEATURE_COUNT} features, got {result.size}")
    return result


def aggregate_fingerprints(rows: Sequence[Sequence[float]]) -> np.ndarray:
    """Coordinate-wise median across frames of a single source video."""
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != FEATURE_COUNT or values.shape[0] == 0:
        raise ValueError("Need one or more valid 70-feature frame fingerprints")
    return np.median(values, axis=0)
