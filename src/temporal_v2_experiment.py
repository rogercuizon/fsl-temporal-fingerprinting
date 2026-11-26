"""Train/select the separate temporal V2 experiment on fit/validation data only.

Run from the repository root with: python -m src.temporal_v2_experiment
This command never reads the official test manifest, test sequences, or test predictions.
"""
from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .temporal_fingerprinting import load_sequence_archive, split_training_rows
from .temporal_fingerprinting_v2 import (
    PROFILE_WEIGHTS, RobustGroupScaler, calibrate_threshold, make_components,
    metric_report, save_model_artifacts, score_validation_grid, select_training_prototypes,
)


ROOT = Path(__file__).resolve().parents[1]
V1_DIR = ROOT / "output" / "temporal_v1"
DATA_DIR = ROOT / "output" / "mobile_v2" / "dataset"
OUT_DIR = ROOT / "output" / "temporal_v2"
EXCLUSION_CSV = V1_DIR / "duplicate_sample_exclusions.csv"
TRAIN_MANIFEST = DATA_DIR / "train_manifest.csv"
TRAIN_ARCHIVE = V1_DIR / "training_sequences.npz"
V1_VALIDATION = V1_DIR / "validation_predictions.csv"
V1_ARTIFACTS = V1_DIR / "temporal_model_artifacts.npz"

SEED = 42
VALIDATION_FRACTION = .20
TARGET_CORRECT_VALIDATION_COVERAGE = .95
SEQUENCE_LENGTH = 24
PROFILES = tuple(PROFILE_WEIGHTS)
WINDOW_RATIOS = (.10, .25)
PROTOTYPE_COUNTS = (1, 2)


def canonical(path: object) -> str:
    return str(path).strip().replace("/", "\\").casefold()


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer columns for empty output: {path}")
    columns = fieldnames or list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in (TRAIN_MANIFEST, TRAIN_ARCHIVE, V1_VALIDATION, V1_ARTIFACTS, EXCLUSION_CSV):
        if not path.is_file():
            raise FileNotFoundError(f"Required fit/validation-only input is missing: {path}")

    manifest = pd.read_csv(TRAIN_MANIFEST, dtype=str, keep_default_na=False)
    required = {"video_path", "class_id", "gesture_label", "official_split"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"Training manifest is missing columns: {sorted(required-set(manifest.columns))}")
    if set(manifest["official_split"]) != {"train"}:
        raise ValueError("Training manifest must contain official_split=train only")
    excluded_df = pd.read_csv(EXCLUSION_CSV, dtype=str, keep_default_na=False)
    exclusion_col = "training_video_path" if "training_video_path" in excluded_df else "training_video_path"
    if exclusion_col not in excluded_df.columns:
        raise ValueError("Duplicate exclusion CSV must contain training_video_path")
    excluded_paths = {canonical(p) for p in excluded_df[exclusion_col]}
    train_rows = [r for r in manifest.to_dict("records") if canonical(r["video_path"]) not in excluded_paths]
    if len(train_rows) != 1702 or len(excluded_paths) != 2:
        raise ValueError(f"Expected documented clean train set=1702 / exclusions=2; got {len(train_rows)} / {len(excluded_paths)}")

    fit_rows, validation_rows = split_training_rows(
        train_rows, validation_fraction=VALIDATION_FRACTION, seed=SEED)
    if (len(fit_rows), len(validation_rows)) != (1361, 341):
        raise AssertionError(f"Unexpected fit/validation sizes: {len(fit_rows)}, {len(validation_rows)}")
    archive_rows = load_sequence_archive(TRAIN_ARCHIVE)
    by_path = {canonical(row["video_path"]): row for row in archive_rows}
    if len(by_path) != len(archive_rows):
        raise ValueError("Training sequence archive has duplicate video paths")
    train_paths = {canonical(row["video_path"]) for row in train_rows}
    if not train_paths.issubset(by_path):
        raise ValueError(f"Training archive is missing {len(train_paths-set(by_path))} clean training rows")
    if any(by_path[canonical(row["video_path"])]["official_split"] != "train" for row in train_rows):
        raise ValueError("Non-training sequence found in training archive lookup")
    validation_v1 = pd.read_csv(V1_VALIDATION, dtype={"video_path": str, "class_id": str})
    expected_val_paths = {canonical(row["video_path"]) for row in validation_rows}
    if {canonical(p) for p in validation_v1["video_path"]} != expected_val_paths:
        raise ValueError("Recreated validation partition does not match saved Version 1 validation paths")
    val_labels_v1 = {canonical(row.video_path): str(row.true_label) for row in validation_v1.itertuples()}
    if any(str(row["gesture_label"]) != val_labels_v1[canonical(row["video_path"])] for row in validation_rows):
        raise ValueError("Recreated validation labels disagree with saved Version 1 validation records")

    print(f"Clean official training rows: {len(train_rows)}; excluded duplicate train paths: {len(excluded_paths)}")
    print(f"Fit rows: {len(fit_rows)}; validation rows: {len(validation_rows)}")
    print("Validation path set exactly matches saved V1 validation predictions.")
    print("No official test manifest/features/predictions are loaded by this script.")

    fit_records = fit_rows
    val_records = validation_rows
    fit_raw = [by_path[canonical(row["video_path"])]["features"] for row in fit_records]
    val_raw = [by_path[canonical(row["video_path"])]["features"] for row in val_records]
    fit_components_raw = [make_components(x) for x in fit_raw]
    scaler = RobustGroupScaler.fit(fit_components_raw)
    fit_components = [scaler.transform(x, SEQUENCE_LENGTH) for x in fit_components_raw]
    val_components = []
    val_preprocessing_ms = []
    for raw in val_raw:
        started = time.perf_counter()
        val_components.append(scaler.transform(make_components(raw), SEQUENCE_LENGTH))
        val_preprocessing_ms.append((time.perf_counter() - started) * 1000.0)
    all_labels = sorted({str(row["gesture_label"]) for row in train_rows})

    candidate_results = []
    candidate_records = {}
    candidate_predictions = {}
    candidate_scores = {}
    selected_prototypes = None
    selected_scaler = scaler
    for profile_name in PROFILES:
        weights = PROFILE_WEIGHTS[profile_name]
        prototypes = select_training_prototypes(fit_records, fit_components, max(PROTOTYPE_COUNTS), weights)
        missing = sorted(set(all_labels) - set(prototypes))
        if missing:
            raise RuntimeError(f"No valid fit reference sequences for classes: {missing}")
        grid = score_validation_grid(val_records, val_components, prototypes, weights,
                                     WINDOW_RATIOS, PROTOTYPE_COUNTS, val_preprocessing_ms)
        for k in PROTOTYPE_COUNTS:
            for window in WINDOW_RATIOS:
                name = f"{profile_name}_k{k}_w{window:.2f}"
                print(f"Scoring validation candidate {name} ...", flush=True)
                scored = grid[(k, window)]
                candidate_scores[name] = scored
                threshold = calibrate_threshold(
                    scored["truth"], scored["nearest"], scored["distance"],
                    correct_target_coverage=TARGET_CORRECT_VALIDATION_COVERAGE)
                metrics = metric_report(scored["truth"], scored["nearest"], scored["distance"],
                                        threshold, scored["latency_ms"], all_labels)
                metrics.update({"candidate": name, "feature_profile": profile_name,
                                "prototype_count_per_class_max": k, "window_ratio": window,
                                "prototype_total": sum(min(k, len(v)) for v in prototypes.values())})
                candidate_results.append(metrics)
                candidate_records[name] = metrics
                candidate_predictions[name] = pd.DataFrame({
                    "video_path": [str(r["video_path"]) for r in val_records],
                    "class_id": [str(r["class_id"]) for r in val_records],
                    "true_label": scored["truth"], "nearest_label": scored["nearest"],
                    "predicted_label": metrics["predicted"], "nearest_distance": scored["distance"],
                    "matching_latency_ms": scored["latency_ms"],
                })

    # Use the best validation macro-F1; break ties by accuracy, then coverage.
    ordered = sorted(candidate_results,
                     key=lambda r: (r["f1_macro"], r["accuracy"], r["coverage"], -r["matching_latency_mean_ms"]),
                     reverse=True)
    winner = ordered[0]
    selected_name = winner["candidate"]
    profile = winner["feature_profile"]
    k = int(winner["prototype_count_per_class_max"])
    window = float(winner["window_ratio"])
    weights = PROFILE_WEIGHTS[profile]
    selected_prototypes = select_training_prototypes(fit_records, fit_components, k, weights)

    # Use the cached validation scores; all V1/V2 comparisons share identical rows.
    selected_scores = candidate_scores[selected_name]
    selected_threshold = calibrate_threshold(selected_scores["truth"], selected_scores["nearest"],
                                             selected_scores["distance"], TARGET_CORRECT_VALIDATION_COVERAGE)
    selected_metrics = metric_report(selected_scores["truth"], selected_scores["nearest"],
                                     selected_scores["distance"], selected_threshold,
                                     selected_scores["latency_ms"], all_labels)
    selected_metrics.update({"candidate": selected_name, "feature_profile": profile,
                             "prototype_count_per_class_max": k, "window_ratio": window,
                             "prototype_total": sum(len(v) for v in selected_prototypes.values())})

    # Version 1 is taken from the untouched saved validation prediction file and its frozen threshold.
    with np.load(V1_ARTIFACTS, allow_pickle=False) as v1_art:
        v1_labels = [str(x) for x in v1_art["class_labels"].tolist()]
        v1_threshold = float(v1_art["rejection_threshold"])
    v1_pred = validation_v1.set_index(validation_v1.video_path.map(canonical)).loc[
        [canonical(row["video_path"]) for row in validation_rows]].reset_index(drop=True)
    v1_truth = v1_pred["true_label"].astype(str).tolist()
    v1_predictions = v1_pred["predicted_label"].astype(str).tolist()
    from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
    v1_p, v1_r, v1_f, v1_support = precision_recall_fscore_support(
        v1_truth, v1_predictions, labels=v1_labels, average=None, zero_division=0)
    v1_accepted = np.asarray([p != "UNKNOWN" for p in v1_predictions])
    v1_metrics = {
        "sample_count": len(v1_truth), "accepted_count": int(v1_accepted.sum()),
        "accuracy": float(accuracy_score(v1_truth, v1_predictions)),
        "precision_macro": float(np.mean(v1_p)), "recall_macro": float(np.mean(v1_r)),
        "f1_macro": float(np.mean(v1_f)), "coverage": float(v1_accepted.mean()),
        "rejection_rate": float((~v1_accepted).mean()),
        "accuracy_accepted": float(accuracy_score(np.asarray(v1_truth)[v1_accepted],
                                                    np.asarray(v1_predictions)[v1_accepted])),
        "matching_latency_mean_ms": float(v1_pred["matching_latency_ms"].mean()),
        "matching_latency_p95_ms": float(v1_pred["matching_latency_ms"].quantile(.95)),
        "threshold_distance_units": v1_threshold,
        "confusion_labels": v1_labels + ["UNKNOWN"],
        "confusion_matrix": confusion_matrix(v1_truth, v1_predictions,
                                               labels=v1_labels + ["UNKNOWN"]).tolist(),
        "per_class": [{"gesture_label": label, "precision": float(v1_p[i]),
                       "recall": float(v1_r[i]), "f1_score": float(v1_f[i]),
                       "support": int(v1_support[i])} for i, label in enumerate(v1_labels)],
    }
    if len(v1_truth) != len(selected_metrics["truth"]) or v1_truth != selected_metrics["truth"]:
        raise AssertionError("V1 and V2 validation rows do not have identical ordered labels")

    compact = [v1_metrics] + candidate_results
    comparison = pd.DataFrame([{key: row[key] for key in (
        "candidate", "sample_count", "accuracy", "precision_macro", "recall_macro", "f1_macro",
        "coverage", "rejection_rate", "accuracy_accepted", "matching_latency_mean_ms",
        "matching_latency_p95_ms", "threshold_distance_units", "feature_profile",
        "prototype_count_per_class_max", "window_ratio", "prototype_total") if key in row}
        for row in compact])
    comparison["candidate"] = comparison["candidate"].fillna("Version 1 (saved)")
    comparison.to_csv(OUT_DIR / "validation_comparison.csv", index=False)
    (OUT_DIR / "candidate_metrics.json").write_text(
        json.dumps([{k: v for k, v in row.items() if k not in {"confusion_matrix", "per_class", "truth", "nearest_label", "predicted", "nearest_distance", "matching_latency_ms"}}
                    for row in candidate_results], indent=2), encoding="utf-8")

    selected_dir = OUT_DIR / "selected_validation"
    selected_dir.mkdir(parents=True, exist_ok=True)
    selected_pred = candidate_predictions[selected_name].copy()
    # Use the fresh pass's exactly aligned predictions/latencies.
    selected_pred["nearest_label"] = selected_scores["nearest"]
    selected_pred["predicted_label"] = selected_metrics["predicted"]
    selected_pred["nearest_distance"] = selected_scores["distance"]
    selected_pred["matching_latency_ms"] = selected_scores["latency_ms"]
    selected_pred.to_csv(selected_dir / "validation_predictions.csv", index=False)
    pd.DataFrame(selected_metrics["confusion_matrix"],
                 index=selected_metrics["confusion_labels"],
                 columns=selected_metrics["confusion_labels"]).to_csv(
                     selected_dir / "validation_confusion_matrix.csv", index_label="actual_label")
    pd.DataFrame(selected_metrics["per_class"]).to_csv(selected_dir / "validation_per_class_metrics.csv", index=False)

    # Keep all candidate validation details so comparisons can be audited.
    candidate_root = OUT_DIR / "candidates"
    candidate_root.mkdir(exist_ok=True)
    for name, result in candidate_records.items():
        folder = candidate_root / name
        folder.mkdir(exist_ok=True)
        candidate_predictions[name].to_csv(folder / "validation_predictions.csv", index=False)
        pd.DataFrame(result["confusion_matrix"], index=result["confusion_labels"],
                     columns=result["confusion_labels"]).to_csv(folder / "confusion_matrix.csv", index_label="actual_label")
        pd.DataFrame(result["per_class"]).to_csv(folder / "per_class_metrics.csv", index=False)

    save_model_artifacts(OUT_DIR / "selected_model_artifacts.npz", all_labels,
                         selected_prototypes, selected_scaler, k, profile, window,
                         selected_threshold, SEQUENCE_LENGTH)
    metadata = {
        "version": "temporal_v2_exploratory",
        "selected_candidate": selected_name,
        "feature_profile": profile,
        "feature_group_weights": weights,
        "feature_definition": "existing palm-scale-normalized local hand shape; first-wrist-relative and absolute wrist paths; masked shape/position/inter-hand velocity; normalized XY movement direction; acceleration; two-hand presence/inter-hand relation",
        "missing_hand_policy": "presence masks retained; motion only computed across adjacent valid detections; absent group values remain zero after fit-only scaling",
        "distance": "presence-aware group-balanced weighted Euclidean local cost plus path-length-normalized constrained DTW",
        "prototype_selection": "training-only first medoid in robust feature-summary space plus farthest diverse training exemplars",
        "prototype_count_per_class_max": k,
        "window_ratio": window,
        "sequence_length": SEQUENCE_LENGTH,
        "unknown_threshold_distance_units": selected_threshold,
        "threshold_is_probability": False,
        "threshold_selection": "95th percentile rank among correctly classified validation nearest-prototype distances only",
        "validation_fraction": VALIDATION_FRACTION,
        "split_seed": SEED,
        "clean_official_training_rows": len(train_rows),
        "documented_training_duplicate_exclusions": sorted(excluded_paths),
        "fit_rows": len(fit_rows), "validation_rows": len(validation_rows),
        "validation_paths_match_saved_v1": True,
        "official_test_manifest_or_media_loaded": False,
        "official_test_used_for_model_or_selection": False,
        "validation_metrics": {key: selected_metrics[key] for key in (
            "sample_count", "accepted_count", "accuracy", "precision_macro", "recall_macro", "f1_macro",
            "coverage", "rejection_rate", "accuracy_accepted", "matching_latency_mean_ms",
            "matching_latency_p95_ms")},
        "research_status": "exploratory; official test has already been inspected and must not be used for V2 selection",
    }
    (OUT_DIR / "selected_model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    def top_pairs(metric_row: dict) -> list[dict]:
        classes = metric_row["confusion_labels"]
        matrix = np.asarray(metric_row["confusion_matrix"])
        pairs = []
        for i, actual in enumerate(classes):
            for j, predicted in enumerate(classes):
                if i != j and actual != "UNKNOWN" and predicted != "UNKNOWN" and matrix[i, j] > 0:
                    pairs.append({"actual_label": actual, "predicted_label": predicted,
                                  "count": int(matrix[i, j])})
        return sorted(pairs, key=lambda x: (-x["count"], x["actual_label"], x["predicted_label"]))[:10]
    write_csv(OUT_DIR / "version1_top_confusions.csv", top_pairs(v1_metrics),
              ["actual_label", "predicted_label", "count"])
    write_csv(selected_dir / "top_confusions.csv", top_pairs(selected_metrics),
              ["actual_label", "predicted_label", "count"])

    delta = {key: selected_metrics[key] - v1_metrics[key] for key in (
        "accuracy", "precision_macro", "recall_macro", "f1_macro", "coverage", "accuracy_accepted")}
    selected_per_class = pd.DataFrame(selected_metrics["per_class"])
    v1_per_class = pd.DataFrame(v1_metrics["per_class"])
    candidate_text = comparison.to_string(index=False, float_format=lambda value: f"{value:.4f}")
    best_v2_row = next(row for row in candidate_results if row["candidate"] == selected_name)
    shape_k1 = next(row for row in candidate_results if row["feature_profile"] == "shape_preserving"
                    and row["prototype_count_per_class_max"] == 1 and row["window_ratio"] == .25)
    shape_k2_narrow = next(row for row in candidate_results if row["feature_profile"] == "shape_preserving"
                           and row["prototype_count_per_class_max"] == 2 and row["window_ratio"] == .10)
    best_v2_top, v1_top = top_pairs(selected_metrics), top_pairs(v1_metrics)
    selected_high = selected_per_class.sort_values(["recall", "support"], ascending=[False, False]).head(8)
    selected_low = selected_per_class.sort_values(["recall", "support"], ascending=[True, False]).head(8)
    v1_high = v1_per_class.sort_values(["recall", "support"], ascending=[False, False]).head(8)
    v1_low = v1_per_class.sort_values(["recall", "support"], ascending=[True, False]).head(8)
    v1_per_class.merge(selected_per_class, on="gesture_label", suffixes=("_v1", "_v2")).to_csv(
        OUT_DIR / "version1_vs_selected_v2_per_class.csv", index=False)
    write_csv(OUT_DIR / "selected_v2_top_confusions.csv", best_v2_top,
              ["actual_label", "predicted_label", "count"])
    lines = [
        "TEMPORAL FINGERPRINT VERSION 2 — FIT/VALIDATION REPORT",
        "V2 is exploratory; official test results were not loaded or used for selection.",
        "",
        f"Clean training rows={len(train_rows)} (two documented training duplicates excluded); fit={len(fit_rows)}; validation={len(validation_rows)}.",
        "The validation path/label set exactly matches the saved Version 1 validation predictions (341 videos).",
        f"Version 1 saved validation: accuracy={v1_metrics['accuracy']:.6f}, macro precision={v1_metrics['precision_macro']:.6f}, macro recall={v1_metrics['recall_macro']:.6f}, macro F1={v1_metrics['f1_macro']:.6f}, coverage={v1_metrics['coverage']:.6f}, accepted accuracy={v1_metrics['accuracy_accepted']:.6f}, mean latency={v1_metrics['matching_latency_mean_ms']:.3f} ms, p95={v1_metrics['matching_latency_p95_ms']:.3f} ms, threshold={v1_threshold:.8f} DTW units.",
        f"Selected V2 candidate={selected_name}; profile={profile}; prototypes/class max={k}; warping window={window:.2f}; threshold={selected_threshold:.8f} DTW units.",
        f"V2 validation: accuracy={selected_metrics['accuracy']:.6f}, macro precision={selected_metrics['precision_macro']:.6f}, macro recall={selected_metrics['recall_macro']:.6f}, macro F1={selected_metrics['f1_macro']:.6f}, coverage={selected_metrics['coverage']:.6f}, accepted accuracy={selected_metrics['accuracy_accepted']:.6f}, mean latency={selected_metrics['matching_latency_mean_ms']:.3f} ms, p95={selected_metrics['matching_latency_p95_ms']:.3f} ms.",
        "V2 minus V1 validation deltas: " + ", ".join(f"{key}={value:+.6f}" for key, value in delta.items()),
        "",
        "All-candidate comparison (same 341 validation videos):",
        candidate_text,
        "",
        "Observed validation effects within the Version 2 grid:",
        f"- Shape-preserving k=2/window=0.25 vs shape-preserving k=1/window=0.25: accuracy {shape_k1['accuracy']:.6f} -> {best_v2_row['accuracy']:.6f}, macro-F1 {shape_k1['f1_macro']:.6f} -> {best_v2_row['f1_macro']:.6f}; mean latency {shape_k1['matching_latency_mean_ms']:.2f} -> {best_v2_row['matching_latency_mean_ms']:.2f} ms.",
        f"- Shape-preserving k=2/window=0.25 vs k=2/window=0.10: accuracy {shape_k2_narrow['accuracy']:.6f} -> {best_v2_row['accuracy']:.6f}, macro-F1 {shape_k2_narrow['f1_macro']:.6f} -> {best_v2_row['f1_macro']:.6f}.",
        "- The shape-preserving profile outperformed balanced and motion-emphasis profiles in this validation grid. This is evidence about this split only, not proof that motion is unhelpful generally.",
        "- No V2 candidate exceeded Version 1 validation accuracy or macro-F1. V2 had higher macro precision, but lower accuracy, macro recall, macro-F1, coverage, and accepted-only accuracy. Do not replace Version 1 based on this run.",
        "- V2 matching latency includes component construction, fit-only scaling/resampling, weighted frame-cost calculation, and DTW on cached landmark sequences. It excludes video decoding and MediaPipe extraction; timing is hardware/runtime dependent.",
        "",
        "Top Version 1 validation confusions (actual -> predicted; no similarity claim):",
        pd.DataFrame(v1_top).to_string(index=False) if v1_top else "No accepted class confusions.",
        "Top selected Version 2 validation confusions (actual -> predicted; no similarity claim):",
        pd.DataFrame(best_v2_top).to_string(index=False) if best_v2_top else "No accepted class confusions.",
        "Selected V2 classes with highest validation recall (support is small):",
        selected_high.to_string(index=False),
        "Selected V2 classes with lowest validation recall (support is small):",
        selected_low.to_string(index=False),
        "Version 1 classes with highest validation recall:", v1_high.to_string(index=False),
        "Version 1 classes with lowest validation recall:", v1_low.to_string(index=False),
        "",
        "All candidate metrics, confusion matrices, per-class metrics, and predictions are stored under candidates/.",
        "Selected model scaler and training-only prototypes are in selected_model_artifacts.npz.",
        "Threshold is a DTW distance cutoff, not a calibrated confidence probability.",
        "Confused label pairs are not declared visually similar without independent visual annotations.",
        "The official test set has already been inspected; no V2 score on it is produced. An independent final test set is needed for confirmatory comparison.",
        "No signer IDs, semantic one-/two-hand labels, or static/dynamic annotations are available.",
        "Unknown-sign rejection cannot be validated using the current known-class validation set.",
    ]
    (OUT_DIR / "validation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:9]))
    print("\\nValidation candidate comparison:")
    print(comparison.to_string(index=False))
    print(f"\\nSaved Version 2 experiment to: {OUT_DIR}")


if __name__ == "__main__":
    main()

