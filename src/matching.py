"""Official-split-safe landmark prototype matching and validation selection.

Official test features are never parsed by the preparation pipeline. Feature scaling
and class prototypes are fitted on an internal split of official training videos;
the rejection threshold is calibrated on the held-out validation portion only.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from collections import Counter, defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

from .fingerprint import FEATURE_COUNT, FEATURE_NAMES

DISTANCE_METHODS = ("euclidean", "training_scaled", "robust_scaled", "group_balanced")
PROTOTYPE_METHODS = ("mean", "median")
GROUPS = ((0, 60), (60, 65), (65, 70))
GROUP_NAMES = ("coordinates", "angles", "fingertip_distances")


def canonical_path(path: str) -> str:
    """Normalize manifest/workspace path separators and case for matching only."""
    return path.strip().replace("/", "\\").casefold()


def read_rows(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def feature_vector(row: dict) -> np.ndarray:
    values = np.asarray([float(row[f"feature_{i}"]) for i in range(1, FEATURE_COUNT + 1)], dtype=float)
    if values.shape != (FEATURE_COUNT,) or not np.isfinite(values).all():
        raise ValueError(f"Invalid fingerprint feature vector for {row.get('video_path', '<unknown path>')}")
    return values


def load_official_training_data(fingerprints_csv: str | Path, split_mapping_csv: str | Path):
    """Load only official training feature vectors; retain test paths as metadata only."""
    split_rows = read_rows(split_mapping_csv)
    split_by_path = {canonical_path(r["video_path"]): r for r in split_rows}
    if len(split_by_path) != len(split_rows):
        raise ValueError("Duplicate video paths in official split mapping")
    training_rows, test_paths, assignments = [], set(), []
    seen = set()
    with Path(fingerprints_csv).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.DictReader(stream):
            key = canonical_path(row["video_path"])
            if key in seen:
                raise ValueError(f"Duplicate fingerprint video path: {row['video_path']}")
            seen.add(key)
            split_row = split_by_path.get(key)
            if split_row is None:
                raise ValueError(f"Fingerprint path absent from official split mapping: {row['video_path']}")
            if row["gesture_label"] != split_row["gesture_label"]:
                raise ValueError(f"Gesture label differs from official split map: {row['video_path']}")
            part = split_row["official_split"]
            assignments.append({"video_path": row["video_path"], "gesture_label": row["gesture_label"],
                                "official_split": part, "internal_partition": ""})
            if part == "train":
                training_rows.append({"video_path": row["video_path"], "gesture_label": row["gesture_label"],
                                      "features": feature_vector(row)})
            elif part == "test":
                # Do not parse or use test feature values at this stage.
                test_paths.add(key)
            else:
                raise ValueError(f"Unsupported/unmatched official split for {row['video_path']}: {part}")
    if seen != set(split_by_path):
        missing = sorted(set(split_by_path) - seen)
        raise ValueError(f"Official split map and fingerprints have different video paths; missing fingerprints: {missing[:5]}")
    if test_paths & {canonical_path(r["video_path"]) for r in training_rows}:
        raise AssertionError("Official test path leaked into training feature data")
    return training_rows, test_paths, assignments


def split_official_training(training_rows: list[dict], official_test_paths: set[str],
                            validation_fraction: float = 0.2, seed: int = 42):
    """Make a stratified internal split from official training rows only."""
    if not training_rows:
        raise ValueError("No official training rows available")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    official_test_paths = {canonical_path(p) for p in official_test_paths}
    paths = [canonical_path(r["video_path"]) for r in training_rows]
    if len(paths) != len(set(paths)):
        raise ValueError("Duplicate paths among official training rows")
    if set(paths) & official_test_paths:
        raise ValueError("Official test video path entered internal training/validation split")
    labels = [r["gesture_label"] for r in training_rows]
    fit_rows, validation_rows = train_test_split(
        training_rows, test_size=validation_fraction, random_state=seed, stratify=labels)
    assert_no_official_test_leakage(fit_rows, validation_rows, official_test_paths)
    return list(fit_rows), list(validation_rows)


def assert_no_official_test_leakage(training_rows, validation_rows, official_test_paths: set[str]) -> None:
    official_test_paths = {canonical_path(p) for p in official_test_paths}
    fit_paths = {canonical_path(r["video_path"]) for r in training_rows}
    val_paths = {canonical_path(r["video_path"]) for r in validation_rows}
    if fit_paths & val_paths:
        raise AssertionError("Internal training and validation video paths overlap")
    if (fit_paths | val_paths) & official_test_paths:
        raise AssertionError("Official test video path entered training or validation")


def make_prototype(features, method: str) -> np.ndarray:
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or not len(x) or not np.isfinite(x).all():
        raise ValueError("Prototype input must be a non-empty finite matrix with 70 features")
    if method == "mean":
        return x.mean(axis=0)
    if method == "median":
        return np.median(x, axis=0)
    raise ValueError("Prototype method must be 'mean' or 'median'")


class FeatureScaler:
    """Training-only per-feature scaling parameters for fingerprint distances."""
    def __init__(self, kind: str):
        if kind not in {"none", "standard", "robust"}:
            raise ValueError("Scaler kind must be none, standard, or robust")
        self.kind = kind
        self.center_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, features):
        x = np.asarray(features, dtype=float)
        if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or not len(x) or not np.isfinite(x).all():
            raise ValueError("Scaler input must be a non-empty finite 70-feature matrix")
        if self.kind == "none":
            self.center_ = np.zeros(FEATURE_COUNT)
            self.scale_ = np.ones(FEATURE_COUNT)
        elif self.kind == "standard":
            self.center_ = x.mean(axis=0)
            self.scale_ = x.std(axis=0)
        else:
            self.center_ = np.median(x, axis=0)
            q75, q25 = np.percentile(x, [75, 25], axis=0)
            self.scale_ = q75 - q25
        self.scale_[~np.isfinite(self.scale_) | (self.scale_ < 1e-12)] = 1.0
        return self


def fingerprint_distance(a, b, method: str, scale=None) -> float:
    """Compute one of the four documented distances; scales must be fit on training data.

    euclidean: ||x-p||_2
    training_scaled: sqrt(sum_i ((x_i-p_i)/s_i)^2), s_i=train SD
    robust_scaled: sqrt(sum_i ((x_i-p_i)/IQR_i)^2), IQR from training rows
    group_balanced: sqrt((M_coord + M_angle + M_tip)/3), each M is the
      mean squared standardized difference inside its 60/5/5-feature group.
    """
    x, p = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if x.shape != (FEATURE_COUNT,) or p.shape != (FEATURE_COUNT,) or not (np.isfinite(x).all() and np.isfinite(p).all()):
        raise ValueError("Distance inputs must be finite 70-feature vectors")
    delta = x - p
    if method == "euclidean":
        return float(np.linalg.norm(delta))
    if method not in {"training_scaled", "robust_scaled", "group_balanced"}:
        raise ValueError(f"Unknown distance method: {method}")
    if scale is None:
        raise ValueError(f"{method} requires a scale fitted on training data")
    scale = np.asarray(scale, dtype=float)
    if scale.shape != (FEATURE_COUNT,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("Distance scale must contain 70 finite positive values")
    standardized_squared = (delta / scale) ** 2
    if method in {"training_scaled", "robust_scaled"}:
        return float(math.sqrt(float(standardized_squared.sum())))
    group_means = [float(standardized_squared[start:end].mean()) for start, end in GROUPS]
    return float(math.sqrt(sum(group_means) / len(group_means)))


def fit_prototypes(training_rows: list[dict], prototype_method: str):
    classes = sorted({r["gesture_label"] for r in training_rows})
    prototypes, counts = {}, {}
    for label in classes:
        members = [r["features"] for r in training_rows if r["gesture_label"] == label]
        prototypes[label] = make_prototype(members, prototype_method)
        counts[label] = len(members)
    return prototypes, counts


def nearest_prototype(query, prototypes: dict[str, np.ndarray], method: str, scale: np.ndarray):
    scored = [(label, fingerprint_distance(query, proto, method, scale))
              for label, proto in prototypes.items()]
    return min(scored, key=lambda x: (x[1], x[0]))


def apply_rejection(label: str, distance: float, threshold: float) -> tuple[str, bool]:
    """Return UNKNOWN when the nearest-prototype distance exceeds threshold."""
    accepted = float(distance) <= float(threshold)
    return (str(label) if accepted else "UNKNOWN", accepted)


def _metrics(y_true, y_pred, labels):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    return {"accuracy": float(accuracy_score(y_true, y_pred)),
            "precision_macro": float(precision), "recall_macro": float(recall),
            "macro_f1": float(f1),
            "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist()}


def run_validation_selection(fingerprints_csv: str | Path, split_mapping_csv: str | Path,
                             output_dir: str | Path, seed: int = 42,
                             validation_fraction: float = 0.2,
                             target_correct_validation_coverage: float = 0.95) -> dict:
    """Compare eight configurations on one fixed validation set; select and save artifacts.

    Selection maximizes validation macro-F1 without rejection, then validation accuracy.
    The threshold is calibrated only after selection from correctly classified validation
    distances using the requested empirical quantile with the higher order statistic.
    """
    if not 0 < target_correct_validation_coverage <= 1:
        raise ValueError("target_correct_validation_coverage must be in (0,1]")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_rows, official_test_paths, assignments = load_official_training_data(fingerprints_csv, split_mapping_csv)
    fit_rows, validation_rows = split_official_training(training_rows, official_test_paths,
                                                        validation_fraction, seed)
    assert_no_official_test_leakage(fit_rows, validation_rows, official_test_paths)
    fit_paths = {canonical_path(r["video_path"]) for r in fit_rows}
    val_paths = {canonical_path(r["video_path"]) for r in validation_rows}
    test_paths = {canonical_path(r["video_path"]) for r in assignments if r["official_split"] == "test"}
    if fit_paths & test_paths or val_paths & test_paths:
        raise AssertionError("Official test path leaked into fitting or validation")

    for item in assignments:
        key = canonical_path(item["video_path"])
        if key in fit_paths:
            item["internal_partition"] = "prototype_fit"
        elif key in val_paths:
            item["internal_partition"] = "validation"
        else:
            item["internal_partition"] = "official_test_untouched"
    with (output_dir / "split_assignments.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["video_path","gesture_label","official_split","internal_partition"])
        writer.writeheader(); writer.writerows(assignments)

    x_fit = np.vstack([r["features"] for r in fit_rows])
    y_fit = np.asarray([r["gesture_label"] for r in fit_rows], dtype=str)
    x_val = np.vstack([r["features"] for r in validation_rows])
    y_val = np.asarray([r["gesture_label"] for r in validation_rows], dtype=str)
    class_labels = sorted(set(y_fit) | set(y_val))
    scaler_kinds = {"euclidean":"none", "training_scaled":"standard",
                    "robust_scaled":"robust", "group_balanced":"standard"}
    configurations=[]
    for prototype_method in PROTOTYPE_METHODS:
        for distance_method in DISTANCE_METHODS:
            scaler = FeatureScaler(scaler_kinds[distance_method]).fit(x_fit)
            prototypes, counts = fit_prototypes(fit_rows, prototype_method)
            preds, distances = [], []
            for query in x_val:
                label, distance = nearest_prototype(query, prototypes, distance_method, scaler.scale_)
                preds.append(label); distances.append(distance)
            metrics = _metrics(y_val, np.asarray(preds), class_labels)
            configurations.append({"prototype_method":prototype_method,"distance_method":distance_method,
                                   "scaler":scaler,"prototypes":prototypes,"counts":counts,
                                   "predictions":np.asarray(preds),"distances":np.asarray(distances),
                                   "metrics":metrics})
    # Deterministic selection criterion uses validation results only.
    selected = max(configurations, key=lambda c: (c["metrics"]["macro_f1"], c["metrics"]["accuracy"]))
    correct_distances = selected["distances"][selected["predictions"] == y_val]
    if not len(correct_distances):
        raise ValueError("Selected validation configuration has no correctly classified rows; cannot calibrate rejection")
    threshold = float(np.quantile(correct_distances, target_correct_validation_coverage, method="higher"))
    decisions = [apply_rejection(label, distance, threshold)
                 for label, distance in zip(selected["predictions"], selected["distances"])]
    accepted = np.asarray([item[1] for item in decisions], dtype=bool)
    rejected_predictions = np.asarray([item[0] for item in decisions], dtype=str)
    # Macro metrics average the known gesture classes; UNKNOWN is shown as its own
    # confusion-matrix prediction column and reflected in coverage/rejection rate.
    rejected_metrics = _metrics(y_val, rejected_predictions, class_labels)
    coverage = float(np.mean(accepted))

    comparison_rows=[]
    for c in configurations:
        m=c["metrics"]
        comparison_rows.append({"prototype_method":c["prototype_method"],"distance_method":c["distance_method"],
          "validation_accuracy":m["accuracy"],"validation_precision_macro":m["precision_macro"],
          "validation_recall_macro":m["recall_macro"],"validation_macro_f1":m["macro_f1"],
          "validation_confusion_labels":json.dumps(class_labels,ensure_ascii=False),
          "validation_confusion_matrix":json.dumps(m["confusion_matrix"]),
          "selected_by_validation":c is selected})
    comparison_fields=list(comparison_rows[0])
    with (output_dir/"distance_method_comparison.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=comparison_fields);writer.writeheader();writer.writerows(comparison_rows)

    validation_result_rows=[]
    for row,truth,pred,distance,keep in zip(validation_rows,y_val,selected["predictions"],selected["distances"],accepted):
        validation_result_rows.append({"video_path":row["video_path"],"gesture_label":truth,
          "predicted_before_rejection":pred,"nearest_distance":float(distance),
          "accepted_by_threshold":bool(keep),"predicted_label":"" if not keep else pred,
          "result_label":"UNKNOWN" if not keep else pred})
    with (output_dir/"validation_results.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(validation_result_rows[0]));writer.writeheader();writer.writerows(validation_result_rows)

    cm=confusion_matrix(y_val,rejected_predictions,labels=class_labels+["UNKNOWN"])
    with (output_dir/"validation_confusion_matrix.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["actual\\predicted"]+class_labels+["UNKNOWN"])
        for label,row in zip(class_labels+["UNKNOWN"],cm):writer.writerow([label]+row.tolist())

    pair_counts=Counter((str(t),str(p)) for t,p in zip(y_val,selected["predictions"]) if t!=p)
    pair_rows=[{"actual_label":a,"predicted_label":p,"error_count":n} for (a,p),n in pair_counts.most_common()]
    with (output_dir/"validation_pair_errors.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["actual_label","predicted_label","error_count"]);writer.writeheader();writer.writerows(pair_rows)

    classes=sorted(selected["prototypes"])
    prototype_matrix=np.vstack([selected["prototypes"][label] for label in classes])
    npz_path=output_dir/"selected_model_artifacts.npz"
    np.savez_compressed(npz_path, class_labels=np.asarray(classes,dtype=str), prototypes=prototype_matrix,
      scaler_center=selected["scaler"].center_, scaler_scale=selected["scaler"].scale_,
      scaler_kind=np.asarray(selected["scaler"].kind), distance_method=np.asarray(selected["distance_method"]),
      prototype_method=np.asarray(selected["prototype_method"]), rejection_threshold=np.asarray(threshold),
      group_weights=np.asarray([1/3,1/3,1/3]), group_lengths=np.asarray([60,5,5]))
    reference_rows=[]
    for label in classes:
        row={"gesture_label":label,"prototype_method":selected["prototype_method"],
             "training_sample_count":selected["counts"][label]}
        row.update({f"feature_{i+1}":float(v) for i,v in enumerate(selected["prototypes"][label])})
        reference_rows.append(row)
    with (output_dir/"reference_fingerprints.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(reference_rows[0]));writer.writeheader();writer.writerows(reference_rows)
    with (output_dir/"selected_model_scaler.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["feature_index","feature_name","scaling_method","center_fit_on_training","scale_fit_on_training"])
        for i,(name,center,scale) in enumerate(zip(FEATURE_NAMES,selected["scaler"].center_,selected["scaler"].scale_),1):
            writer.writerow([i,name,selected["scaler"].kind,float(center),float(scale)])

    train_counts=Counter(r["gesture_label"] for r in fit_rows)
    val_counts=Counter(r["gesture_label"] for r in validation_rows)
    metadata={"algorithm":"nearest landmark-fingerprint class prototype",
      "selected_prototype_method":selected["prototype_method"],"selected_distance_method":selected["distance_method"],
      "selection_criterion":"maximum validation macro-F1 without rejection; tie-break by validation accuracy",
      "rejection_threshold":threshold,"threshold_calibration":"higher empirical quantile of correctly classified validation nearest-prototype distances",
      "target_correct_validation_coverage":target_correct_validation_coverage,
      "correct_validation_distance_count":int(len(correct_distances)),"observed_validation_coverage_after_threshold":coverage,
      "unknown_rejection_rate_validation":float(1-coverage),"distance_is_probability":False,
      "validation_accuracy_before_rejection":selected["metrics"]["accuracy"],
      "validation_precision_macro_before_rejection":selected["metrics"]["precision_macro"],
      "validation_recall_macro_before_rejection":selected["metrics"]["recall_macro"],
      "validation_macro_f1_before_rejection":selected["metrics"]["macro_f1"],
      "validation_accuracy_with_rejection":rejected_metrics["accuracy"],
      "validation_precision_macro_with_rejection":rejected_metrics["precision_macro"],
      "validation_recall_macro_with_rejection":rejected_metrics["recall_macro"],
      "validation_macro_f1_with_rejection":rejected_metrics["macro_f1"],
      "validation_confusion_labels":class_labels+["UNKNOWN"],"validation_confusion_matrix":cm.tolist(),
      "seed":seed,"validation_fraction_of_official_train":validation_fraction,
      "official_training_video_count":len(training_rows),"prototype_fit_video_count":len(fit_rows),
      "validation_video_count":len(validation_rows),"official_test_video_count":len(official_test_paths),
      "prototype_fit_class_counts":dict(train_counts),"validation_class_counts":dict(val_counts),
      "prototype_fit_video_paths":[r["video_path"] for r in fit_rows],
      "validation_video_paths":[r["video_path"] for r in validation_rows],
      "official_test_used_for_fitting_or_threshold":False,
      "official_test_evaluation_executed":False,
      "artifact_npz":npz_path.name,"reference_csv":"reference_fingerprints.csv",
      "scaler_csv":"selected_model_scaler.csv",
      "limitations":["Official test set has no explicit unknown-sign examples.",
        "Only 4-5 official test videos per current class.",
        "No signer metadata was supplied; signer-independent evaluation is unavailable.",
        "Validation is small and is used for both method selection and threshold calibration."]}
    (output_dir/"selected_model_metadata.json").write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding="utf-8")

    report_lines=["LANDMARK FINGERPRINT MATCHING — VALIDATION ONLY",
      "No official test feature values were parsed by this preparation pipeline. No final official test evaluation was executed.",
      f"Official split: {len(training_rows)} train; {len(official_test_paths)} test. Internal fit={len(fit_rows)}; validation={len(validation_rows)} (seed={seed}).",
      f"Selected prototype: {selected['prototype_method']}; selected distance: {selected['distance_method']}.",
      "Selection: highest validation macro-F1 before rejection; ties break on validation accuracy.",
      f"Validation before rejection: accuracy={selected['metrics']['accuracy']:.4f}, precision_macro={selected['metrics']['precision_macro']:.4f}, recall_macro={selected['metrics']['recall_macro']:.4f}, macro-F1={selected['metrics']['macro_f1']:.4f}.",
      f"UNKNOWN threshold={threshold:.8f}, calibrated from {len(correct_distances)} correctly classified validation distances at target {target_correct_validation_coverage:.0%} coverage; observed validation coverage={coverage:.1%}.",
      f"Validation with rejection: accuracy={rejected_metrics['accuracy']:.4f}, precision_macro={rejected_metrics['precision_macro']:.4f}, recall_macro={rejected_metrics['recall_macro']:.4f}, macro-F1={rejected_metrics['macro_f1']:.4f}; reject rate={1-coverage:.1%}.",
      "Distance values are not calibrated confidence probabilities.","",
      "DISTANCE FORMULAS (all scale parameters fit on internal official-training rows only)",
      "Raw Euclidean: D(x,p)=sqrt(sum_i (x_i-p_i)^2).",
      "Training-scaled Euclidean: D=sqrt(sum_i ((x_i-p_i)/s_i)^2), where s_i is the training standard deviation (near-zero scales become 1).",
      "Robust-scaled: same norm using training IQR for s_i (near-zero IQR becomes 1).",
      "Group-balanced: D=sqrt((M_coord+M_angle+M_tip)/3); each M is the mean squared standard-scaled difference in the 60 coordinate, 5 angle, or 5 fingertip-distance features.",
      "Prototypes are per-feature means or medians over internal training videos only; each video contributes one 70-feature row.","",
      "VALIDATION METHOD COMPARISON"]
    for c in configurations:
        m=c['metrics'];mark=' [SELECTED]' if c is selected else ''
        report_lines.append(f"{c['prototype_method']} / {c['distance_method']}: acc={m['accuracy']:.4f}, P={m['precision_macro']:.4f}, R={m['recall_macro']:.4f}, macro-F1={m['macro_f1']:.4f}{mark}")
    report_lines += ["","VALIDATION MISCLASSIFICATIONS (candidate similar-class confusions; confirm similarity with an FSL reviewer)"]
    if pair_rows:
        report_lines += [f"{r['actual_label']} -> {r['predicted_label']}: {r['error_count']}" for r in pair_rows]
    else:
        report_lines.append("No validation misclassifications before rejection.")
    report_lines += ["","UNKNOWN REJECTION LIMITATION",
      "The official test has only known gesture classes and no explicit unknown-sign examples. The validation threshold therefore does not fully validate rejection of genuinely unknown signs. Collect a separate unknown-sign validation/test set before making open-set rejection claims.",
      "","ARTIFACTS",
      "reference_fingerprints.csv: one selected prototype per class, with fit sample counts.",
      "selected_model_artifacts.npz: selected prototype matrix, training-fitted scaler arrays, method names, group weights, and validation-calibrated threshold.",
      "selected_model_scaler.csv: inspectable per-feature scaler center/scale.",
      "selected_model_metadata.json: method, threshold, split counts, validation metrics, and training/validation path provenance.",
      "validation_results.csv, validation_confusion_matrix.csv, distance_method_comparison.csv, validation_pair_errors.csv, split_assignments.csv.",
      "","FINAL TEST STATUS: NOT RUN. Run src/evaluate_official_test.py only after separate approval."]
    (output_dir/"matching_report.txt").write_text("\n".join(report_lines)+"\n",encoding="utf-8")
    return metadata

