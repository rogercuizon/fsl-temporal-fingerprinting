"""Final official-test evaluator. Intentionally requires an explicit confirmation flag."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .matching import canonical_path, fingerprint_distance, read_rows


def evaluate_official_test(project_root: str | Path) -> dict:
    """Evaluate once on official test rows; call only after the user authorizes final test."""
    root=Path(project_root).resolve()
    matching=root/"output"/"matching"
    metadata=json.loads((matching/"selected_model_metadata.json").read_text(encoding="utf-8-sig"))
    arrays=np.load(matching/metadata["artifact_npz"],allow_pickle=False)
    labels=[str(x) for x in arrays["class_labels"].tolist()]
    prototypes={label:arrays["prototypes"][i] for i,label in enumerate(labels)}
    scale=arrays["scaler_scale"]
    method=str(arrays["distance_method"].item())
    threshold=float(arrays["rejection_threshold"].item())
    if method!="robust_scaled" or str(arrays["prototype_method"].item())!="mean":
        raise ValueError("Saved artifacts do not match the approved mean / robust_scaled model")
    if not np.isclose(threshold,9.222295068261808,rtol=0,atol=1e-10):
        raise ValueError("Saved UNKNOWN threshold differs from the frozen validation threshold")
    train_val={canonical_path(p) for p in metadata["prototype_fit_video_paths"]+metadata["validation_video_paths"]}
    split_rows=read_rows(root/"output"/"active_segment_review"/"official_split_mapping.csv")
    test_map={canonical_path(r["video_path"]):r for r in split_rows if r["official_split"]=="test"}
    if len(test_map)!=metadata["official_test_video_count"]:
        raise ValueError("Official test split count differs from the frozen matching metadata")
    if train_val & set(test_map):
        raise ValueError("An official test path appears in prototype-fit or validation paths")
    results=[]
    with (root/"output"/"fingerprints.csv").open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            key=canonical_path(row["video_path"])
            if key not in test_map:
                continue
            if row["gesture_label"]!=test_map[key]["gesture_label"]:
                raise ValueError(f"Test label differs from official split map for {row['video_path']}")
            x=np.asarray([float(row[f"feature_{i}"]) for i in range(1,71)],dtype=float)
            start=time.perf_counter()
            scores=[(label,fingerprint_distance(x,proto,method,scale)) for label,proto in prototypes.items()]
            latency_ms=(time.perf_counter()-start)*1000.0
            predicted,distance=min(scores,key=lambda pair:(pair[1],pair[0]))
            accepted=distance<=threshold
            results.append({"video_path":row["video_path"],"true_label":row["gesture_label"],
                            "predicted_before_rejection":predicted,"nearest_distance":distance,
                            "accepted":accepted,"predicted_label":predicted if accepted else "UNKNOWN",
                            "matching_latency_ms":latency_ms})
    if len(results)!=len(test_map):
        raise ValueError(f"Expected {len(test_map)} official test fingerprints; found {len(results)}")
    y_true=[r["true_label"] for r in results];y_pred=[r["predicted_label"] for r in results]
    evaluation_labels=sorted(set(labels))
    precision,recall,f1,_=precision_recall_fscore_support(y_true,y_pred,labels=evaluation_labels,average="macro",zero_division=0)
    matrix_labels=evaluation_labels+["UNKNOWN"]
    cm=confusion_matrix(y_true,y_pred,labels=matrix_labels)
    accepted_results=[r for r in results if r["accepted"]]
    report={"official_test_videos":len(results),"accepted_predictions":len(accepted_results),
            "unknown_predictions":len(results)-len(accepted_results),
            "accuracy_including_rejections":float(accuracy_score(y_true,y_pred)),
            "precision_macro":float(precision),"recall_macro":float(recall),"macro_f1":float(f1),
            "accuracy_among_accepted":float(np.mean([r["predicted_before_rejection"]==r["true_label"] for r in accepted_results])) if accepted_results else None,
            "coverage":float(np.mean([r['accepted'] for r in results])),
            "rejection_rate":float(1-np.mean([r['accepted'] for r in results])),
            "mean_matching_latency_ms":float(np.mean([r["matching_latency_ms"] for r in results])),
            "median_matching_latency_ms":float(np.median([r["matching_latency_ms"] for r in results])),
            "total_matching_latency_ms":float(np.sum([r["matching_latency_ms"] for r in results])),
            "selected_distance_method":method,"prototype_method":str(arrays['prototype_method'].item()),
            "threshold_frozen_before_test":threshold,"confusion_labels":matrix_labels,
            "confusion_matrix":cm.tolist(),"per_class":classification_report(y_true,y_pred,labels=evaluation_labels,output_dict=True,zero_division=0),
            "unknown_sign_limitation":"The official test set has no explicit unknown-sign examples; unknown-sign rejection is not validated."}
    result_dir=root/"output"/"final_evaluation"
    result_dir.mkdir(parents=True,exist_ok=True)
    with (result_dir/"final_predictions.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(results[0]));writer.writeheader();writer.writerows(results)
    (result_dir/"final_metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding="utf-8")
    with (result_dir/"final_confusion_matrix.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["actual\\predicted"]+matrix_labels)
        for label,row in zip(matrix_labels,cm):writer.writerow([label]+row.tolist())
    with (result_dir/"final_per_class_metrics.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        fields=["gesture_label","precision","recall","f1_score","support"]
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for label in evaluation_labels:
            stats=report["per_class"][label]
            writer.writerow({"gesture_label":label,"precision":stats["precision"],"recall":stats["recall"],
                             "f1_score":stats["f1-score"],"support":int(stats["support"])})
    pair_counts={}
    for r in results:
        if r["predicted_label"]!="UNKNOWN" and r["predicted_label"]!=r["true_label"]:
            key=(r["true_label"],r["predicted_label"])
            pair_counts[key]=pair_counts.get(key,0)+1
    with (result_dir/"final_pairwise_errors.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["true_label","predicted_label","count"]);writer.writeheader()
        for (true_label,predicted_label),count in sorted(pair_counts.items(),key=lambda item:(-item[1],item[0])):
            writer.writerow({"true_label":true_label,"predicted_label":predicted_label,"count":count})
    lines=["FINAL OFFICIAL TEST EVALUATION",f"Test samples: {len(results)}",
           f"Accepted predictions: {len(accepted_results)}",f"UNKNOWN rejections: {len(results)-len(accepted_results)}",
           f"Accuracy (UNKNOWN counted incorrect): {report['accuracy_including_rejections']:.4f}",
           f"Macro precision: {precision:.4f}",f"Macro recall: {recall:.4f}",f"Macro-F1: {f1:.4f}",
           f"Coverage: {report['coverage']:.4f}",f"Rejection rate: {report['rejection_rate']:.4f}",
           f"Accuracy among accepted: {report['accuracy_among_accepted']:.4f}" if report['accuracy_among_accepted'] is not None else "Accuracy among accepted: N/A",
           f"Mean matching latency: {report['mean_matching_latency_ms']:.4f} ms/sample",
           f"Median matching latency: {report['median_matching_latency_ms']:.4f} ms/sample",
           f"Total matching latency: {report['total_matching_latency_ms']:.4f} ms",
           f"Model: {report['prototype_method']} prototype / {method}; fixed threshold={threshold:.10f}",
           "UNKNOWN counts are rejection decisions, distinct from accepted but incorrect class predictions.",
           report["unknown_sign_limitation"],"Metrics describe this official test set only and do not establish universal accuracy."]
    (result_dir/"final_evaluation_report.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-final-evaluation",action="store_true",
                        help="Explicitly authorize running the frozen model on the official test set.")
    args=parser.parse_args(argv)
    if not args.confirm_final_evaluation:
        parser.error("Final evaluation is gated. Obtain user approval, then pass --confirm-final-evaluation.")
    root=Path(__file__).resolve().parents[1]
    report=evaluate_official_test(root)
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__":
    main()
