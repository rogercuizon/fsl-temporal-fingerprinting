"""Read-only validation and normalized working manifests for the 105-class mobile dataset."""
from __future__ import annotations

import csv
import zipfile
from collections import Counter
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def prepare_mobile_manifests(source_dir: str | Path, output_dir: str | Path) -> dict:
    """Create normalized copies while preserving official train/test membership.

    Class IDs are authoritative for mapping text through labels.csv. The function
    validates all manifest clip paths against the archive without extracting media.
    """
    source = Path(source_dir).resolve()
    destination = Path(output_dir)
    labels_rows = _read_csv(source / "labels.csv")
    labels = {str(row["id"]).strip(): row for row in labels_rows}
    if len(labels) != len(labels_rows) or len(labels) != 105:
        raise ValueError("Expected 105 unique label IDs in labels.csv")

    with zipfile.ZipFile(source / "clips.zip") as archive:
        members = {name.replace("\\", "/").casefold()
                   for name in archive.namelist() if not name.endswith("/")}

    normalized = {}
    manifest_mismatches = Counter()
    all_paths: dict[str, str] = {}
    for split in ("train", "test"):
        rows = _read_csv(source / f"{split}.csv")
        if not rows:
            raise ValueError(f"{split}.csv is empty")
        new_rows = []
        split_paths = set()
        for row in rows:
            video_path = row["vid_path"].strip().replace("/", "\\")
            path_key = video_path.casefold()
            if path_key in split_paths:
                raise ValueError(f"Duplicate path in {split}.csv: {video_path}")
            split_paths.add(path_key)
            if path_key in all_paths:
                raise ValueError(f"Train/test path overlap: {video_path}")
            all_paths[path_key] = split
            archive_key = video_path.replace("\\", "/").casefold()
            if archive_key not in members:
                raise ValueError(f"Manifest video missing from clips.zip: {video_path}")

            class_id = str(row["id_label"]).strip()
            label_row = labels.get(class_id)
            if label_row is None:
                raise ValueError(f"Unknown class ID {class_id} in {split}.csv")
            folder_id = Path(video_path).parts[-2]
            if folder_id != class_id:
                raise ValueError(f"Folder ID does not match label ID for {video_path}")
            original_label = row["label"]
            canonical_label = label_row["label"].strip()
            if original_label != canonical_label:
                manifest_mismatches[class_id] += 1

            new_rows.append({
                "video_path": video_path,
                "class_id": class_id,
                "gesture_label": canonical_label,
                "cebuano_translation": label_row["cebuano_translation"].strip(),
                "category": label_row["category"].strip(),
                "official_split": split,
                "source_manifest_label": original_label,
            })
        normalized[split] = new_rows

    if len(all_paths) != len(members):
        absent_from_manifests = members - set(all_paths)
        if absent_from_manifests:
            raise ValueError(f"{len(absent_from_manifests)} archive clips are absent from official manifests")

    destination.mkdir(parents=True, exist_ok=True)
    fields = ["video_path", "class_id", "gesture_label", "cebuano_translation",
              "category", "official_split", "source_manifest_label"]
    for split, rows in normalized.items():
        with (destination / f"{split}_manifest.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    report = {
        "classes": len(labels),
        "archive_video_count": len(members),
        "train_video_count": len(normalized["train"]),
        "test_video_count": len(normalized["test"]),
        "split_overlap": 0,
        "manifest_label_mismatch_rows_normalized": sum(manifest_mismatches.values()),
        "normalized_mismatch_counts_by_id": dict(sorted(manifest_mismatches.items(), key=lambda item: int(item[0]))),
        "original_files_modified": False,
        "test_media_extracted_or_features_read": False,
    }
    with (destination / "dataset_audit.txt").open("w", encoding="utf-8") as stream:
        stream.write("FSL-105 MOBILE DATASET MANIFEST AUDIT\n")
        for key, value in report.items():
            stream.write(f"{key}: {value}\n")
        stream.write("Labels in working manifests are taken from labels.csv by class ID.\n")
        stream.write("Official train/test membership is retained from the supplied manifests.\n")
    return report
