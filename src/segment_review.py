"""Frame-level hand-detection timeline and non-destructive preview generation."""
from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
import math
import tempfile

import cv2
import numpy as np

from .landmark_extractor import LandmarkExtractor
from .video_processor import scan_videos


def _write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _detection_runs(timeline: list[dict]) -> list[list[dict]]:
    """Return runs of consecutive detected *sampled* frames, not adjacent source frames."""
    runs: list[list[dict]] = []
    current: list[dict] = []
    for row in timeline:
        if row["hand_detected"]:
            if current and row["sample_index"] != current[-1]["sample_index"] + 1:
                runs.append(current)
                current = []
            current.append(row)
        elif current:
            runs.append(current)
            current = []
    if current:
        runs.append(current)
    return runs


def _candidate_description(runs: list[list[dict]], fps: float) -> str:
    if not runs:
        return "No detected frames"
    pieces = []
    for number, run in enumerate(runs, 1):
        kind = "consecutive-sampled-run" if len(run) > 1 else "isolated-detection"
        start, end = run[0], run[-1]
        pieces.append(
            f"run_{number} {kind}: source frames {start['source_frame_number']}-"
            f"{end['source_frame_number']} ({start['timestamp_seconds']:.2f}-"
            f"{end['timestamp_seconds']:.2f}s; {len(run)} detection(s))"
        )
    return "; ".join(pieces)


def _representative_rows(summary: list[dict]) -> list[dict]:
    usable = [r for r in summary if r["total_sampled_frames"] > 0]
    if not usable:
        return []
    ordered = sorted(usable, key=lambda r: (r["detection_rate"], r["video_path"]))
    candidates = [ordered[0], ordered[len(ordered) // 2], ordered[-1]]
    low_label = ordered[0]["gesture_label"]
    same_class = sorted((r for r in usable if r["gesture_label"] == low_label),
                        key=lambda r: (r["detection_rate"], r["video_path"]))
    candidates.extend([same_class[0], same_class[-1]])
    unique = []
    seen = set()
    for row in candidates:
        if row["video_path"] not in seen:
            seen.add(row["video_path"])
            unique.append(row)
    return unique


def _make_preview(root: Path, outdir: Path, item: dict, timeline: list[dict], sample_every: int,
                  model_path: Path) -> str:
    """Create a contact sheet with original and annotated samples from one clip."""
    import mediapipe as mp

    source = root / Path(item["video_path"])
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        cap.release()
        return ""
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    by_index = {r["sample_index"]: r for r in timeline}
    detected = [r for r in timeline if r["hand_detected"]]
    runs = _detection_runs(timeline)
    longest = max(runs, key=len, default=[])
    n = len(timeline)
    selected = [1, max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4), max(1, n)]
    if longest:
        selected += [longest[0]["sample_index"], longest[len(longest)//2]["sample_index"], longest[-1]["sample_index"]]
        before = [r for r in timeline if not r["hand_detected"] and r["sample_index"] < longest[0]["sample_index"]]
        if before:
            selected.append(before[-1]["sample_index"])
    if detected and not longest:
        selected.append(detected[0]["sample_index"])
    selected = sorted({i for i in selected if 1 <= i <= n})
    # Keep the sheet readable while preserving neutral and active-segment anchors.
    if len(selected) > 8:
        keep = {selected[0], selected[-1]}
        if longest:
            keep.update((longest[0]["sample_index"], longest[len(longest)//2]["sample_index"], longest[-1]["sample_index"]))
        remaining = [i for i in selected if i not in keep]
        keep.update(remaining[:max(0, 8-len(keep))])
        selected = sorted(keep)

    frames: dict[int, tuple[np.ndarray, np.ndarray, bool, int]] = {}
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    target = set(selected)
    frame_no = 0
    sample_index = 0
    with mp.tasks.vision.HandLandmarker.create_from_options(options) as detector:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            current = frame_no
            frame_no += 1
            if current % sample_every:
                continue
            sample_index += 1
            expected = by_index.get(sample_index, {})
            if sample_index not in target:
                # Keep the VIDEO-mode tracking state aligned with the original pass.
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(current*1000/fps))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), round(current*1000/fps))
            annotated = frame.copy()
            hands = result.hand_landmarks or []
            if hands:
                hand = hands[0]
                height, width = annotated.shape[:2]
                points = [(round(p.x*width), round(p.y*height)) for p in hand]
                for edge in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                    cv2.line(annotated, points[edge.start], points[edge.end], (0, 255, 0), 2)
                for x, y in points:
                    cv2.circle(annotated, (x, y), 3, (0, 0, 255), -1)
            frames[sample_index] = (frame.copy(), annotated, bool(hands), current)
    cap.release()

    if not frames:
        return ""
    tile_w, tile_h, cap_h, gap, top = 320, 180, 23, 9, 48
    cols = min(4, len(frames))
    rows = math.ceil(len(frames) / cols)
    sheet = np.full((top + rows*(2*tile_h+2*cap_h+3*gap)+gap,
                     gap + cols*(tile_w+gap), 3), 245, dtype=np.uint8)
    rate = item["detection_rate"]
    title = f"{item['gesture_label']} | {item['video_path']} | rate {rate:.1%} | sampled-frame candidates only"
    cv2.putText(sheet, title[:100], (gap, 29), cv2.FONT_HERSHEY_SIMPLEX, .53, (15,15,15), 1, cv2.LINE_AA)
    for idx, (sample_i, (original, annotated, found, source_frame)) in enumerate(sorted(frames.items())):
        col, row = idx % cols, idx // cols
        x = gap + col*(tile_w+gap)
        y0 = top + row*(2*tile_h+2*cap_h+3*gap)
        for half, image in enumerate((original, annotated)):
            y = y0 + half*(tile_h+cap_h+gap)
            resized = cv2.resize(image, (tile_w,tile_h), interpolation=cv2.INTER_AREA)
            sheet[y:y+tile_h, x:x+tile_w] = resized
            label = f"{'ORIGINAL' if half == 0 else ('LANDMARKS' if found else 'NO HAND')} | sample {sample_i} | frame {source_frame}"
            cv2.putText(sheet, label, (x,y+tile_h+16), cv2.FONT_HERSHEY_SIMPLEX, .38, (0,0,0), 1, cv2.LINE_AA)
    filename = f"{item['video_path'].replace('\\','_').replace('/','_').replace('.','_')}.jpg"
    path = outdir / filename
    cv2.imwrite(str(path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return filename


def run_active_segment_review(root: str | Path, output_dir: str | Path, sample_every: int = 5) -> dict:
    """Rerun frame detection, export full sampled-frame timeline, segments, summaries, and previews."""
    root, output_dir = Path(root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = output_dir / "previews"
    preview_dir.mkdir(parents=True, exist_ok=True)
    extractor = LandmarkExtractor(max_hands=1, sample_every=sample_every)
    videos = scan_videos(root)
    timeline_rows: list[dict] = []
    summaries: list[dict] = []
    segments: list[dict] = []
    timelines_by_path: dict[str, list[dict]] = {}

    for path, label in videos:
        relative = str(path.relative_to(root))
        result = extractor.extract_video(path)
        rows = []
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) if capture.isOpened() else 0.0
        capture.release()
        if fps <= 0:
            fps = 30.0
        for source_row in result.timeline:
            row = {"video_path": relative, "gesture_label": label,
                   "sample_index": source_row["sample_index"],
                   "source_frame_number": source_row["source_frame_number"],
                   "timestamp_seconds": source_row["source_frame_number"] / fps,
                   "detected_hand_count": source_row["detected_hand_count"],
                   "hand_detected": source_row["hand_detected"],
                   "candidate_run_id": ""}
            rows.append(row)
        runs = _detection_runs(rows)
        run_number_by_sample = {}
        for run_no, run in enumerate(runs, 1):
            run_id = f"{relative}:{run_no:02d}"
            for r in run:
                r["candidate_run_id"] = run_id
                run_number_by_sample[r["sample_index"]] = run_id
            segments.append({"video_path": relative, "gesture_label": label,
                             "candidate_run_id": run_id,
                             "run_type": "consecutive_sampled_detections" if len(run)>1 else "isolated_detection",
                             "consecutive_detected_samples": len(run),
                             "start_sample_index": run[0]["sample_index"],
                             "end_sample_index": run[-1]["sample_index"],
                             "start_source_frame": run[0]["source_frame_number"],
                             "end_source_frame": run[-1]["source_frame_number"],
                             "start_time_seconds": run[0]["timestamp_seconds"],
                             "end_time_seconds": run[-1]["timestamp_seconds"],
                             "review_status": "MANUAL_REVIEW_REQUIRED_NOT_ASSUMED_SIGN"})
        detected = [r for r in rows if r["hand_detected"]]
        rate = len(detected)/len(rows) if rows else 0.0
        summaries.append({"video_path": relative, "gesture_label": label,
                          "total_sampled_frames": len(rows), "detected_frames": len(detected),
                          "detection_rate": rate,
                          "first_detected_frame": detected[0]["source_frame_number"] if detected else "",
                          "last_detected_frame": detected[-1]["source_frame_number"] if detected else "",
                          "active_segment_candidate": _candidate_description(runs, fps),
                          "candidate_run_count": len(runs),
                          "longest_consecutive_sampled_run": max((len(r) for r in runs), default=0),
                          "status": result.error or "processed",
                          "review_status": "MANUAL_REVIEW_REQUIRED_CANDIDATE_ONLY" if detected else "NO_DETECTION_MANUAL_REVIEW_REQUIRED"})
        for r in rows:
            timeline_rows.append(r)
        timelines_by_path[relative] = rows

    _write_csv(output_dir / "frame_detection_timeline.csv", timeline_rows,
               ["video_path","gesture_label","sample_index","source_frame_number","timestamp_seconds",
                "detected_hand_count","hand_detected","candidate_run_id"])
    _write_csv(output_dir / "video_segment_summary.csv", summaries,
               ["video_path","gesture_label","total_sampled_frames","detected_frames","detection_rate",
                "first_detected_frame","last_detected_frame","active_segment_candidate","candidate_run_count",
                "longest_consecutive_sampled_run","status","review_status"])
    _write_csv(output_dir / "candidate_detection_runs.csv", segments,
               ["video_path","gesture_label","candidate_run_id","run_type","consecutive_detected_samples",
                "start_sample_index","end_sample_index","start_source_frame","end_source_frame",
                "start_time_seconds","end_time_seconds","review_status"])

    representatives = _representative_rows(summaries)
    preview_selection=[]
    for item in representatives:
        preview = _make_preview(root, preview_dir, item, timelines_by_path[item["video_path"]], sample_every,
                                extractor.model_path)
        preview_selection.append({"role":"lowest/median/highest or within-class contrast",
                                  "video_path":item["video_path"],"gesture_label":item["gesture_label"],
                                  "detected_frames":item["detected_frames"],
                                  "total_sampled_frames":item["total_sampled_frames"],
                                  "detection_rate":item["detection_rate"],"preview_file":preview})
    _write_csv(output_dir / "preview_selection.csv", preview_selection,
               ["role","video_path","gesture_label","detected_frames","total_sampled_frames","detection_rate","preview_file"])
    return {"video_count":len(videos),"timeline_rows":len(timeline_rows),"candidate_runs":len(segments),
            "summary_rows":len(summaries),"preview_count":len(preview_selection),
            "output_dir":str(output_dir)}
