from src.landmark_extractor import LandmarkExtractor
from src.video_processor import load_label_mapping, scan_videos


def test_unreadable_video_is_reported():
    result = LandmarkExtractor().extract_video("this-file-does-not-exist.MOV")
    assert "Could not open video" in result.error


def test_label_mapping_uses_dataset_ids(tmp_path):
    (tmp_path / "labels.csv").write_text("id,label,category\n1,HELLO,GREETING\n", encoding="utf-8")
    assert load_label_mapping(tmp_path) == {"1": "HELLO"}
