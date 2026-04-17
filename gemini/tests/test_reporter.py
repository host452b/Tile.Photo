from src.reporter import build_text_report


def test_text_report_contains_key_stats():
    tile_paths = [f"/tile/{i}.jpg" for i in range(5)]
    usage = {0: 89, 1: 5, 2: 2, 3: 0, 4: 0}
    total_cells = sum(usage.values())
    tags = {
        "/tile/0.jpg": "2019 Japan trip",
        "/tile/1.jpg": "2019 Japan trip",
        "/tile/2.jpg": "work",
        "/tile/3.jpg": "untagged",
        "/tile/4.jpg": "untagged",
    }
    report = build_text_report(tile_paths, usage, tags, total_cells)
    # Must mention total photos used vs pool size
    assert "5" in report  # pool size
    assert "3" in report  # actually-used count
    # Must mention the top offender
    assert "/tile/0.jpg" in report
    assert "89" in report
    # Must mention cold photos
    assert "冷宫" in report or "cold" in report.lower()
    # Must mention top tag breakdown
    assert "Japan" in report
