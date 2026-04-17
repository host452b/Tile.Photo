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


from pathlib import Path
from PIL import Image


def test_save_usage_plot_writes_png(tmp_path: Path):
    from src.reporter import save_usage_plot
    usage = {i: (100 - i * 5) for i in range(20)}
    out_path = tmp_path / "hist.png"
    save_usage_plot(usage, str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0


def test_save_cold_wall_writes_png(tmp_path: Path):
    from src.reporter import save_cold_wall
    tile_paths = []
    for i in range(6):
        p = tmp_path / f"t{i}.jpg"
        Image.new("RGB", (64, 64), (i * 40, 100, 100)).save(p)
        tile_paths.append(str(p))
    usage = {0: 10, 1: 5}  # 2..5 cold
    out_path = tmp_path / "cold.png"
    save_cold_wall(tile_paths, usage, str(out_path), cols=2, tile_px=32)
    assert out_path.exists() and out_path.stat().st_size > 0
    img = Image.open(out_path)
    # 4 cold tiles, cols=2 → 2 rows × 2 cols
    assert img.size == (2 * 32, 2 * 32)
