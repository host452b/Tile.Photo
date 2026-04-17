from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from mosaic.report import generate_report, _region_label


def test_region_label_top_means_sky():
    pos = [(0, 5), (0, 10), (1, 7)]
    assert _region_label(pos, rows=10, cols=20) == "天空"


def test_region_label_bottom_means_ground():
    pos = [(9, 5), (8, 10)]
    assert _region_label(pos, rows=10, cols=20) == "地面"


def test_region_label_center_means_subject():
    pos = [(5, 10), (5, 11), (4, 10)]
    assert _region_label(pos, rows=10, cols=20) == "主体"


def test_report_writes_markdown_with_stats(tmp_path):
    pool = {
        f"p{i}.jpg": {
            "mtime": 0.0,
            "lab_mean": np.array([50.0, 0.0, 0.0], dtype=np.float32),
            "thumbnail": np.full((16, 16, 3), i * 20, dtype=np.uint8),
        }
        for i in range(10)
    }
    usage = Counter({"p0.jpg": 50, "p1.jpg": 30, "p2.jpg": 5})
    positions = {"p0.jpg": [(0, 0), (0, 1)], "p1.jpg": [(5, 5)], "p2.jpg": [(9, 9)]}

    out_path = tmp_path / "report.md"
    chart_path = tmp_path / "chart.png"
    cold_path = tmp_path / "cold.png"

    generate_report(
        pool=pool, usage=usage, positions=positions, grid_shape=(10, 10),
        output_md=out_path, output_chart=chart_path, output_cold=cold_path,
    )

    md = out_path.read_text()
    assert "10" in md  # pool total
    assert "3" in md   # used count
    assert "p0.jpg" in md  # top tile
    assert chart_path.exists()
    assert cold_path.exists()
