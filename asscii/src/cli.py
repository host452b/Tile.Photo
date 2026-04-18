from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.pipeline import build, parse_bg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="asscii", description="photo → ASCII mosaic")
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cols", type=int, default=120)
    parser.add_argument("--rows", type=int, default=68)
    parser.add_argument("--density", type=float, default=0.6,
                        help="0.0 = sparsest (2 glyphs), 1.0 = densest (70 glyphs)")
    parser.add_argument("--bg", type=str, default="auto",
                        help="'auto' (冷宫 color), '#RRGGBB', or 'r,g,b'")
    parser.add_argument("--cell-px", type=int, default=12)
    parser.add_argument("--invert", action="store_true")
    parser.add_argument("--cold-pct", type=float, default=0.2)
    args = parser.parse_args(argv)

    with Image.open(args.input) as src:
        src = src.convert("RGB")
        _, result = build(
            src,
            cols=args.cols,
            rows=args.rows,
            density=args.density,
            bg=parse_bg(args.bg),
            cell_px=args.cell_px,
            invert=args.invert,
            cold_pct=args.cold_pct,
            output=args.output,
        )

    total = sum(result.char_usage.values())
    unique = len(result.char_usage)
    rarest = result.char_usage.most_common()[-1] if result.char_usage else ("", 0)
    print(f"wrote {args.output}  {result.cols}x{result.rows}  "
          f"cells={total}  distinct_chars={unique}  rarest={rarest[0]!r}×{rarest[1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
