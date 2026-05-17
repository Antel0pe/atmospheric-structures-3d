from __future__ import annotations

import argparse
import re
from pathlib import Path

from PIL import Image


LEVEL_RE = re.compile(r"map_([0-9]+(?:p[0-9]+)?)hpa\.png$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a crossfade GIF from equivalent-latitude map plots."
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=Path("/tmp/temperature-equivalent-latitude-process/output/maps"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/tmp/temperature-equivalent-latitude-process/output/"
            "equivalent_latitude_maps_high_to_low_pressure.gif"
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1200,
        help="GIF output width in pixels. Height preserves aspect ratio.",
    )
    parser.add_argument(
        "--transition-frames",
        type=int,
        default=5,
        help="Number of blended frames inserted between adjacent pressure levels.",
    )
    parser.add_argument(
        "--hold-ms",
        type=int,
        default=160,
        help="Duration for exact pressure-level frames.",
    )
    parser.add_argument(
        "--transition-ms",
        type=int,
        default=80,
        help="Duration for blended transition frames.",
    )
    return parser.parse_args()


def pressure_from_path(path: Path) -> float:
    match = LEVEL_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse pressure level from {path.name}")
    return float(match.group(1).replace("p", "."))


def resize(image: Image.Image, width: int) -> Image.Image:
    if image.width == width:
        return image
    height = round(image.height * width / image.width)
    return image.resize((width, height), Image.Resampling.LANCZOS)


def main() -> None:
    args = parse_args()
    paths = sorted(
        args.maps_dir.glob("map_*hpa.png"),
        key=pressure_from_path,
        reverse=True,
    )
    if not paths:
        raise FileNotFoundError(f"No map PNGs found in {args.maps_dir}")

    frames: list[Image.Image] = []
    durations: list[int] = []
    previous: Image.Image | None = None

    for path in paths:
        current = resize(Image.open(path).convert("RGB"), args.width)
        if previous is not None:
            for step in range(1, args.transition_frames + 1):
                alpha = step / (args.transition_frames + 1)
                frames.append(Image.blend(previous, current, alpha))
                durations.append(args.transition_ms)
        frames.append(current)
        durations.append(args.hold_ms)
        previous = current

    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(args.output)
    print(f"levels={len(paths)} frames={len(frames)}")
    print("order=" + ",".join(f"{pressure_from_path(path):g}" for path in paths))


if __name__ == "__main__":
    main()
