from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


ROOT = Path("/tmp/temperature-equivalent-latitude-process/output")


def slug(level: float) -> str:
    return f"{level:g}".replace(".", "p").replace("-", "m") + "hpa"


def main() -> None:
    levels: list[tuple[float, float]] = []
    with (ROOT / "selected_buckets.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            levels.append((float(row["pressure_level_hpa"]), float(row["white_center"])))
    levels.sort(reverse=True)

    arrays = {
        level: np.load(ROOT / "arrays" / f"equivalent_latitude_smoothed_{slug(level)}.npy")
        for level, _ in levels
    }
    centers = {level: center for level, center in levels}

    print("adjacent_level_metrics")
    print(
        "upper_hpa,lower_hpa,field_corr,centered_corr,"
        "sign_agree,white_band_jaccard_5deg,mean_abs_diff_deg"
    )
    rows = []
    for (p1, c1), (p2, c2) in zip(levels[:-1], levels[1:]):
        a = arrays[p1].astype(np.float64)
        b = arrays[p2].astype(np.float64)
        mask = np.isfinite(a) & np.isfinite(b)
        av = a[mask]
        bv = b[mask]
        corr = float(np.corrcoef(av, bv)[0, 1])
        ac = av - c1
        bc = bv - c2
        centered_corr = float(np.corrcoef(ac, bc)[0, 1])
        sign_agree = float(np.mean(np.sign(ac) == np.sign(bc)))
        wa = np.abs(a - c1) < 5.0
        wb = np.abs(b - c2) < 5.0
        white_jaccard = float(
            np.logical_and(wa, wb).sum() / max(1, np.logical_or(wa, wb).sum())
        )
        mean_abs_diff = float(np.mean(np.abs(av - bv)))
        rows.append((p1, p2, corr, centered_corr, sign_agree, white_jaccard, mean_abs_diff))
        print(
            f"{p1:g},{p2:g},{corr:.3f},{centered_corr:.3f},"
            f"{sign_agree:.3f},{white_jaccard:.3f},{mean_abs_diff:.2f}"
        )

    print("\nblock_pair_corrs")
    blocks = [
        ("lower trop 1000-700", [1000, 975, 950, 925, 900, 875, 850, 825, 800, 775, 750, 700]),
        ("mid trop 700-400", [700, 650, 600, 550, 500, 450, 400]),
        ("upper trop 400-200", [400, 350, 300, 250, 225, 200]),
        ("strat 200-1", [200, 175, 150, 125, 100, 70, 50, 30, 20, 10, 7, 5, 3, 2, 1]),
    ]
    for name, block_levels in blocks:
        corr_values = []
        sign_values = []
        white_values = []
        available = [level for level in block_levels if level in arrays]
        for p1, p2 in zip(available[:-1], available[1:]):
            a = arrays[p1].ravel()
            b = arrays[p2].ravel()
            corr_values.append(float(np.corrcoef(a, b)[0, 1]))
            ac = a - centers[p1]
            bc = b - centers[p2]
            sign_values.append(float(np.mean(np.sign(ac) == np.sign(bc))))
            wa = np.abs(arrays[p1] - centers[p1]) < 5.0
            wb = np.abs(arrays[p2] - centers[p2]) < 5.0
            white_values.append(
                float(np.logical_and(wa, wb).sum() / max(1, np.logical_or(wa, wb).sum()))
            )
        print(
            name,
            "mean_corr",
            f"{np.mean(corr_values):.3f}",
            "mean_sign_agree",
            f"{np.mean(sign_values):.3f}",
            "mean_white_jaccard",
            f"{np.mean(white_values):.3f}",
        )

    key_levels = [1000, 925, 850, 700, 500, 400, 300, 250, 200, 150, 100, 50, 20, 10, 1]
    images = []
    for level in key_levels:
        path = ROOT / "maps" / f"map_{slug(level)}.png"
        image = Image.open(path).convert("RGB")
        image.thumbnail((450, 210), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (460, 245), "white")
        canvas.paste(image, (5, 25))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, 5), f"{level:g} hPa, white {centers[level]:.1f}", fill=(0, 0, 0))
        images.append(canvas)

    cols = 3
    rows_count = (len(images) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 460, rows_count * 245), (245, 245, 245))
    for index, image in enumerate(images):
        sheet.paste(image, ((index % cols) * 460, (index // cols) * 245))
    out = ROOT / "equivalent_latitude_key_levels_contact_sheet.png"
    sheet.save(out)
    print("\ncontact_sheet", out)


if __name__ == "__main__":
    main()
