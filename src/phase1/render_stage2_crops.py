"""Render private original-versus-crop composites for every Stage 2 record."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import textwrap

from .crop import normalized_to_pixel_box
from .manifest import read_manifest


def _read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"Invalid JSONL at line {line_number}: {error}") from error
    return records


def _fit(image, size: tuple[int, int]):
    from PIL import ImageOps

    return ImageOps.contain(image, size)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--emoset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    from PIL import Image, ImageDraw, ImageOps

    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Private render directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest = {row["sample_id"]: row for row in read_manifest(args.manifest)}
    records = _read_jsonl(args.records)
    index_rows: list[dict[str, str]] = []

    for record in records:
        if record.get("stage") != "stage2" or record.get("status") != "success":
            continue
        sample_id = record["sample_id"]
        if sample_id not in manifest:
            raise KeyError(f"Record sample_id is absent from manifest: {sample_id}")
        row = manifest[sample_id]
        image_path = (args.emoset_root / row["image_relpath"]).resolve()
        original = ImageOps.exif_transpose(Image.open(image_path)).convert("RGB")
        parse = record.get("crop_parse") or {}
        parse_status = str(parse.get("status", "missing_parse"))
        pixel_box: tuple[int, int, int, int] | None = None

        if parse_status == "valid" and parse.get("normalized_box"):
            normalized = tuple(float(value) for value in parse["normalized_box"])
            pixel_box = normalized_to_pixel_box(normalized, original.width, original.height)
            crop = original.crop(pixel_box)
        else:
            crop = None

        panel_width, panel_height = 700, 620
        header_height = 90
        canvas = Image.new("RGB", (panel_width * 2, panel_height + header_height), "white")
        draw = ImageDraw.Draw(canvas)
        draw.text((20, 15), f"{sample_id} | target emotion: {row['emotion']}", fill="black")
        draw.text((20, 42), f"parse_status: {parse_status}", fill="black")

        original_view = _fit(original, (panel_width - 30, panel_height - 40))
        canvas.paste(original_view, ((panel_width - original_view.width) // 2, header_height + 20))
        draw.text((20, header_height), "ORIGINAL", fill="black")

        if crop is not None:
            crop_view = _fit(crop, (panel_width - 30, panel_height - 40))
            x = panel_width + (panel_width - crop_view.width) // 2
            canvas.paste(crop_view, (x, header_height + 20))
            draw.text((panel_width + 20, header_height), "VENUS CROP", fill="black")
        else:
            draw.rectangle(
                (panel_width + 20, header_height + 20, panel_width * 2 - 20, panel_height + header_height - 20),
                fill="#eeeeee",
            )
            draw.text((panel_width + 40, header_height + 40), "NO VALID CROP", fill="black")
            wrapped = textwrap.wrap(str(record.get("response", "")), width=75)[:18]
            draw.multiline_text(
                (panel_width + 40, header_height + 80),
                "\n".join(wrapped),
                fill="black",
                spacing=5,
            )

        filename = f"{sample_id}.jpg"
        canvas.save(output_dir / filename, quality=92)
        index_rows.append(
            {
                "sample_id": sample_id,
                "target_emotion": row["emotion"],
                "parse_status": parse_status,
                "pixel_box": "" if pixel_box is None else " ".join(str(value) for value in pixel_box),
                "render_file": filename,
            }
        )

    with (output_dir / "render_index.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sample_id", "target_emotion", "parse_status", "pixel_box", "render_file"),
        )
        writer.writeheader()
        writer.writerows(index_rows)
    print(f"Rendered {len(index_rows)} private Stage 2 evaluation composites to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
