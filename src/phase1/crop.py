"""Pure helpers for parsing and rendering Venus Stage 2 crop coordinates."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import re
from typing import Any


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
_POINT_RE = re.compile(rf"\(\s*({_NUMBER})\s*,\s*({_NUMBER})\s*\)")


@dataclass(frozen=True)
class CropParseResult:
    status: str
    normalized_box: tuple[float, float, float, float] | None
    point_count: int
    message: str

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.normalized_box is not None:
            data["normalized_box"] = list(self.normalized_box)
        return data


def parse_crop_box(response: str) -> CropParseResult:
    """Parse exactly two ``(x,y)`` points in Venus's normalized 0-1000 space."""

    points = [(float(x), float(y)) for x, y in _POINT_RE.findall(response)]
    if not points:
        return CropParseResult("no_coordinates", None, 0, "No coordinate pair was found.")
    if len(points) != 2:
        return CropParseResult(
            "wrong_coordinate_count",
            None,
            len(points),
            "Expected exactly two coordinate pairs.",
        )

    (x1, y1), (x2, y2) = points
    box = (x1, y1, x2, y2)
    if not all(math.isfinite(value) for value in box):
        return CropParseResult("non_finite", None, 2, "Coordinates must be finite numbers.")
    if not all(0.0 <= value <= 1000.0 for value in box):
        return CropParseResult("out_of_range", box, 2, "Coordinates must be within [0, 1000].")
    if x1 >= x2 or y1 >= y2:
        return CropParseResult(
            "non_positive_area",
            box,
            2,
            "Top-left must be strictly above and left of bottom-right.",
        )
    return CropParseResult("valid", box, 2, "Valid normalized crop box.")


def normalized_to_pixel_box(
    normalized_box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """Convert a valid normalized box to a clipped PIL crop box."""

    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be positive.")

    x1, y1, x2, y2 = normalized_box
    left = max(0, min(width, math.floor(x1 * width / 1000.0)))
    top = max(0, min(height, math.floor(y1 * height / 1000.0)))
    right = max(0, min(width, math.ceil(x2 * width / 1000.0)))
    bottom = max(0, min(height, math.ceil(y2 * height / 1000.0)))
    if left >= right or top >= bottom:
        raise ValueError("Scaled crop has non-positive area.")
    return left, top, right, bottom
