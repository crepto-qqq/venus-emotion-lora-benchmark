"""Apply deterministic, image-audited corrections to selected guidance text."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Sequence


CORRECTIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "amusement_05012": (
        (
            "'THE AMAZING ADVENTURES OF SPIDERMAN'",
            "'THE AMAZING ADVENTURES OF SPIDER-MAN'",
        ),
    ),
    "awe_05016": (
        (
            "- Crop slightly from the bottom to reduce the watermark and strengthen attention on the lake’s scale and the distant horizon.",
            "- Lift shadow detail slightly around the nearer swimmer so the human scale remains readable against the broad lake.",
        ),
    ),
    "fear_05005": (
        (
            "- A pale doll with dark eyes and blood-like marks sits in front while holding a narrow dark object.",
            "- A pale, doll-like costumed person with dark eye makeup and blood-like facial marks sits in front, holding a curved dark hook or handle.",
        ),
        (
            "The vertical arrangement layers the pale doll beneath the partially hidden animal-headed figure, and the near-black background makes both faces appear isolated and uncanny.",
            "The vertical arrangement layers the pale costumed figure beneath the partially hidden animal-headed figure, and the near-black background makes both faces appear isolated and uncanny.",
        ),
        (
            "- Crop some empty black space from the sides to tighten the relationship between the looming figure and the doll.",
            "- Crop some empty black space from the sides to tighten the relationship between the looming figure and the costumed figure.",
        ),
    ),
    "fear_05006": (
        (
            "- A large hippopotamus model faces the camera with its mouth opened wide and rows of pointed teeth exposed.",
            "- A large dinosaur model faces the camera with its mouth open wide and rows of pointed teeth exposed.",
        ),
        (
            "The close frontal viewpoint exaggerates the open jaws and makes the model appear to lunge out of an otherwise bright landscape, creating immediate visual threat.",
            "The close frontal viewpoint exaggerates the dinosaur’s open jaws and makes the model appear to lunge out of an otherwise bright landscape, creating immediate visual threat.",
        ),
        (
            "- Crop tighter around the head and mouth so the surrounding field does not reduce the animal's apparent scale.",
            "- Crop tighter around the head and mouth so the surrounding field does not reduce the dinosaur's apparent scale.",
        ),
    ),
    "fear_05007": (
        (
            "- A human-shaped figure in a long black coat stands motionless with its face completely hidden.",
            "- A human-shaped silhouette formed by a long black coat and broad-brimmed hat stands motionless with its face completely hidden.",
        ),
        (
            "- Reduce the bright window highlights and add slight edge contrast around the hood so the silhouette remains dominant but legible.",
            "- Reduce the bright window highlights and add slight edge contrast around the hat and shoulders so the silhouette remains dominant but legible.",
        ),
    ),
    "sadness_05008": (
        (
            "- Their slouched posture and loosely hanging hands create a withdrawn, heavy body language.",
            "- Their slouched posture and hands clasped low between their knees create withdrawn, heavy body language.",
        ),
    ),
    "sadness_05004": (
        (
            "- Increase the negative space around the solitary figure or crop to emphasize the person’s isolation against the vast background.",
            "- Preserve the broad empty sky while cropping slightly from the lower-left edge so the solitary silhouette remains isolated.",
        ),
    ),
    "disgust_05006": (
        (
            "- A dense cluster of small yellow-bodied insects is packed tightly together on the leaf surface.",
            "- A dense cluster of tiny yellow-and-black spider-like arthropods is packed tightly together on the leaf surface.",
        ),
        (
            "- Numerous thin white filaments and dark insect legs create a messy, crawling texture across the center of the image.",
            "- Their many overlapping, pale-banded legs create a tangled, crawling texture across the center of the image.",
        ),
        (
            "The close-up composition and central placement force attention onto the crowded infestation, while the dark green and yellow-brown palette gives the scene a dirty, unhealthy feel.",
            "The close-up composition and central placement force attention onto the crowded mass, while the dark green and yellow-brown palette gives the scene a dirty, unhealthy feel.",
        ),
        (
            "- Crop even tighter around the densest part of the insect cluster to intensify the sense of infestation.",
            "- Crop even tighter around the densest part of the arthropod cluster to intensify the sense of a dense crawling mass.",
        ),
    ),
    "excitement_05001": (
        (
            "- Increase shutter speed or use a slight motion-freeze crop on the strumming hand to sharpen the moment of energy rather than blurring it away",
            "- Use a slightly faster shutter speed to render the strumming hand more clearly while retaining enough motion to convey energy.",
        ),
    ),
    "excitement_05002": (
        (
            "- Position the frame so the raised hands are not clipped at the edge, since they are a key expressive element",
            "- Reduce the harsh highlights on the two central faces while keeping the vivid colours and raised hand gestures.",
        ),
    ),
    "excitement_05004": (
        (
            "- Four seated women leaning in attentively, one with a slight open-mouthed smile",
            "- Four other women stand closely together, turned toward the speaker with attentive or amused expressions.",
        ),
        (
            "The standing figure's raised, speaking posture and the seated group's forward lean and attentive expressions create a shared focal point, and the warm spotlighting isolates the group from the dark curtain, drawing attention to their engaged interaction rather than a still, posed scene.",
            "The standing speaker and the clustered group’s turned heads and engaged expressions create a shared focal point, while the warm stage lighting isolates the five performers from the dark curtain and emphasizes their interaction.",
        ),
    ),
    "excitement_05006": (
        (
            "- Both arms extended toward the lens, one hand pointing directly at the viewer",
            "- Both arms are extended in exaggerated gestures, with one finger pointing downward and the other toward the subject’s cheek.",
        ),
        (
            "The subject's direct point and forward-leaning stance break the fourth wall, pulling the viewer into the moment, while the isolated spotlight against the near-black background concentrates all visual attention on his exaggerated expression and vivid costume colours, reinforcing the high-energy mood.",
            "The foreshortened hands and front-facing grin pull the viewer into the moment, while the isolated spotlight against the near-black background concentrates attention on the exaggerated expression and vivid costume colours.",
        ),
        (
            "- Tighten the crop slightly around the head and pointing hand to emphasize the direct address to the camera",
            "- Tighten the crop slightly around the head and gesturing hands to emphasize the animated pose.",
        ),
    ),
}


class GuidanceCorrectionError(ValueError):
    """Raised when a locked correction cannot be applied exactly once."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def apply_corrections(*, dataset_root: Path, apply: bool) -> dict[str, Any]:
    remaining = set(CORRECTIONS)
    changed_by_path: dict[Path, list[dict[str, Any]]] = {}
    already_corrected_count = 0
    for path in sorted(dataset_root.glob("*/*.jsonl")):
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        changed = False
        for record in rows:
            source_id = record.get("source_image_id")
            replacements = CORRECTIONS.get(source_id)
            if replacements is None:
                continue
            target = record.get("target_response")
            if not isinstance(target, str):
                raise GuidanceCorrectionError(f"{source_id} target_response is invalid")
            target_changed = False
            for old, new in replacements:
                old_occurrences = target.count(old)
                new_occurrences = target.count(new)
                if old_occurrences == 1 and new_occurrences == 0:
                    target = target.replace(old, new, 1)
                    target_changed = True
                elif old_occurrences == 0 and new_occurrences == 1:
                    continue
                else:
                    raise GuidanceCorrectionError(
                        f"{source_id} expected exactly one source or corrected phrase; "
                        f"found old={old_occurrences}, new={new_occurrences}: {old}"
                    )
            if target_changed:
                record["target_response"] = target
            provenance = record.get("provenance")
            if not isinstance(provenance, dict):
                raise GuidanceCorrectionError(f"{source_id} provenance is invalid")
            existing_notes = str(provenance.get("review_notes", "")).strip()
            correction_note = (
                "BridgeTrain-v1 selected-target audit corrected a visible-detail or actionability issue; "
                "no model or API regeneration was used."
            )
            expected_correction = {
                "tool": "phase3/tools/apply_guidance_corrections.py",
                "version": 1,
                "replacement_count": len(replacements),
                "method": "Exact text replacement after source-image audit; no model or API call.",
            }
            provenance_changed = False
            if provenance.get("review_status") != "edited":
                provenance["review_status"] = "edited"
                provenance_changed = True
            if correction_note not in existing_notes:
                provenance["review_notes"] = (
                    f"{existing_notes} {correction_note}".strip()
                    if existing_notes
                    else correction_note
                )
                provenance_changed = True
            if provenance.get("bridge_train_v1_correction") != expected_correction:
                provenance["bridge_train_v1_correction"] = expected_correction
                provenance_changed = True
            remaining.remove(source_id)
            if target_changed or provenance_changed:
                changed = True
            else:
                already_corrected_count += 1
        if changed:
            changed_by_path[path] = rows

    if remaining:
        raise GuidanceCorrectionError(
            "correction records were not found: " + ", ".join(sorted(remaining))
        )
    if apply:
        for path, rows in changed_by_path.items():
            payload = "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in rows
            )
            descriptor, temporary_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as output:
                    output.write(payload)
                    output.flush()
                    os.fsync(output.fileno())
                os.replace(temporary, path)
            finally:
                if temporary.exists():
                    temporary.unlink()

    return {
        "valid": True,
        "applied": apply,
        "corrected_record_count": len(CORRECTIONS),
        "changed_file_count": len(changed_by_path),
        "already_corrected_record_count": already_corrected_count,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=project_root() / "data/datasets")
    parser.add_argument("--apply", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    report = apply_corrections(dataset_root=args.dataset_root.resolve(), apply=args.apply)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
