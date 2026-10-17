"""Build a private, unscored, condition-blind Eval80 review page.

The page contains the original images, frozen target emotions and exact B1/C
responses under balanced A/B labels. A separate private key records the
condition mapping. This module never reads AI judge decisions or scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import secrets
from typing import Any

from .core import sha256_file
from .quality_benchmark import (
    _CONFIG, _EXCLUSION, _FREEZE, _HANDBOOK, _MANIFEST,
    _safe_image_path, _validated_inputs,
)


_ROOT = Path(__file__).resolve().parents[2]


def _private_file(path: Path, suffix: str) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to((_ROOT / "tmp").resolve()) or path.suffix != suffix:
        raise ValueError(f"Review outputs must be {suffix} files beneath the repository's ignored tmp/ directory.")
    return resolved


def _fenced_exact(text: str) -> str:
    longest_tildes = max((len(match.group()) for match in re.finditer(r"~+", text)), default=0)
    fence = "~" * max(3, longest_tildes + 1)
    return f"{fence}\n{text}\n{fence}"


def _json_bytes(payload: dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def build_unscored_review_pack(
    *, b1_run_dir: Path, c_run_dir: Path, selection_manifest: Path,
    image_dir: Path, output: Path, key_output: Path,
    inference_manifest: Path = _MANIFEST, freeze_record: Path = _FREEZE,
    display_exclusion: Path = _EXCLUSION, config: Path = _CONFIG,
    handbook: Path = _HANDBOOK, seed: int | None = None,
) -> tuple[Path, Path]:
    """Validate frozen sources before writing 78 anonymous pairs and a hidden key."""
    output = _private_file(output, ".md")
    key_output = _private_file(key_output, ".json")
    if output == key_output or output.exists() or key_output.exists():
        raise FileExistsError("Review page and private key must both be new files.")
    manifest_rows, selection_by_id, runs, binding = _validated_inputs(
        b1_dir=b1_run_dir, c_dir=c_run_dir,
        manifest_path=inference_manifest, selection_path=selection_manifest,
        image_dir=image_dir, freeze_path=freeze_record,
        config_path=config, handbook_path=handbook,
        exclusion_path=display_exclusion,
    )
    scope = binding["scope"]
    cases = sorted(
        (row for row in manifest_rows if row["blind_id"] in scope.included_ids),
        key=lambda row: row["blind_id"],
    )
    expected_ids = {f"E{index:03d}" for index in range(2, 80)}
    if len(cases) != 78 or {row["blind_id"] for row in cases} != expected_ids:
        raise ValueError("Anonymous guidance review requires exactly E002–E079.")

    # The fresh random seed is retained only in the separate, private condition
    # key. The balanced assignment is independent of model-judge outcomes.
    if seed is None:
        seed = secrets.randbits(64)
    if not isinstance(seed, int) or seed < 0:
        raise ValueError("Randomization seed must be a nonnegative integer.")
    shuffled_ids = sorted(expected_ids)
    random.Random(seed).shuffle(shuffled_ids)
    b1_in_a = set(shuffled_ids[:39])
    key: dict[str, Any] = {
        "schema_version": 1,
        "purpose": "private_condition_key_for_unscored_review",
        "evaluation_set": "Eval80-v1",
        "image_count": 78,
        "seed": seed,
        "balanced_A_counts": {"B1": 39, "C": 39},
        "freeze_sha256": binding["freeze_sha256"],
        "inference_manifest_sha256": binding["inference_manifest_sha256"],
        "selection_manifest_sha256": binding["selection_manifest_sha256"],
        "display_exclusion_sha256": scope.source_sha256,
        "source_records_sha256": {
            "B1": runs["B1"]["records_sha256"],
            "C": runs["C"]["records_sha256"],
        },
        "items": {},
    }
    lines = [
        "# Eval80 原图与双回答人工复核（匿名 A/B）", "",
        "> 本页没有自动判定、胜负或分数。请先独立记录看法，完成并锁定笔记后再查看单独保存的条件映射。", "",
        "> 回答格式可能暴露训练条件，故匿名标签不能保证完全盲评；请重点核对图片事实、建议可执行性及目标情绪。", "",
        "- 复核范围：78 张原图（E002–E079）；E001、E080 依已批准的展示排除记录不进入 guidance 复核。",
        f"- 冻结推理清单 SHA-256：`{binding['inference_manifest_sha256']}`",
        f"- 私有目标情绪清单 SHA-256：`{binding['selection_manifest_sha256']}`",
        f"- 展示排除记录 SHA-256：`{scope.source_sha256}`", "",
        "## 目录", "",
    ]
    lines.extend(f"- [{row['blind_id']}](#{row['blind_id'].lower()})" for row in cases)
    for row in cases:
        blind_id = row["blind_id"]
        path = _safe_image_path(image_dir, row["image_filename"])
        if sha256_file(path) != row["image_sha256"]:
            raise ValueError(f"{blind_id}: source image changed after frozen validation.")
        emotion = selection_by_id[blind_id]["emotion"]
        a_condition = "B1" if blind_id in b1_in_a else "C"
        b_condition = "C" if a_condition == "B1" else "B1"
        a = runs[a_condition]["records"][blind_id]
        b = runs[b_condition]["records"][blind_id]
        for record in (a, b):
            if (record["image_sha256"] != row["image_sha256"]
                    or hashlib.sha256(record["response"].encode("utf-8")).hexdigest()
                    != record["response_sha256"]):
                raise ValueError(f"{blind_id}: response or image hash mismatch.")
        key["items"][blind_id] = {
            "A": a_condition, "B": b_condition,
            "image_sha256": row["image_sha256"],
            "A_response_sha256": a["response_sha256"],
            "B_response_sha256": b["response_sha256"],
        }
        lines.extend([
            "", f"## {blind_id}", "",
            f"目标情绪：`{emotion}`", "",
            f"![{blind_id} 原图](<{path.as_posix()}>)", "",
            f"原图：`{path.as_posix()}`", "",
            "### 回答 A", "", _fenced_exact(a["response"]), "",
            "### 回答 B", "", _fenced_exact(b["response"]), "",
            "### 人工笔记", "",
            "视觉事实：", "", "建议是否具体可执行：", "",
            "是否保持目标情绪：", "", "整体偏好及原因：", "", "---",
        ])

    output.parent.mkdir(parents=True, exist_ok=True)
    key_output.parent.mkdir(parents=True, exist_ok=True)
    page_bytes = ("\n".join(lines).rstrip() + "\n").encode("utf-8")
    key["review_page_sha256"] = hashlib.sha256(page_bytes).hexdigest()
    # The key is intentionally not referenced by the reviewer-facing page.
    with key_output.open("xb") as handle:
        handle.write(_json_bytes(key))
    with output.open("xb") as handle:
        handle.write(page_bytes)
    return output, key_output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build a private, unscored 78-image A/B review pack.")
    for flag in ("b1-run-dir", "c-run-dir", "selection-manifest", "image-dir", "output", "key-output"):
        parser.add_argument(f"--{flag}", type=Path, required=True)
    parser.add_argument("--inference-manifest", type=Path, default=_MANIFEST)
    parser.add_argument("--freeze-record", type=Path, default=_FREEZE)
    parser.add_argument("--display-exclusion", type=Path, default=_EXCLUSION)
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--handbook", type=Path, default=_HANDBOOK)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args(argv)
    page, key = build_unscored_review_pack(
        b1_run_dir=args.b1_run_dir, c_run_dir=args.c_run_dir,
        selection_manifest=args.selection_manifest, image_dir=args.image_dir,
        output=args.output, key_output=args.key_output,
        inference_manifest=args.inference_manifest, freeze_record=args.freeze_record,
        display_exclusion=args.display_exclusion, config=args.config,
        handbook=args.handbook, seed=args.seed,
    )
    print(json.dumps({
        "status": "awaiting_blinded_human_review", "image_count": 78,
        "review_page": str(page), "review_page_sha256": sha256_file(page),
        "separate_private_key": str(key), "private_key_sha256": sha256_file(key),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
