#!/usr/bin/env python3
"""Prepare and lock a private, blind human calibration of 16 guidance pairs.

Only the existing reviewer-facing CSV and images are read. This tool never opens
the condition answer key, historical AI scores, or Eval80 results. The resulting
human lock can be compared with a judge run after the human submission is final.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = ROOT / "tmp/pre-c-blind-comparison-interactive-20261014/review/review.csv"
DEFAULT_PACKET = ROOT / "tmp/pre-c-blind-comparison-interactive-20261014/pairwise-calibration"
SOURCE_FIELDS = {"anonymous_item_id", "image_code", "image_file", "target_emotion", "model_response"}
SUBMISSION_FIELDS = ("image_code", "reviewer_id", "preferred", "obvious_hallucinated_worse", "reason")
PREFERENCES = {"A", "B", "tie"}
HALLUCINATION_FLAGS = {"A", "B", "none", "uncertain"}
TRANSLATIONS_FILE = "answer_translations_zh.json"
EMOTION_ZH = {
    "amusement": "愉悦／好玩", "anger": "愤怒", "awe": "敬畏／惊叹",
    "contentment": "满足／宁静", "disgust": "厌恶", "excitement": "兴奋",
    "fear": "恐惧", "sadness": "悲伤",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def read_source(source: Path) -> list[dict]:
    """Check the blind source without opening private mappings or AI labels."""
    with source.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not SOURCE_FIELDS.issubset(reader.fieldnames):
            raise ValueError("Source is missing anonymous review columns")
        rows = list(reader)
    if len(rows) != 32:
        raise ValueError("Expected exactly 32 anonymous responses across 16 images")
    groups: dict[str, dict] = {}
    seen_items: set[str] = set()
    for row in rows:
        code = row["image_code"]
        if not code.startswith("D") or len(code) != 4 or not code[1:].isdigit():
            raise ValueError(f"Unexpected anonymous image code: {code!r}")
        item = row["anonymous_item_id"]
        if item in seen_items or item not in {f"{code}-A", f"{code}-B"}:
            raise ValueError(f"Duplicate or invalid anonymous item: {item!r}")
        seen_items.add(item)
        image_rel = row["image_file"]
        if image_rel != f"images/{code}.jpg":
            raise ValueError(f"Unexpected image path for {code}: {image_rel!r}")
        response = row["model_response"]
        if not response.strip():
            raise ValueError(f"Missing model response for {item}")
        group = groups.setdefault(code, {"image_code": code, "image_file": image_rel,
                                         "target_emotion": row["target_emotion"], "answers": {}})
        if group["target_emotion"] != row["target_emotion"] or group["image_file"] != image_rel:
            raise ValueError(f"Conflicting image metadata for {code}")
        group["answers"][item[-1]] = response
    if len(groups) != 16 or any(set(group["answers"]) != {"A", "B"} for group in groups.values()):
        raise ValueError("Expected one A/B pair for each of 16 images")
    result = []
    for code in sorted(groups):
        group = groups[code]
        image = source.parent / group["image_file"]
        if not image.is_file():
            raise ValueError(f"Missing image: {image}")
        group["source_image"] = image
        group["image_sha256"] = sha256_file(image)
        result.append(group)
    return result


def read_translations(path: Path, source: Path, groups: list[dict]) -> dict[str, str]:
    """Bind every Chinese aid to the exact blind answer it translates."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (not isinstance(payload, dict)
            or set(payload) != {"kind", "version", "source_review_csv_sha256", "records"}
            or payload.get("kind") != "blind_answer_translations_zh"
            or payload.get("version") != 1
            or payload.get("source_review_csv_sha256") != sha256_file(source)):
        raise ValueError("Translation file identity or source CSV hash is invalid")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != 32:
        raise ValueError("Translation file must contain exactly 32 answers")
    expected = {
        f"{group['image_code']}-{letter}": sha256_text(group["answers"][letter])
        for group in groups for letter in ("A", "B")
    }
    translations: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict) or set(record) != {
                "anonymous_item_id", "source_response_sha256", "translation_zh"}:
            raise ValueError("Translation record has unexpected fields")
        item = record["anonymous_item_id"]
        translation = record["translation_zh"]
        if (not isinstance(item, str) or item not in expected or item in translations
                or record["source_response_sha256"] != expected[item]
                or not isinstance(translation, str) or not translation.strip()):
            raise ValueError(f"Translation does not match blinded answer: {item!r}")
        translations[item] = translation
    if set(translations) != set(expected):
        raise ValueError("Translations do not cover the 32 blinded answers")
    return translations


def _html(groups: list[dict], form_id: str, translations: dict[str, str] | None = None) -> str:
    cards = []
    for index, group in enumerate(groups, 1):
        code = escape(group["image_code"])
        answers = ""
        for letter in ("A", "B"):
            original = escape(group["answers"][letter])
            if translations is None:
                answers += f'<section class="answer"><h3>回答 {letter}</h3><pre>{original}</pre></section>'
            else:
                translated = escape(translations[f"{code}-{letter}"])
                answers += (
                    f'<section class="answer"><h3>回答 {letter}</h3>'
                    '<h4>中文译文（辅助阅读）</h4>'
                    f'<pre class="translation" lang="zh-CN">{translated}</pre>'
                    '<h4>英文原文（核对依据）</h4>'
                    f'<pre class="original" lang="en">{original}</pre></section>'
                )
        preference = "".join(
            f'<label><input required type="radio" name="preferred-{code}" value="{value}">{label}</label>'
            for value, label in (("A", "A 更好"), ("B", "B 更好"), ("tie", "平局／无法区分"))
        )
        hallucination = "".join(
            f'<label><input required type="radio" name="obvious_hallucinated_worse-{code}" value="{value}">{label}</label>'
            for value, label in (("none", "没有"), ("A", "A 明显更差"),
                                 ("B", "B 明显更差"), ("uncertain", "无法确定"))
        )
        emotion = group["target_emotion"]
        emotion_label = (f"{EMOTION_ZH[emotion]}（{emotion}）"
                         if translations is not None and emotion in EMOTION_ZH else emotion)
        cards.append(
            f'<article class="case" id="{code}" data-code="{code}">'
            f'<h2>{index}/16 · {code} · 目标情绪：{escape(emotion_label)}</h2>'
            f'<img src="images/{code}.jpg" alt="匿名图片 {code}">'
            f'<div class="answers">{answers}</div>'
            f'<fieldset><legend>哪份指导建议更好？</legend>{preference}</fieldset>'
            '<fieldset><legend>是否有一份因严重虚构可见内容而明显更差？</legend>'
            f'{hallucination}</fieldset>'
            '<label class="reason">简短理由（指出可见依据、建议的具体性或主要错误）'
            '<textarea name="reason" rows="2" required></textarea></label>'
            '<p class="status" aria-live="polite">待评分</p></article>'
        )
    payload = json.dumps({
        ("form_id" if translations is not None else "manifest_sha256"): form_id,
        "rows": [{"image_code": group["image_code"]} for group in groups],
    }, ensure_ascii=False, separators=(",", ":"))
    payload = payload.replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    css = """
body{font:16px/1.55 system-ui,sans-serif;color:#202732;background:#f4f6f9;max-width:1320px;margin:auto;padding:1rem}
h1{margin:.2rem 0}.intro{max-width:72ch}.toolbar{position:sticky;top:0;z-index:3;background:white;border:1px solid #cfd7e2;border-radius:10px;padding:.7rem 1rem;display:flex;align-items:center;gap:1rem;flex-wrap:wrap;box-shadow:0 2px 8px #0002}
input,textarea,button{font:inherit}.toolbar input{padding:.35rem}.toolbar button{background:#185abc;color:white;border:0;border-radius:6px;padding:.6rem .9rem;cursor:pointer}
.case{background:#fff;margin:1.5rem 0;padding:1.2rem;border-radius:10px;box-shadow:0 2px 10px #0001}.case.done{outline:2px solid #2a8b5a}
.case>img{display:block;max-width:min(100%,750px);max-height:560px;object-fit:contain;margin:1rem auto}
.answers{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:1rem}.answer{background:#f3f6fa;padding:1rem;border-radius:8px}.answer h3{margin-top:0}.answer pre{background:white;padding:.7rem;border-radius:6px;white-space:pre-wrap;overflow-wrap:anywhere;font:inherit}
fieldset{display:flex;gap:1.2rem;flex-wrap:wrap;border:1px solid #ccd5df;border-radius:6px;margin:1rem 0;padding:.7rem}legend{font-weight:700}fieldset label{cursor:pointer}.reason{display:block;font-weight:700}.reason textarea{display:block;box-sizing:border-box;width:100%;margin-top:.4rem;padding:.55rem;font-weight:400}.status{font-size:.9rem;color:#626d7d}.case.done .status{color:#206b47;font-weight:700}
@media(max-width:650px){.toolbar{position:static}.answers{grid-template-columns:1fr}}"""
    if translations is not None:
        css += "\n.answer h4{margin:.8rem 0 .25rem}.answer .translation{border-left:3px solid #185abc}.answer .original{border-left:3px solid #8b949e}"
    js = r"""
(() => {
  'use strict';
  const data = JSON.parse(document.getElementById('calibration-data').textContent);
  const key = 'pre-c-pairwise-human-calibration:' + data.REVIEW_KEY;
  const cards = [...document.querySelectorAll('.case')];
  const reviewer = document.getElementById('reviewer-id');
  const progress = document.getElementById('progress');
  const note = document.getElementById('save-note');
  const fieldName = (card, name) => `${name}-${card.dataset.code}`;
  const value = (card, name) => card.querySelector(`input[name="${fieldName(card,name)}"]:checked`)?.value ?? '';
  function collect(card) {
    return {image_code:card.dataset.code, reviewer_id:reviewer.value.trim(),
      preferred:value(card,'preferred'), obvious_hallucinated_worse:value(card,'obvious_hallucinated_worse'),
      reason:card.querySelector('textarea[name="reason"]').value.trim()};
  }
  function valid(row) {
    return row.reviewer_id && row.preferred && row.obvious_hallucinated_worse && row.reason &&
      !(['A','B'].includes(row.obvious_hallucinated_worse) &&
        (row.preferred === row.obvious_hallucinated_worse || row.preferred === 'tie'));
  }
  function update() {
    let complete = 0;
    for (const card of cards) {
      const row = collect(card), good = valid(row);
      card.classList.toggle('done', Boolean(good));
      card.querySelector('.status').textContent = good ? '已完成' :
        (row.preferred && row.obvious_hallucinated_worse && ['A','B'].includes(row.obvious_hallucinated_worse) &&
          (row.preferred === row.obvious_hallucinated_worse || row.preferred === 'tie') ?
          '“明显更差”的回答必须与所选较好回答相反，请检查' : '待填写完整');
      if (good) complete++;
    }
    progress.textContent = `已完成 ${complete}/16`;
  }
  function save() {
    try {
      localStorage.setItem(key, JSON.stringify({reviewer_id:reviewer.value,rows:cards.map(collect)}));
      note.textContent = '草稿已保存在此浏览器；请定期下载 CSV。';
    } catch (_) { note.textContent = '本地自动保存不可用；请定期下载 CSV。'; }
  }
  function restore() {
    try {
      const raw = localStorage.getItem(key);
      if (!raw) return;
      const draft = JSON.parse(raw);
      reviewer.value = draft.reviewer_id ?? '';
      const byId = new Map((draft.rows ?? []).map(row => [row.image_code,row]));
      for (const card of cards) {
        const row = byId.get(card.dataset.code); if (!row) continue;
        for (const name of ['preferred','obvious_hallucinated_worse']) {
          const input = [...card.querySelectorAll(`input[name="${fieldName(card,name)}"]`)].find(input => input.value === row[name]);
          if (input) input.checked = true;
        }
        card.querySelector('textarea[name="reason"]').value = row.reason ?? '';
      }
      note.textContent = '已恢复本地草稿。';
    } catch (_) { note.textContent = '无法恢复草稿，可继续评分。'; }
  }
  const cell = v => `"${String(v ?? '').replaceAll('"','""')}"`;
  function download() {
    if (!reviewer.value.trim()) { reviewer.focus(); alert('请先填写评分者 ID。'); return; }
    const rows = cards.map(collect), complete = rows.every(valid);
    const cols = ['image_code','reviewer_id','preferred','obvious_hallucinated_worse','reason'];
    const csv = '\uFEFF' + cols.map(cell).join(',') + '\r\n' +
      rows.map(row => cols.map(key => cell(row[key])).join(',')).join('\r\n') + '\r\n';
    const url = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));
    const link = document.createElement('a'); link.href=url;
    link.download = complete ? 'human_calibration_submission.csv' : 'human_calibration_draft.csv';
    document.body.append(link); link.click(); link.remove();
    setTimeout(() => URL.revokeObjectURL(url),1000);
    if (!complete) alert('下载的是草稿；全部 16 张图完成后才能锁定。');
  }
  for (const card of cards) for (const event of ['input','change'])
    card.addEventListener(event,() => {update();save();});
  reviewer.addEventListener('input',() => {update();save();});
  document.getElementById('download').addEventListener('click',download);
  restore(); update();
})();""".replace("REVIEW_KEY", "form_id" if translations is not None else "manifest_sha256")
    intro = ("只看图片、目标情绪和 A/B 回答，选指导建议更好的一份；两份相当时选平局。"
             "若某份回答<strong>严重虚构画面关键内容，而且因此明显更差</strong>，再标记它。"
             "完成后下载 CSV。请勿打开同级目录以外的条件密钥或旧 AI 分数，直到人工结果锁定。")
    if translations is not None:
        intro = ("只看图片、目标情绪和 A/B 回答，选指导建议更好的一份；两份相当时选平局。"
                 "比较建议是否<strong>符合画面、具体可执行、能帮助达到目标情绪</strong>。"
                 "若某份回答<strong>严重虚构画面关键内容，而且因此明显更差</strong>，再标记它。"
                 "英文原文与中文译文同时展示；译文只帮助阅读，如有歧义请以英文原文为准。"
                 "完成后下载 CSV。请勿打开条件密钥或旧 AI 分数，直到人工结果锁定。")
    return f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>16 图匿名成对校准</title><style>{css}</style></head><body>
<h1>16 图匿名成对校准</h1>
<p class="intro">{intro}</p>
<p class="intro">这是 BridgeTrain 开发集上的<strong>自动评审器校准</strong>，不是 Eval80 的人工质量分。理由只需一两句话，重点写能从画面核实的依据和建议差异。</p>
<div class="toolbar"><label>评分者 ID <input id="reviewer-id" autocomplete="off"></label><strong id="progress">已完成 0/16</strong><button id="download" type="button">下载 CSV</button><span id="save-note">草稿仅在此浏览器本地保存。</span></div>
{''.join(cards)}
<script id="calibration-data" type="application/json">{payload}</script><script>{js}</script>
</body></html>"""


def prepare(source: Path, packet: Path) -> dict:
    if packet.exists():
        raise FileExistsError(f"Calibration packet already exists: {packet}")
    private_tmp = (ROOT / "tmp").resolve()
    packet = packet.resolve()
    if not packet.is_relative_to(private_tmp):
        raise ValueError("Packet must stay in the repository's Git-ignored tmp/")
    groups = read_source(source)
    packet.mkdir(parents=True)
    image_dir = packet / "images"
    image_dir.mkdir()
    manifest = {
        "kind": "pre_c_pairwise_human_calibration_manifest",
        "version": 1,
        "source_review_csv_sha256": sha256_file(source),
        "image_count": 16,
        "records": [],
    }
    for group in groups:
        code = group["image_code"]
        shutil.copyfile(group["source_image"], image_dir / f"{code}.jpg")
        manifest["records"].append({
            "image_code": code,
            "image_file": f"images/{code}.jpg",
            "image_sha256": group["image_sha256"],
            "target_emotion": group["target_emotion"],
            "response_a_sha256": sha256_text(group["answers"]["A"]),
            "response_b_sha256": sha256_text(group["answers"]["B"]),
        })
    manifest_path = packet / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_hash = sha256_file(manifest_path)
    (packet / "review.html").write_text(_html(groups, manifest_hash), encoding="utf-8")
    with (packet / "template.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUBMISSION_FIELDS)
        writer.writeheader()
        writer.writerows({"image_code": group["image_code"]} for group in groups)
    (packet / "README.md").write_text(
        "# 16 图匿名成对校准\n\n"
        "打开 `review.html`，填写评分者 ID，对 16 张图逐一选 A/B/平局、明显因幻觉更差的回答或没有/不确定，并写简短理由。"
        "下载 `human_calibration_submission.csv` 后运行：\n\n"
        "```powershell\n"
        "python phase3/tools/prepare_pre_c_pairwise_calibration.py lock --submission '<下载文件绝对路径>'\n"
        "```\n\n"
        "锁定脚本会核对 16 张图片、两份回答、CSV 列和每项选择，生成 `human_lock.json`。"
        "锁定之前请勿打开 `../private/answer_key.json` 或旧 AI 打分文件。"
        "此目录含私有 EmoSet 图片和回答，只能留在 Git 忽略的 `tmp/` 中。\n",
        encoding="utf-8",
    )
    return {"packet": str(packet), "manifest_sha256": manifest_hash, "image_count": 16}


def _bilingual_form_id(source_hash: str, translations_hash: str) -> str:
    return sha256_text(f"pre-c-pairwise-bilingual-v2:{source_hash}:{translations_hash}")


def bilingualize(packet: Path, source: Path) -> dict:
    """Upgrade the still-unlocked blind packet after 32 translations are ready."""
    if (packet / "human_lock.json").exists() or (packet / "human_submission.csv").exists():
        raise FileExistsError("A locked or reserved human submission cannot be changed")
    manifest, _, _ = _verify_packet(packet, source)
    translation_path = packet / TRANSLATIONS_FILE
    groups = read_source(source)
    translations = read_translations(translation_path, source, groups)
    translations_hash = sha256_file(translation_path)
    if manifest["version"] == 2:
        return {"packet": str(packet), "manifest_sha256": sha256_file(packet / "manifest.json"),
                "answer_translations_zh_sha256": translations_hash, "image_count": 16}
    source_hash = sha256_file(source)
    form_id = _bilingual_form_id(source_hash, translations_hash)
    html = _html(groups, form_id, translations)
    upgraded = {
        **manifest,
        "version": 2,
        "answer_translations_zh_sha256": translations_hash,
        "form_id": form_id,
        "review_html_sha256": sha256_text(html),
    }
    manifest_path = packet / "manifest.json"
    html_path = packet / "review.html"
    prior_manifest, prior_html = manifest_path.read_bytes(), html_path.read_bytes()
    try:
        with html_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(html)
        manifest_path.write_text(json.dumps(upgraded, ensure_ascii=False, indent=2) + "\n",
                                 encoding="utf-8")
        _verify_packet(packet, source)
    except Exception:
        html_path.write_bytes(prior_html)
        manifest_path.write_bytes(prior_manifest)
        raise
    (packet / "README.md").write_text(
        "# 16 图匿名成对校准（中英双语）\n\n"
        "打开 `review.html`。每份匿名回答同时显示中文译文和英文原文；译文仅供理解，歧义以英文原文为准。"
        "对 16 张图逐一选 A/B/平局、明显因幻觉更差的回答或没有/不确定，并写简短理由。"
        "下载 `human_calibration_submission.csv` 后运行：\n\n"
        "```powershell\n"
        "python phase3/tools/prepare_pre_c_pairwise_calibration.py lock --submission '<下载文件绝对路径>'\n"
        "```\n\n"
        "锁定脚本会核对 16 张图片、32 份匿名英文回答、32 份中文译文和表单哈希。"
        "锁定之前请勿打开 `../private/answer_key.json` 或旧 AI 打分文件。"
        "此目录含私有 EmoSet 图片和回答，只能留在 Git 忽略的 `tmp/` 中。\n",
        encoding="utf-8",
    )
    return {"packet": str(packet), "manifest_sha256": sha256_file(manifest_path),
            "answer_translations_zh_sha256": translations_hash, "image_count": 16}


def _verify_packet(packet: Path, source: Path) -> tuple[dict, str, str]:
    manifest_path = packet / "manifest.json"
    manifest_hash = sha256_file(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (manifest.get("kind") != "pre_c_pairwise_human_calibration_manifest"
            or manifest.get("version") not in {1, 2}
            or manifest.get("image_count") != 16):
        raise ValueError("Unexpected calibration manifest")
    groups = read_source(source)
    if manifest.get("source_review_csv_sha256") != sha256_file(source):
        raise ValueError("Source review CSV changed since calibration packet preparation")
    expected = [{
        "image_code": group["image_code"],
        "image_file": f"images/{group['image_code']}.jpg",
        "image_sha256": group["image_sha256"],
        "target_emotion": group["target_emotion"],
        "response_a_sha256": sha256_text(group["answers"]["A"]),
        "response_b_sha256": sha256_text(group["answers"]["B"]),
    } for group in groups]
    if manifest.get("records") != expected:
        raise ValueError("Manifest does not match the blinded source answers")
    for record in expected:
        if sha256_file(packet / record["image_file"]) != record["image_sha256"]:
            raise ValueError(f"Packet image changed: {record['image_code']}")
    review_html = packet / "review.html"
    if manifest["version"] == 2:
        translation_path = packet / TRANSLATIONS_FILE
        translations_hash = sha256_file(translation_path)
        if translations_hash != manifest.get("answer_translations_zh_sha256"):
            raise ValueError("Chinese translations changed since bilingual packet preparation")
        translations = read_translations(translation_path, source, groups)
        form_id = _bilingual_form_id(sha256_file(source), translations_hash)
        if manifest.get("form_id") != form_id:
            raise ValueError("Bilingual form identity does not match translation source")
        expected_html = _html(groups, form_id, translations)
        if manifest.get("review_html_sha256") != sha256_text(expected_html):
            raise ValueError("Reviewer HTML hash in manifest is invalid")
    else:
        expected_html = _html(groups, manifest_hash)
    if review_html.read_text(encoding="utf-8") != expected_html:
        raise ValueError("Reviewer HTML changed since packet preparation")
    return manifest, manifest_hash, sha256_file(review_html)


def lock(packet: Path, source: Path, submission: Path) -> dict:
    target = packet / "human_lock.json"
    saved_submission = packet / "human_submission.csv"
    if target.exists() or saved_submission.exists():
        raise FileExistsError(f"Human calibration already locked or reserved: {target}")
    manifest, manifest_hash, review_html_hash = _verify_packet(packet, source)
    submission_bytes = submission.read_bytes()
    with submission.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != list(SUBMISSION_FIELDS):
            raise ValueError(f"Submission must have exactly these columns: {SUBMISSION_FIELDS}")
        rows = list(reader)
    if len(rows) != 16:
        raise ValueError(f"Submission must contain exactly 16 rows, got {len(rows)}")
    by_id: dict[str, dict] = {}
    reviewers: set[str] = set()
    for row in rows:
        if None in row or any(value is None for value in row.values()):
            raise ValueError("Submission contains missing or extra CSV cells")
        code = row["image_code"].strip()
        if code in by_id:
            raise ValueError(f"Duplicate image in submission: {code}")
        reviewer = row["reviewer_id"].strip()
        preferred = row["preferred"].strip()
        hallucinated = row["obvious_hallucinated_worse"].strip()
        reason = row["reason"].strip()
        if not reviewer or preferred not in PREFERENCES or hallucinated not in HALLUCINATION_FLAGS or not reason:
            raise ValueError(f"Incomplete or invalid selection for {code}")
        if hallucinated in {"A", "B"} and preferred in {hallucinated, "tie"}:
            raise ValueError(f"Contradictory winner and obvious hallucinated worse answer for {code}")
        reviewers.add(reviewer)
        by_id[code] = {"image_code": code, "preferred": preferred,
                       "obvious_hallucinated_worse": hallucinated, "reason": reason}
    expected_codes = [record["image_code"] for record in manifest["records"]]
    if set(by_id) != set(expected_codes) or len(reviewers) != 1:
        raise ValueError("Submission IDs or reviewer IDs do not match the 16-image calibration")
    records = [by_id[code] for code in expected_codes]
    result = {
        "kind": "pre_c_pairwise_human_calibration_lock",
        "status": "locked",
        "locked_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest_sha256": manifest_hash,
        "review_html_sha256": review_html_hash,
        **({"answer_translations_zh_sha256": manifest["answer_translations_zh_sha256"]}
           if manifest["version"] == 2 else {}),
        "submission_file": "human_submission.csv",
        "submission_sha256": hashlib.sha256(submission_bytes).hexdigest(),
        "reviewer_id": next(iter(reviewers)),
        "image_count": 16,
        "clear_preference_count": sum(row["preferred"] in {"A", "B"} for row in records),
        "records": records,
    }
    with saved_submission.open("xb") as handle:
        handle.write(submission_bytes)
    with target.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return {"lock": str(target), "clear_preference_count": result["clear_preference_count"],
            "manifest_sha256": manifest_hash}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare", help="Prepare blinded 16-pair form once")
    prepare_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    prepare_parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    bilingual_parser = subparsers.add_parser("bilingualize", help="Upgrade unlocked form with 32 Chinese translations")
    bilingual_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    bilingual_parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    lock_parser = subparsers.add_parser("lock", help="Validate and freeze completed human choices")
    lock_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    lock_parser.add_argument("--packet", type=Path, default=DEFAULT_PACKET)
    lock_parser.add_argument("--submission", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        result = prepare(args.source, args.packet)
    elif args.command == "bilingualize":
        result = bilingualize(args.packet, args.source)
    else:
        result = lock(args.packet, args.source, args.submission)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
