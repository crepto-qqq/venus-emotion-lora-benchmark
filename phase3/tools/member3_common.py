#!/usr/bin/env python3
"""Shared, Git-safe helpers for Phase 3 Member 3 LoRA tooling."""
from __future__ import annotations

import hashlib, json, os, re, shlex, subprocess, tempfile, math
from pathlib import Path
from typing import Any

TOOL_VERSION = "member3-v1"
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
ROOT = Path(__file__).resolve().parents[2]
RELEASE_DIR = ROOT / "phase3/releases/bridge-train-v1"
EXPECTED_RELEASE_HASHES = {
    "bridge-train-v1.train.json": "bc6c54aec636d0e56afad30ff5a004b48ccc9bcddbf76d0e91ab419b274df0d8",
    "bridge-train-v1.validation.json": "eafadfc9b6724ad9075a56e521ede4eae0db3ad541b2797c83b4230f4ad8f98b",
    "bridge-train-v1.manifest.jsonl": "441e7b7d69d57ca74387796b469a1a754d7ce69f0f24ee9230404910384b738b",
    "bridge-train-v1.rejections.jsonl": "928c07335f11df6024d73c6a00bcefcda7c88dcdfdb308147cf7926beee168be",
    "bridge-train-v1.summary.json": "600d0ba014721cdb4d4b39618a2fa6a24e1fef8e95136792c2c9256513ec2086",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def canonical_sha256(value: Any, omit_key: str | None = None) -> str:
    if omit_key and isinstance(value, dict):
        value = {k: v for k, v in value.items() if k != omit_key}
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: Any, *, canonical: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if canonical:
        payload = canonical_json_bytes(value)
    else:
        payload = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as f: f.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def merge_patch(target: Any, patch: Any) -> Any:
    """RFC 7396 JSON Merge Patch."""
    if not isinstance(patch, dict): return patch
    if not isinstance(target, dict): target = {}
    out = dict(target)
    for key, value in patch.items():
        if value is None: out.pop(key, None)
        else: out[key] = merge_patch(out.get(key), value)
    return out


def dotted_leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if not isinstance(value, dict): return {prefix} if prefix else set()
    result: set[str] = set()
    for k, v in value.items():
        p = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict): result |= dotted_leaf_paths(v, p)
        else: result.add(p)
    return result


def task_type(sample_id: str) -> str:
    if sample_id.endswith("__classification"): return "classification"
    if sample_id.endswith("__guidance") or sample_id.endswith("__joint_guidance"): return "joint_guidance"
    raise ValueError(f"unknown BridgeTrain task suffix: {sample_id}")


def verify_release_hashes(root: Path = ROOT) -> list[str]:
    failures=[]
    release=root/"phase3/releases/bridge-train-v1"
    for name, expected in EXPECTED_RELEASE_HASHES.items():
        p=release/name
        if not p.is_file(): failures.append(f"missing release file: {name}")
        elif sha256_file(p)!=expected: failures.append(f"release hash mismatch: {name}")
    if not failures:
        s=read_json(release/"bridge-train-v1.summary.json")
        if s.get("full_dataset_ready") is not True: failures.append("full_dataset_ready is not true")
        if s.get("formal_training_authorized") is not False: failures.append("dataset formal_training_authorized must remain false")
    return failures


def git_commit(root: Path = ROOT) -> str:
    try:
        return subprocess.run(["git","-C",str(root),"rev-parse","HEAD"],capture_output=True,text=True,check=True).stdout.strip()
    except Exception:
        return "unavailable-no-git-metadata"


def validate_resource_limits(limits: dict[str, Any] | None) -> list[str]:
    if not isinstance(limits, dict): return ["resource_limits must be supplied before a paid optimizer run"]
    req={"allowed_gpu","max_hourly_usd","total_budget_usd","max_wall_clock_minutes","max_attempts","oom_policy"}
    failures=[f"missing resource limit: {k}" for k in sorted(req-set(limits))]
    for key in ("max_hourly_usd", "total_budget_usd", "max_wall_clock_minutes", "max_attempts"):
        value = limits.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            failures.append(f"{key} must be a finite positive number")
    if not isinstance(limits.get("max_attempts"), int): failures.append("max_attempts must be an integer")
    if not isinstance(limits.get("allowed_gpu"), str) or not limits["allowed_gpu"].strip(): failures.append("allowed_gpu must be explicit")
    if limits.get("oom_policy") not in {"stop_and_request","same_gpu_retry_only","user_authorized_gpu_change"}: failures.append("invalid oom_policy")
    return failures


def build_finetune_command(config: dict[str, Any], interface: dict[str, Any]) -> list[str]:
    runtime=config["runtime"]; lora=config["lora"]; train=config["training"]; data=config["data"]; out=config["output"]
    values={
      "model_name_or_path": runtime["model_path"], "data_path": data["train_path"], "output_dir": out["run_dir"],
      "use_lora": True, "q_lora": lora["mode"]=="qlora", "fix_vit": train["fix_vit"],
      "lora_r": lora["rank"], "lora_alpha": lora["alpha"], "lora_dropout": lora["dropout"], "lora_bias": lora["bias"],
      "lora_target_modules": lora["target_modules"], "lora_modules_to_save": lora["modules_to_save"],
      "bf16": train["mixed_precision"]=="bf16", "fp16": train["mixed_precision"]=="fp16",
      "per_device_train_batch_size": train["micro_batch_size"], "gradient_accumulation_steps": train["gradient_accumulation_steps"],
      "gradient_checkpointing": train["gradient_checkpointing"], "learning_rate": train["learning_rate"], "weight_decay": train["weight_decay"],
      "max_grad_norm": train["max_grad_norm"], "num_train_epochs": train["epochs"], "max_steps": train["max_steps"],
      "lr_scheduler_type": train["scheduler"], "warmup_ratio": train["warmup_ratio"], "logging_steps": train["logging_steps"],
      "save_strategy": train["save_strategy"], "save_steps": train["save_steps"], "save_total_limit": train["save_total_limit"],
      "seed": train["seed"], "dataloader_num_workers": train["workers"], "model_max_length": train["max_sequence_length"],
      "remove_unused_columns": False, "report_to": train["report_to"], "cache_dir": runtime["cache_dir"],
    }
    mapping=interface["config_to_cli"]
    command=[runtime["python"], interface["entry_point"]]
    for key,val in values.items():
        flag=mapping.get(key)
        if not flag: raise ValueError(f"no interface mapping for {key}")
        if val is None: continue
        if isinstance(val,bool): command += [flag, "True" if val else "False"]
        elif isinstance(val,list):
            if val: command += [flag, *map(str,val)]
        else: command += [flag,str(val)]
    return command


def command_text(command: list[str]) -> str:
    return shlex.join(command)


def verify_patched_checkout(checkout: Path, lock: dict[str, Any]) -> str:
    """Reconstruct the reviewed patch in a temp directory; never patch shared checkout."""
    expected=lock['upstream']['commit']
    actual=subprocess.check_output(['git','-C',str(checkout),'rev-parse','HEAD'],text=True).strip()
    if actual!=expected: raise ValueError('upstream commit mismatch')
    patch=ROOT/lock['upstream']['patch']
    if sha256_file(patch)!=lock['upstream']['patch_sha256']: raise ValueError('reviewed patch hash mismatch')
    original=subprocess.check_output(['git','-C',str(checkout),'show',expected+':finetune.py'])
    with tempfile.TemporaryDirectory(prefix='member3-source-check-') as directory:
        target=Path(directory)/'finetune.py';target.write_bytes(original)
        subprocess.run(['git','apply',str(patch)],cwd=directory,check=True,capture_output=True)
        expected_hash=sha256_file(target)
    if sha256_file(checkout/'finetune.py')!=expected_hash: raise ValueError('finetune.py differs from pinned source plus reviewed patch')
    return expected_hash
