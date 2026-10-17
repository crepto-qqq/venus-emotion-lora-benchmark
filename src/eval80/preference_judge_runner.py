"""Run blind, swapped-order preference decisions against a local vision endpoint.

The wire prompt and JSON schema match the separately calibrated pairwise
preflight. Auxiliary preference fields are retained only in raw private logs;
the scored contract uses the overall winner and concise reason.
"""

from __future__ import annotations

import argparse
import base64
from binascii import Error as BinasciiError
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import ProxyHandler, Request, build_opener

from .core import sha256_file
from .preference_benchmark import (
    _identity_mode, _is_loopback_endpoint, _lock_preference_judge, _packet_sha256,
    _read_jsonl, _read_object, _validate_judgements, _validate_model_receipt,
    _validate_preference_config, _validate_preflight_receipt,
    _validate_runtime_identity, _validate_script_trace,
)
from .quality_judge_runner import _NoRedirectHandler, _append_jsonl, _post_chat


def _preference_schema() -> dict[str, Any]:
    choice = {"type": "string", "enum": ["A", "B", "tie"]}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "preferred": choice,
            "better_grounding": choice,
            "better_actionability": choice,
            "better_target_fit": choice,
            "rationale": {"type": "string"},
        },
        "required": [
            "preferred", "better_grounding", "better_actionability",
            "better_target_fit", "rationale",
        ],
    }


def _make_payload(packet: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(packet["image_path"])
    if sha256_file(image_path) != packet["image_sha256"]:
        raise ValueError("Preference image differs from its frozen SHA-256.")
    for letter in ("A", "B"):
        answer = packet[f"answer_{letter}"]
        if hashlib.sha256(answer.encode("utf-8")).hexdigest() != packet[f"answer_{letter}_sha256"]:
            raise ValueError(f"Preference answer {letter} differs from its frozen SHA-256.")
    mime = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_url = f"data:{mime};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    # Preserve the calibrated wire text and image-first content-part order.
    prompt = (config["prompt"] + "\n\nFixed target emotion: " + packet["target_emotion"]
              + "\n\nAnswer A:\n" + packet["answer_A"]
              + "\n\nAnswer B:\n" + packet["answer_B"])
    decoding = config["decoding"]
    if (set(decoding) - {"temperature", "seed", "max_tokens", "top_p"}
            or "max_tokens" not in decoding):
        raise ValueError("Preference decoding needs max_tokens and supported parameters only.")
    if config["image_detail"] not in {"auto", "low", "high"}:
        raise ValueError("Preference image detail must be auto, low or high.")
    return {
        "model": config["model_id"],
        **decoding,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url,
                                                  "detail": config["image_detail"]}},
            {"type": "text", "text": prompt},
        ]}],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "paired_guidance_preference", "strict": True,
            "schema": _preference_schema(),
        }},
    }


def _request_sha256(payload: dict[str, Any]) -> str:
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _preflight_request_image_sha256(
    request: dict[str, Any], config: dict[str, Any], schema: dict[str, Any],
) -> str:
    """Verify a calibration request used the frozen wire shape and a data image."""
    expected_keys = {"model", "messages", "response_format"} | set(config["decoding"])
    if (set(request) != expected_keys or request.get("model") != config["model_id"]
            or any(request.get(key) != value for key, value in config["decoding"].items())
            or request.get("response_format") != {
                "type": "json_schema", "json_schema": {
                    "name": "paired_guidance_preference", "strict": True,
                    "schema": schema,
                },
            }):
        raise ValueError("Preference calibration request differs from frozen model or schema.")
    messages = request.get("messages")
    if (not isinstance(messages, list) or len(messages) != 1
            or not isinstance(messages[0], dict)
            or messages[0].get("role") != "user"):
        raise ValueError("Preference calibration request has unexpected messages.")
    content = messages[0].get("content")
    if (not isinstance(content, list) or len(content) != 2
            or not isinstance(content[0], dict) or content[0].get("type") != "image_url"
            or not isinstance(content[1], dict) or content[1].get("type") != "text"):
        raise ValueError("Preference calibration request changed image/text order.")
    text = content[1].get("text")
    if (not isinstance(text, str)
            or not text.startswith(config["prompt"] + "\n\nFixed target emotion: ")
            or "\n\nAnswer A:\n" not in text or "\n\nAnswer B:\n" not in text):
        raise ValueError("Preference calibration request changed the prompt template.")
    image = content[0].get("image_url")
    if (not isinstance(image, dict) or image.get("detail") != config["image_detail"]
            or not isinstance(image.get("url"), str)
            or not image["url"].startswith("data:image/")):
        raise ValueError("Preference calibration request has no frozen image part.")
    try:
        prefix, encoded = image["url"].split(",", 1)
        if not prefix.endswith(";base64"):
            raise ValueError("Image URL is not base64.")
        pixels = base64.b64decode(encoded, validate=True)
    except (ValueError, BinasciiError) as error:
        raise ValueError("Preference calibration image data URL is malformed.") from error
    return hashlib.sha256(pixels).hexdigest()


def _probe_ollama_digest(
    endpoint: str, model_id: str, expected_digest: str, timeout: int,
    *, phase: str,
) -> dict[str, str]:
    """Read only local Ollama tag metadata; never transmit a private image."""
    parsed = urlparse(endpoint)
    if (parsed.hostname not in {"localhost", "127.0.0.1", "::1"}
            or parsed.port != 11434 or parsed.scheme not in {"http", "https"}):
        raise ValueError("Ollama runtime digest probe requires loopback port 11434.")
    request = Request(f"{parsed.scheme}://{parsed.netloc}/api/tags", method="GET")
    opener = build_opener(ProxyHandler({}), _NoRedirectHandler())
    try:
        with opener.open(request, timeout=timeout) as result:
            body = result.read(1_000_001)
    except HTTPError as error:
        raise RuntimeError(f"Ollama digest endpoint HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(
            f"Ollama digest endpoint transport error: {type(error.reason).__name__}"
        ) from error
    if len(body) > 1_000_000:
        raise ValueError("Ollama digest endpoint response is too large.")
    data = json.loads(body.decode("utf-8"))
    models = data.get("models") if isinstance(data, dict) else None
    if not isinstance(models, list):
        raise ValueError("Ollama digest endpoint has no model list.")
    matches = [row for row in models if isinstance(row, dict) and row.get("name") == model_id]
    if len(matches) != 1 or matches[0].get("digest") != expected_digest:
        raise ValueError("Live Ollama model digest differs from the frozen receipt.")
    return {
        "phase": phase,
        "observed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "digest_sha256": expected_digest,
    }


def _extract_judgement(
    response: dict[str, Any], packet: dict[str, Any], package: dict[str, Any],
) -> dict[str, Any]:
    if response.get("model") != package["judge_model_id"]:
        raise ValueError("Preference response model differs from pinned judge model ID.")
    expected_fingerprint = package.get("expected_system_fingerprint")
    if expected_fingerprint and response.get("system_fingerprint") != expected_fingerprint:
        raise ValueError("Preference response fingerprint differs from frozen config.")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Preference response must contain one choice.")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Preference response did not finish normally.")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Preference response has no JSON content.")
    decision = json.loads(message["content"])
    expected = {
        "preferred", "better_grounding", "better_actionability",
        "better_target_fit", "rationale",
    }
    if not isinstance(decision, dict) or set(decision) != expected:
        raise ValueError("Preference decision differs from calibrated JSON schema.")
    for name in expected - {"rationale"}:
        if decision[name] not in {"A", "B", "tie"}:
            raise ValueError(f"Preference decision {name} must be A, B or tie.")
    reason = decision["rationale"]
    if not isinstance(reason, str) or not reason.strip() or len(reason) > 800:
        raise ValueError("Preference decision needs a concise rationale.")
    response_id = response.get("id")
    if not isinstance(response_id, str) or not response_id:
        raise ValueError("Preference response ID is required for audit.")
    return {
        "anonymous_item_id": packet["anonymous_item_id"],
        "packet_sha256": _packet_sha256(packet),
        "preference_config_sha256": package["preference_config_sha256"],
        "judge_model_id": package["judge_model_id"],
        "winner": decision["preferred"],
        "reason": reason,
        "response_metadata": {
            "response_id": response_id,
            "model": response["model"],
            "system_fingerprint": response.get("system_fingerprint"),
            "created": response.get("created"),
            "usage": response.get("usage"),
            "finish_reason": choice["finish_reason"],
        },
    }


def judge_preference_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(
        "Run and lock 156 swapped-order B1/C preference decisions on a loopback vision endpoint."
    ))
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--api-key-env", default="EVAL80_JUDGE_API_KEY")
    args = parser.parse_args(argv)
    config = _read_object(args.judge_config)
    _validate_preference_config(config)
    if not _is_loopback_endpoint(config["endpoint"]):
        raise ValueError("Preference judging is restricted to loopback.")
    package = _read_object(args.package_dir / "preference_package.json")
    config_snapshot = args.package_dir / "preference_config_private.json"
    if (sha256_file(args.judge_config) != package["preference_config_sha256"]
            or sha256_file(config_snapshot) != package["preference_config_snapshot_sha256"]
            or sha256_file(Path(__file__)) != package["runner_source_sha256"]
            or sha256_file(Path(__file__).with_name("preference_benchmark.py"))
            != package["benchmark_source_sha256"]):
        raise ValueError("Preference config or code changed after package preparation.")
    receipt = args.package_dir / "model_digest_receipt_private.json"
    if (sha256_file(receipt) != package["model_digest_receipt_sha256"]
            or _validate_model_receipt(_read_object(receipt), config["model_id"])
            != package["model_digest_sha256"]):
        raise ValueError("Preference model digest receipt changed after preparation.")
    if (hashlib.sha256(config["prompt"].encode("utf-8")).hexdigest()
            != package["prompt_sha256"]):
        raise ValueError("Preference prompt differs from frozen package.")
    preflight = args.package_dir / "preflight_receipt_private.json"
    if sha256_file(preflight) != package["preflight_receipt_sha256"]:
        raise ValueError("Preference calibration receipt changed after preparation.")
    preflight_binding = _validate_preflight_receipt(
        preflight, config, package["model_digest_sha256"]
    )
    if any(package.get(key) != value for key, value in preflight_binding.items()):
        raise ValueError("Preference calibration evidence changed after preparation.")
    if (args.package_dir / "preference_judgements_lock.json").exists():
        raise FileExistsError("Preference decisions are already locked; use report-preference.")
    retry = config["retry_policy"]
    max_attempts, timeout, backoff = (
        retry.get("max_attempts"), retry.get("timeout_seconds"), retry.get("backoff_seconds")
    )
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("Preference max_attempts must be an integer from 1 to 5.")
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise ValueError("Preference timeout_seconds must be an integer from 1 to 600.")
    if type(backoff) not in (int, float) or not 0 <= backoff <= 30:
        raise ValueError("Preference backoff_seconds must be between 0 and 30.")
    packet_path = args.package_dir / "preference_items_private.jsonl"
    if sha256_file(packet_path) != package["preference_items_sha256"]:
        raise ValueError("Preference packets changed after preparation.")
    packets = _read_jsonl(packet_path)
    if len(packets) != 156:
        raise ValueError("Preference package does not contain 156 items.")
    state_path = args.package_dir / "preference_run_state.json"
    state_base = {
        "schema_version": 1,
        "preference_package_sha256": sha256_file(args.package_dir / "preference_package.json"),
        "preference_config_sha256": package["preference_config_sha256"],
        "preference_items_sha256": package["preference_items_sha256"],
        "benchmark_source_sha256": package["benchmark_source_sha256"],
        "runner_source_sha256": package["runner_source_sha256"],
    }
    mode = _identity_mode(config)
    if state_path.exists():
        state = _read_object(state_path)
        if (set(state) != set(state_base) | {"runtime_model_identity"}
                or any(state.get(name) != value for name, value in state_base.items())):
            raise ValueError("Preference run cannot resume with different frozen inputs.")
        _validate_runtime_identity(
            state["runtime_model_identity"], config, package, complete=False
        )
    else:
        state = {**state_base, "runtime_model_identity": {"mode": mode, "checks": []}}
    if mode == "ollama_api_tags":
        state["runtime_model_identity"]["checks"].append(
            _probe_ollama_digest(
                config["endpoint"], config["model_id"], package["model_digest_sha256"],
                min(timeout, 30), phase="before",
            )
        )
    if state_path.exists():
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    else:
        with state_path.open("x", encoding="utf-8") as handle:
            json.dump(state, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    raw_path = args.package_dir / "preference_raw_responses_private.jsonl"
    attempts_path = args.package_dir / "preference_attempts_private.jsonl"
    score_path = args.package_dir / "preference_judgements_private.jsonl"
    for path in (raw_path, attempts_path):
        path.touch(exist_ok=True)
    completed = (_validate_judgements(score_path, packet_path, package, require_complete=False)
                 if score_path.exists() else {})
    _validate_script_trace(args.package_dir, package, completed,
                           require_post_check=False)
    prior_attempts = Counter(
        row["anonymous_item_id"] for row in _read_jsonl(attempts_path)
    )
    key = os.environ.get(args.api_key_env)
    for position, packet in enumerate(packets, start=1):
        anonymous_id = packet["anonymous_item_id"]
        if anonymous_id in completed:
            continue
        payload = _make_payload(packet, config)
        request_hash = _request_sha256(payload)
        for attempt in range(prior_attempts[anonymous_id] + 1, max_attempts + 1):
            try:
                raw = _post_chat(config["endpoint"], payload, key, timeout)
                _append_jsonl(raw_path, {
                    "anonymous_item_id": anonymous_id,
                    "attempt": attempt,
                    "packet_sha256": _packet_sha256(packet),
                    "request_sha256": request_hash,
                    "response": raw,
                })
                result = _extract_judgement(raw, packet, package)
                temporary = args.package_dir / "preference_single_validation.tmp"
                with temporary.open("x", encoding="utf-8") as handle:
                    handle.write(json.dumps(result, ensure_ascii=False) + "\n")
                try:
                    _validate_judgements(temporary, packet_path, package, require_complete=False)
                finally:
                    temporary.unlink(missing_ok=True)
                _append_jsonl(score_path, result)
                _append_jsonl(attempts_path, {
                    "anonymous_item_id": anonymous_id,
                    "attempt": attempt,
                    "request_sha256": request_hash,
                    "status": "success",
                    "response_id": result["response_metadata"]["response_id"],
                })
                print(f"[{position:03d}/156] anonymous preference item complete")
                break
            except (ValueError, RuntimeError, KeyError, json.JSONDecodeError) as error:
                _append_jsonl(attempts_path, {
                    "anonymous_item_id": anonymous_id,
                    "attempt": attempt,
                    "request_sha256": request_hash,
                    "status": "error",
                    "error_type": type(error).__name__,
                })
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Preference item {position}/156 failed after {max_attempts} attempts; "
                        "private partial results remain locked from reporting."
                    ) from error
                time.sleep(min(backoff * attempt, 30))
        else:
            raise RuntimeError(
                f"Preference item {position}/156 exhausted its frozen retry budget; "
                "prepare a new judged run after diagnosing the endpoint."
            )
    _validate_judgements(score_path, packet_path, package)
    if mode == "ollama_api_tags":
        state["runtime_model_identity"]["checks"].append(
            _probe_ollama_digest(
                config["endpoint"], config["model_id"], package["model_digest_sha256"],
                min(timeout, 30), phase="after",
            )
        )
        state_path.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n",
                              encoding="utf-8")
    lock = _lock_preference_judge(args.package_dir, score_path)
    print(json.dumps({"locked_items": lock["item_count"],
                      "judgements_sha256": lock["judgements_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(judge_preference_command())
