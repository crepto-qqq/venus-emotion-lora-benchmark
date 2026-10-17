"""Run the frozen Eval80 image-aware judge through a compatible vision endpoint.

Only a loopback endpoint is allowed by default. Remote image transmission needs
an explicit CLI flag after checking the image license and team data policy.
Every call assesses one anonymous response and image, without condition labels.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener

from .core import sha256_file, sha256_text_file
from .quality_benchmark import (
    _HANDBOOK, _SCORE_KEYS, _read_jsonl, _read_object, _validate_judge_config,
    _validate_judgements, lock_guidance_judge_command,
)


def _judge_schema() -> dict[str, Any]:
    dimensions = {key: {"type": "integer", "enum": [0, 1, 2]} for key in _SCORE_KEYS}
    reasons = {key: {"type": "string"} for key in _SCORE_KEYS}
    return {
        "type": "object", "additionalProperties": False,
        "properties": {
            "scores": {"type": "object", "properties": dimensions,
                       "required": list(_SCORE_KEYS), "additionalProperties": False},
            "overall_rationale": {"type": "string"},
            "zero_score_reasons": {"type": "object", "properties": reasons,
                                   "required": list(_SCORE_KEYS), "additionalProperties": False},
            "hallucination": {"type": "boolean"},
            "emotion_conflict": {"type": "boolean"},
            "recommendation_present": {"type": "boolean"},
        },
        "required": ["scores", "overall_rationale", "zero_score_reasons",
                     "hallucination", "emotion_conflict", "recommendation_present"],
    }


def _endpoint_is_allowed(endpoint: str, allow_remote: bool) -> bool:
    parsed = urlparse(endpoint)
    if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.username or parsed.password:
        return False
    if parsed.hostname in ("localhost", "127.0.0.1", "::1"):
        return True
    return allow_remote and parsed.scheme == "https"


def _make_payload(
    packet: dict[str, Any], config: dict[str, Any], handbook: str,
) -> dict[str, Any]:
    image_path = Path(packet["image_path"])
    if sha256_file(image_path) != packet["image_sha256"]:
        raise ValueError("A judge image no longer matches its frozen SHA-256.")
    response = packet["response"]
    if hashlib.sha256(response.encode("utf-8")).hexdigest() != packet["response_sha256"]:
        raise ValueError("A judge answer no longer matches its frozen SHA-256.")
    image_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    image_url = f"data:{image_type};base64,{base64.b64encode(image_path.read_bytes()).decode('ascii')}"
    instruction = (
        "Assess this one anonymous answer against the attached original image and fixed target "
        "emotion. Give five independent integer scores from 0 to 2. For each zero, give a "
        "specific reason in zero_score_reasons; use an empty string for nonzero dimensions. "
        "Do not reward length, headings, style, or the predicted label by itself. If there is "
        "no recommendation, actionability and emotion_preservation must both be zero. "
        "Flag visual hallucinations and advice conflicting with the fixed target. "
        f"\n\nJudge instructions:\n{config['prompt']}"
        f"\n\nFrozen scoring handbook:\n{handbook}"
        f"\n\nFixed target emotion: {packet['target_emotion']}"
        f"\n\nAnonymous model answer:\n{response}"
    )
    decoding = config["decoding"]
    allowed_decoding = {"temperature", "top_p", "max_tokens", "seed"}
    if set(decoding) - allowed_decoding or "max_tokens" not in decoding:
        raise ValueError("Judge decoding needs max_tokens and only supported parameters.")
    if config["image_detail"] not in {"auto", "low", "high"}:
        raise ValueError("OpenAI-compatible image detail must be auto, low, or high.")
    return {
        "model": config["model_id"],
        **decoding,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image_url", "image_url": {
                    "url": image_url, "detail": config["image_detail"],
                }},
            ],
        }],
        "response_format": {"type": "json_schema", "json_schema": {
            "name": "eval80_guidance_judge", "strict": True, "schema": _judge_schema(),
        }},
    }


class _NoRedirectHandler(HTTPRedirectHandler):
    """Never forward a private image or bearer token to a redirect target."""

    def redirect_request(self, request, fp, code, msg, headers, newurl):
        return None


def _post_chat(endpoint: str, payload: dict[str, Any], api_key: str | None, timeout: int) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(endpoint, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    # Loopback traffic must bypass environment HTTP(S) proxies. All redirects
    # are rejected, including same-host redirects, to keep credentials and
    # private image bytes at the one frozen endpoint.
    hostname = urlparse(endpoint).hostname
    handlers = [_NoRedirectHandler()]
    if hostname in ("localhost", "127.0.0.1", "::1"):
        handlers.insert(0, ProxyHandler({}))
    opener = build_opener(*handlers)
    try:
        with opener.open(request, timeout=timeout) as result:
            response = json.loads(result.read().decode("utf-8"))
    except HTTPError as error:
        # Never echo the provider body: some gateways include request metadata.
        raise RuntimeError(f"Judge endpoint HTTP {error.code}") from error
    except URLError as error:
        raise RuntimeError(f"Judge endpoint transport error: {type(error.reason).__name__}") from error
    if not isinstance(response, dict):
        raise ValueError("Judge endpoint returned a non-object response.")
    return response


def _extract_judgement(
    response: dict[str, Any], packet: dict[str, Any], metadata: dict[str, Any],
) -> dict[str, Any]:
    if response.get("model") != metadata["judge_model_id"]:
        raise ValueError("Judge response model differs from the pinned judge model_id.")
    expected_fingerprint = metadata.get("expected_system_fingerprint")
    if expected_fingerprint and response.get("system_fingerprint") != expected_fingerprint:
        raise ValueError("Judge response system fingerprint differs from frozen config.")
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError("Judge response must contain exactly one choice.")
    choice = choices[0]
    if choice.get("finish_reason") != "stop":
        raise ValueError("Judge response did not finish normally.")
    message = choice.get("message")
    if not isinstance(message, dict) or not isinstance(message.get("content"), str):
        raise ValueError("Judge response content is missing.")
    scored = json.loads(message["content"])
    schema_keys = {"scores", "overall_rationale", "zero_score_reasons",
                   "hallucination", "emotion_conflict", "recommendation_present"}
    if not isinstance(scored, dict) or set(scored) != schema_keys:
        raise ValueError("Judge model output differs from the required scoring schema.")
    reasons = scored["zero_score_reasons"]
    scores = scored["scores"]
    if not isinstance(reasons, dict) or set(reasons) != set(_SCORE_KEYS) or not isinstance(scores, dict):
        raise ValueError("Judge model returned invalid score or reason fields.")
    scored["zero_score_reasons"] = {
        key: reasons[key] for key in _SCORE_KEYS if scores.get(key) == 0
    }
    canonical = json.dumps(packet, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    result = {
        "anonymous_item_id": packet["anonymous_item_id"],
        "packet_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "judge_config_sha256": metadata["judge_config_sha256"],
        "judge_model_id": metadata["judge_model_id"],
        **scored,
        "response_metadata": {
            "response_id": response.get("id"),
            "model": response.get("model"),
            "system_fingerprint": response.get("system_fingerprint"),
            "created": response.get("created"),
            "usage": response.get("usage"),
        },
    }
    if not result["response_metadata"]["response_id"]:
        raise ValueError("Judge response ID is required for audit.")
    return result


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _recover_interrupted_score_writes(
    package_dir: Path, metadata: dict[str, Any],
    packets: list[dict[str, Any]], completed: dict[str, dict[str, Any]],
) -> None:
    """Finish a success log entry if a crash followed the durable score write."""
    if not completed:
        return
    attempts_path = package_dir / "judge_attempts_private.jsonl"
    raw_path = package_dir / "judge_raw_responses_private.jsonl"
    attempts = _read_jsonl(attempts_path) if attempts_path.exists() else []
    raw_rows = _read_jsonl(raw_path) if raw_path.exists() else []
    successful_ids = {
        row.get("anonymous_item_id") for row in attempts if row.get("status") == "success"
    }
    packets_by_id = {packet["anonymous_item_id"]: packet for packet in packets}
    for anonymous_id, scored in completed.items():
        if anonymous_id in successful_ids:
            continue
        response_id = scored["response_metadata"].get("response_id")
        matches = [row for row in raw_rows if (
            row.get("anonymous_item_id") == anonymous_id
            and isinstance(row.get("response"), dict)
            and row["response"].get("id") == response_id
        )]
        if len(matches) != 1 or type(matches[0].get("attempt")) is not int:
            raise ValueError("Interrupted judge score has no unique raw response to recover.")
        if _extract_judgement(matches[0]["response"], packets_by_id[anonymous_id], metadata) != scored:
            raise ValueError("Interrupted judge score differs from its raw response.")
        _append_jsonl(attempts_path, {
            "anonymous_item_id": anonymous_id, "attempt": matches[0]["attempt"],
            "status": "success", "response_id": response_id,
        })


def judge_guidance_command(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run 156 anonymous image-aware assessments through a pinned compatible vision endpoint; resume and lock completed scores.")
    parser.add_argument("--package-dir", type=Path, required=True)
    parser.add_argument("--judge-config", type=Path, required=True)
    parser.add_argument("--handbook", type=Path, default=_HANDBOOK)
    parser.add_argument("--api-key-env", default="EVAL80_JUDGE_API_KEY")
    parser.add_argument("--allow-remote-endpoint", action="store_true")
    args = parser.parse_args(argv)
    config = _read_object(args.judge_config)
    _validate_judge_config(config)
    metadata = _read_object(args.package_dir / "judge_package.json")
    if sha256_file(args.judge_config) != metadata["judge_config_sha256"]:
        raise ValueError("Judge config changed after packet preparation.")
    if sha256_text_file(args.handbook) != metadata["handbook_sha256"]:
        raise ValueError("Judge handbook differs from frozen packet preparation.")
    endpoint = config["endpoint"]
    if not _endpoint_is_allowed(endpoint, args.allow_remote_endpoint):
        raise ValueError("Only loopback endpoints are allowed by default; remote HTTPS needs --allow-remote-endpoint.")
    key = os.environ.get(args.api_key_env)
    retry = config["retry_policy"]
    max_attempts = retry.get("max_attempts")
    timeout = retry.get("timeout_seconds")
    backoff = retry.get("backoff_seconds")
    if type(max_attempts) is not int or not 1 <= max_attempts <= 5:
        raise ValueError("retry_policy.max_attempts must be an integer from 1 to 5.")
    if type(timeout) is not int or not 1 <= timeout <= 600:
        raise ValueError("retry_policy.timeout_seconds must be an integer from 1 to 600.")
    if type(backoff) not in (int, float) or not 0 <= backoff <= 30:
        raise ValueError("retry_policy.backoff_seconds must be between 0 and 30.")
    handbook = args.handbook.read_text(encoding="utf-8")
    packet_path = args.package_dir / "judge_items_private.jsonl"
    packets = _read_jsonl(packet_path)
    if len(packets) != 156 or sha256_file(packet_path) != metadata["judge_items_sha256"]:
        raise ValueError("Judge packets changed after preparation.")
    score_path = args.package_dir / "judgements_private.jsonl"
    if (args.package_dir / "judgements_lock.json").exists():
        raise FileExistsError("Judge scores are already locked; use report-guidance.")
    state_path = args.package_dir / "judge_run_state.json"
    state = {
        "schema_version": 1,
        "judge_package_sha256": sha256_file(args.package_dir / "judge_package.json"),
        "judge_config_sha256": metadata["judge_config_sha256"],
        "judge_items_sha256": metadata["judge_items_sha256"],
        "handbook_sha256": metadata["handbook_sha256"],
    }
    if state_path.exists():
        if _read_object(state_path) != state:
            raise ValueError("Resumed judge run has different frozen inputs.")
    else:
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    completed = (
        _validate_judgements(score_path, packet_path, metadata, require_complete=False)
        if score_path.exists() else {}
    )
    attempts_path = args.package_dir / "judge_attempts_private.jsonl"
    raw_path = args.package_dir / "judge_raw_responses_private.jsonl"
    _recover_interrupted_score_writes(args.package_dir, metadata, packets, completed)
    for position, packet in enumerate(packets, start=1):
        anonymous_id = packet["anonymous_item_id"]
        if anonymous_id in completed:
            continue
        payload = _make_payload(packet, config, handbook)
        request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        for attempt in range(1, max_attempts + 1):
            try:
                raw_response = _post_chat(endpoint, payload, key, timeout)
            except (ValueError, RuntimeError, OSError, json.JSONDecodeError, KeyError) as error:
                _append_jsonl(attempts_path, {
                    "anonymous_item_id": anonymous_id, "attempt": attempt, "status": "error",
                    "error_type": type(error).__name__,
                })
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Anonymous judge item {position}/156 failed after {max_attempts} attempts; "
                        "partial results remain private and resumable."
                    ) from error
                time.sleep(min(backoff * attempt, 30))
                continue
            _append_jsonl(raw_path, {
                "anonymous_item_id": anonymous_id,
                "attempt": attempt,
                "request_sha256": request_hash,
                "response": raw_response,
            })
            try:
                result = _extract_judgement(raw_response, packet, metadata)
                # Validate the single normalized row before persisting it.
                temporary = args.package_dir / "judge_single_validation.tmp"
                temporary.write_text(json.dumps(result) + "\n", encoding="utf-8")
                try:
                    _validate_judgements(temporary, packet_path, metadata, require_complete=False)
                finally:
                    temporary.unlink(missing_ok=True)
            except (ValueError, RuntimeError, OSError, json.JSONDecodeError, KeyError) as error:
                _append_jsonl(attempts_path, {
                    "anonymous_item_id": anonymous_id, "attempt": attempt, "status": "error",
                    "error_type": type(error).__name__,
                })
                if attempt == max_attempts:
                    raise RuntimeError(
                        f"Anonymous judge item {position}/156 failed after {max_attempts} attempts; "
                        "partial results remain private and resumable."
                    ) from error
                time.sleep(min(backoff * attempt, 30))
                continue
            # Persistence errors must stop the run: retrying here could append a
            # second score. Resume repairs the score/success-log crash window.
            _append_jsonl(score_path, result)
            _append_jsonl(attempts_path, {
                "anonymous_item_id": anonymous_id, "attempt": attempt, "status": "success",
                "response_id": result["response_metadata"]["response_id"],
            })
            print(f"[{position:03d}/156] anonymous judge item complete")
            break
    _validate_judgements(score_path, packet_path, metadata)
    return lock_guidance_judge_command([
        "--package-dir", str(args.package_dir), "--judgements", str(score_path)
    ])


if __name__ == "__main__":
    raise SystemExit(judge_guidance_command())
