#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT_DEFAULT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly FAILURE_REPORT_TOOL="${SCRIPT_DIR}/../tools/verify_handoff.py"

PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"
RUNTIME_ROOT="/workspace/phase3"
MODEL_PATH="/workspace/models/Venus-Q-Stage1"
MEMBER_ID=""
SOURCE_REPORT_DIR=""
OUTPUT_PATH=""
OUTPUT_PARENT=""
CALLER_REPORT_ROOT=""

usage() {
  cat <<'EOF'
Usage: verify_handoff.sh --member-id ID --report-dir PATH [options]

Validate an existing Member 1 handoff without rerunning the GPU smoke. The
verification result is written under the caller's own report directory.

Options:
  --member-id ID        Calling team member (required)
  --report-dir PATH     Existing Member 1 attempt directory (required)
  --project-root PATH   Project checkout (default: detected repository root)
  --workspace-root PATH Shared runtime root (default: /workspace/phase3)
  --model-path PATH     Existing Stage 1 model snapshot
  --output PATH         New caller-owned JSON path (default: timestamped)
  -h, --help            Show this help
EOF
}

die_without_report() {
  printf 'verify_handoff.sh: %s\n' "$1" >&2
  exit 1
}

seal_output() {
  (
    cd -- "${OUTPUT_PARENT}"
    sha256sum handoff-verification.json >checksums.sha256
  )
  chmod a-w -- "${OUTPUT_PATH}" "${OUTPUT_PARENT}/checksums.sha256"
  chmod a-w -- "${OUTPUT_PARENT}"
}

emit_failure() {
  local failure_code="$1"
  local preserve_existing="${2:-0}"
  local reporter_python=""

  if [[ -x "${RUNTIME_ROOT}/envs/venus-phase3/bin/python" ]]; then
    reporter_python="${RUNTIME_ROOT}/envs/venus-phase3/bin/python"
  elif command -v python3 >/dev/null 2>&1; then
    reporter_python="$(command -v python3)"
  else
    die_without_report "${failure_code} (python3 unavailable; failure evidence could not be written)"
  fi

  mkdir -p -- "${CALLER_REPORT_ROOT}" "${RUNTIME_ROOT}/members/${MEMBER_ID}" \
    "${RUNTIME_ROOT}/runs/${MEMBER_ID}"
  if [[ -e "${OUTPUT_PARENT}" ]]; then
    if [[ "${preserve_existing}" != "1" ]]; then
      die_without_report "${failure_code} (verification output directory already exists)"
    fi
    chmod u+w -- "${OUTPUT_PARENT}" 2>/dev/null || true
    chmod u+w -- "${OUTPUT_PATH}" "${OUTPUT_PARENT}/checksums.sha256" 2>/dev/null || true
  else
    mkdir -- "${OUTPUT_PARENT}"
  fi

  failure_args=(
    failure-report
    --runtime-root "${RUNTIME_ROOT}"
    --report-dir "${SOURCE_REPORT_DIR:-<unavailable>}"
    --member-id "${MEMBER_ID}"
    --failure-code "${failure_code}"
    --output "${OUTPUT_PATH}"
  )
  if [[ "${preserve_existing}" == "1" && -f "${OUTPUT_PATH}" ]]; then
    failure_args+=(--preserve-checks-from "${OUTPUT_PATH}")
  fi
  "${reporter_python}" "${FAILURE_REPORT_TOOL}" "${failure_args[@]}" >/dev/null
  seal_output
  printf 'Handoff verification failed (%s). Sealed caller-owned evidence: %s\n' \
    "${failure_code}" "${OUTPUT_PARENT}" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --member-id) (($# >= 2)) || die_without_report "--member-id requires a value"; MEMBER_ID="$2"; shift 2 ;;
    --report-dir) (($# >= 2)) || die_without_report "--report-dir requires a value"; SOURCE_REPORT_DIR="$2"; shift 2 ;;
    --project-root) (($# >= 2)) || die_without_report "--project-root requires a value"; PROJECT_ROOT="$2"; shift 2 ;;
    --workspace-root) (($# >= 2)) || die_without_report "--workspace-root requires a value"; RUNTIME_ROOT="$2"; shift 2 ;;
    --model-path) (($# >= 2)) || die_without_report "--model-path requires a value"; MODEL_PATH="$2"; shift 2 ;;
    --output) (($# >= 2)) || die_without_report "--output requires a value"; OUTPUT_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die_without_report "unknown argument" ;;
  esac
done

[[ "${MEMBER_ID}" =~ ^member[1-6]$ ]] || die_without_report "--member-id must be member1 through member6"
[[ "${RUNTIME_ROOT}" == /* ]] || die_without_report "--workspace-root must be absolute"
command -v realpath >/dev/null 2>&1 || die_without_report "realpath is required"

RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT}")"
readonly ENV_DIR="${RUNTIME_ROOT}/envs/venus-phase3"
CALLER_REPORT_ROOT="${RUNTIME_ROOT}/reports/${MEMBER_ID}"
readonly CALLER_MEMBER_DIR="${RUNTIME_ROOT}/members/${MEMBER_ID}"
readonly CALLER_RUN_DIR="${RUNTIME_ROOT}/runs/${MEMBER_ID}"
readonly MEMBER1_REPORT_ROOT="${RUNTIME_ROOT}/reports/member1"

if [[ -z "${OUTPUT_PATH}" ]]; then
  verification_id="verification-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  OUTPUT_PATH="${CALLER_REPORT_ROOT}/${verification_id}/handoff-verification.json"
fi
[[ "${OUTPUT_PATH}" == /* ]] || die_without_report "--output must be absolute"
OUTPUT_PATH="$(realpath -m -- "${OUTPUT_PATH}")"
OUTPUT_PARENT="$(dirname -- "${OUTPUT_PATH}")"
[[ "$(dirname -- "${OUTPUT_PARENT}")" == "${CALLER_REPORT_ROOT}" ]] || \
  die_without_report "verification output must be in a direct caller verification directory"
[[ "$(basename -- "${OUTPUT_PARENT}")" =~ ^verification-[A-Za-z0-9._-]+$ ]] || \
  die_without_report "verification directory basename must begin with verification-"
[[ "$(basename -- "${OUTPUT_PATH}")" == "handoff-verification.json" ]] || \
  die_without_report "verification output filename must be handoff-verification.json"
[[ ! -e "${OUTPUT_PARENT}" ]] || die_without_report "verification output directory already exists"

[[ "${RUNTIME_ROOT}" == "/workspace/phase3" ]] || emit_failure "runtime_root_invalid"
[[ "${PROJECT_ROOT}" == /* ]] || emit_failure "project_root_invalid"
[[ "${MODEL_PATH}" == /* ]] || emit_failure "model_path_invalid"
[[ -n "${SOURCE_REPORT_DIR}" && "${SOURCE_REPORT_DIR}" == /* ]] || emit_failure "source_bundle_location_invalid"
[[ ! -L "${SOURCE_REPORT_DIR}" && -d "${SOURCE_REPORT_DIR}" ]] || emit_failure "source_bundle_type_invalid"
[[ -d "${PROJECT_ROOT}" ]] || emit_failure "project_root_invalid"

PROJECT_ROOT="$(realpath -e -- "${PROJECT_ROOT}")"
MODEL_PATH="$(realpath -m -- "${MODEL_PATH}")"
SOURCE_REPORT_DIR="$(realpath -e -- "${SOURCE_REPORT_DIR}")"
[[ "${MODEL_PATH}" == "/workspace/models/Venus-Q-Stage1" ]] || emit_failure "model_path_invalid"
[[ "$(dirname -- "${SOURCE_REPORT_DIR}")" == "${MEMBER1_REPORT_ROOT}" ]] || \
  emit_failure "source_bundle_location_invalid"
[[ "$(basename -- "${SOURCE_REPORT_DIR}")" =~ ^attempt-[0-9]{3,}$ ]] || \
  emit_failure "source_bundle_location_invalid"
[[ -x "${ENV_DIR}/bin/python" ]] || emit_failure "shared_environment_missing"
[[ -f "${SOURCE_REPORT_DIR}/member1-handoff.json" ]] || emit_failure "source_reports_invalid"
[[ -f "${SOURCE_REPORT_DIR}/checksums.json" ]] || emit_failure "checksum_manifest_invalid"
[[ -f "${SOURCE_REPORT_DIR}/checksums.sha256" ]] || emit_failure "checksum_manifest_invalid"

mkdir -p -- "${CALLER_REPORT_ROOT}" "${CALLER_MEMBER_DIR}" "${CALLER_RUN_DIR}"
mkdir -- "${OUTPUT_PARENT}"

readonly BEFORE_MANIFEST="${CALLER_RUN_DIR}/.member1-report-before.$$.sha256"
readonly AFTER_MANIFEST="${CALLER_RUN_DIR}/.member1-report-after.$$.sha256"
cleanup() {
  rm -f -- "${BEFORE_MANIFEST}" "${AFTER_MANIFEST}"
}
trap cleanup EXIT

snapshot_source() {
  local destination="$1"
  (
    cd -- "${SOURCE_REPORT_DIR}"
    find . -mindepth 1 -maxdepth 1 -printf '%P\t%y\t%m\n' | LC_ALL=C sort
    find . -mindepth 1 -maxdepth 1 -type f -print0 | LC_ALL=C sort -z | xargs -0 -r sha256sum
  ) >"${destination}"
}

if ! snapshot_source "${BEFORE_MANIFEST}"; then
  emit_failure "source_snapshot_failed" 1
fi
set +e
"${ENV_DIR}/bin/python" "${FAILURE_REPORT_TOOL}" \
  --project-root "${PROJECT_ROOT}" \
  --workspace-root "${RUNTIME_ROOT}" \
  --report-dir "${SOURCE_REPORT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --member-id "${MEMBER_ID}" \
  --output "${OUTPUT_PATH}"
verification_status=$?
set -e
if ! snapshot_source "${AFTER_MANIFEST}"; then
  emit_failure "source_snapshot_failed" 1
fi

if ! cmp --silent "${BEFORE_MANIFEST}" "${AFTER_MANIFEST}"; then
  emit_failure "source_bundle_changed" 1
fi
if [[ ! -f "${OUTPUT_PATH}" ]]; then
  emit_failure "verifier_output_missing" 1
fi

seal_output
if ((verification_status != 0)); then
  printf 'Handoff verification failed. Sealed caller-owned evidence: %s\n' "${OUTPUT_PARENT}" >&2
  exit "${verification_status}"
fi

printf 'Handoff verified without rerunning smoke. Caller-owned result: %s\n' "${OUTPUT_PATH}"
