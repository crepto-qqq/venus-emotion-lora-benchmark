#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT_DEFAULT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"
RUNTIME_ROOT="/workspace/phase3"
MODEL_PATH="/workspace/models/Venus-Q-Stage1"
MEMBER_ID=""
SOURCE_REPORT_DIR=""
OUTPUT_PATH=""

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

die() {
  printf 'verify_handoff.sh: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --member-id) (($# >= 2)) || die "--member-id requires a value"; MEMBER_ID="$2"; shift 2 ;;
    --report-dir) (($# >= 2)) || die "--report-dir requires a value"; SOURCE_REPORT_DIR="$2"; shift 2 ;;
    --project-root) (($# >= 2)) || die "--project-root requires a value"; PROJECT_ROOT="$2"; shift 2 ;;
    --workspace-root) (($# >= 2)) || die "--workspace-root requires a value"; RUNTIME_ROOT="$2"; shift 2 ;;
    --model-path) (($# >= 2)) || die "--model-path requires a value"; MODEL_PATH="$2"; shift 2 ;;
    --output) (($# >= 2)) || die "--output requires a value"; OUTPUT_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${MEMBER_ID}" =~ ^member[1-6]$ ]] || die "--member-id must be member1 through member6"
[[ "${PROJECT_ROOT}" == /* && "${RUNTIME_ROOT}" == /* && "${MODEL_PATH}" == /* ]] || \
  die "project, workspace, and model paths must be absolute"
[[ -n "${SOURCE_REPORT_DIR}" && "${SOURCE_REPORT_DIR}" == /* ]] || die "--report-dir must be absolute"
[[ -d "${SOURCE_REPORT_DIR}" ]] || die "Member 1 report directory not found: ${SOURCE_REPORT_DIR}"
command -v realpath >/dev/null 2>&1 || die "realpath is required"

PROJECT_ROOT="$(realpath -e -- "${PROJECT_ROOT}")"
RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT}")"
MODEL_PATH="$(realpath -m -- "${MODEL_PATH}")"
SOURCE_REPORT_DIR="$(realpath -e -- "${SOURCE_REPORT_DIR}")"
[[ "${RUNTIME_ROOT}" == "/workspace/phase3" ]] || \
  die "formal handoff verification requires --workspace-root /workspace/phase3"
[[ "${MODEL_PATH}" == "/workspace/models/Venus-Q-Stage1" ]] || \
  die "formal handoff verification requires --model-path /workspace/models/Venus-Q-Stage1"

readonly ENV_DIR="${RUNTIME_ROOT}/envs/venus-phase3"
readonly CALLER_REPORT_ROOT="${RUNTIME_ROOT}/reports/${MEMBER_ID}"
readonly CALLER_MEMBER_DIR="${RUNTIME_ROOT}/members/${MEMBER_ID}"
readonly CALLER_RUN_DIR="${RUNTIME_ROOT}/runs/${MEMBER_ID}"
readonly MEMBER1_REPORT_ROOT="${RUNTIME_ROOT}/reports/member1"

[[ "$(dirname -- "${SOURCE_REPORT_DIR}")" == "${MEMBER1_REPORT_ROOT}" ]] || \
  die "--report-dir must be a direct Member 1 attempt directory"
[[ "$(basename -- "${SOURCE_REPORT_DIR}")" =~ ^attempt-[0-9]{3,}$ ]] || \
  die "--report-dir basename must match attempt-NNN"

[[ -x "${ENV_DIR}/bin/python" ]] || die "shared environment is missing"
[[ -f "${SOURCE_REPORT_DIR}/member1-handoff.json" ]] || die "Member 1 aggregate report is missing"
[[ -f "${SOURCE_REPORT_DIR}/checksums.json" ]] || die "Member 1 JSON checksum manifest is missing"
[[ -f "${SOURCE_REPORT_DIR}/checksums.sha256" ]] || die "Member 1 portable checksum list is missing"
(
  cd -- "${SOURCE_REPORT_DIR}"
  sha256sum --check --strict checksums.sha256
) || die "Member 1 report checksum verification failed"

if [[ -z "${OUTPUT_PATH}" ]]; then
  verification_id="verification-$(date -u +%Y%m%dT%H%M%SZ)-$$"
  OUTPUT_PATH="${CALLER_REPORT_ROOT}/${verification_id}/handoff-verification.json"
fi
[[ "${OUTPUT_PATH}" == /* ]] || die "--output must be absolute"
OUTPUT_PATH="$(realpath -m -- "${OUTPUT_PATH}")"
readonly OUTPUT_PARENT="$(dirname -- "${OUTPUT_PATH}")"
[[ "$(dirname -- "${OUTPUT_PARENT}")" == "${CALLER_REPORT_ROOT}" ]] || \
  die "verification output must be in a direct verification directory below ${CALLER_REPORT_ROOT}"
[[ "$(basename -- "${OUTPUT_PARENT}")" =~ ^verification-[A-Za-z0-9._-]+$ ]] || \
  die "verification directory basename must begin with verification-"
[[ "$(basename -- "${OUTPUT_PATH}")" == "handoff-verification.json" ]] || \
  die "verification output filename must be handoff-verification.json"
[[ ! -e "${OUTPUT_PATH}" ]] || die "verification output already exists"

mkdir -p -- "${CALLER_REPORT_ROOT}" "${CALLER_MEMBER_DIR}" "${CALLER_RUN_DIR}" "$(dirname -- "${OUTPUT_PATH}")"

readonly BEFORE_MANIFEST="${CALLER_RUN_DIR}/.member1-report-before.$$.sha256"
readonly AFTER_MANIFEST="${CALLER_RUN_DIR}/.member1-report-after.$$.sha256"
cleanup() {
  rm -f -- "${BEFORE_MANIFEST}" "${AFTER_MANIFEST}"
}
trap cleanup EXIT

(
  cd -- "${SOURCE_REPORT_DIR}"
  find . -type f -print0 | sort -z | xargs -0 -r sha256sum
) >"${BEFORE_MANIFEST}"

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/phase3/tools/verify_handoff.py" \
  --project-root "${PROJECT_ROOT}" \
  --workspace-root "${RUNTIME_ROOT}" \
  --report-dir "${SOURCE_REPORT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --member-id "${MEMBER_ID}" \
  --output "${OUTPUT_PATH}"

(
  cd -- "${SOURCE_REPORT_DIR}"
  find . -type f -print0 | sort -z | xargs -0 -r sha256sum
) >"${AFTER_MANIFEST}"

cmp --silent "${BEFORE_MANIFEST}" "${AFTER_MANIFEST}" || \
  die "Member 1's immutable report changed during verification"

(
  cd -- "$(dirname -- "${OUTPUT_PATH}")"
  sha256sum "$(basename -- "${OUTPUT_PATH}")" >checksums.sha256
)

printf 'Handoff verified without rerunning smoke. Caller-owned result: %s\n' "${OUTPUT_PATH}"
