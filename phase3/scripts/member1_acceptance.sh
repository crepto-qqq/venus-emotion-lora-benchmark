#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT_DEFAULT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"
RUNTIME_ROOT="/workspace/phase3"
MODEL_PATH="/workspace/models/Venus-Q-Stage1"
MEMBER_ID="member1"
IMAGE_PATH=""
REPORT_DIR=""

usage() {
  cat <<'EOF'
Usage: member1_acceptance.sh --image PATH --report-dir PATH [options]

Run the one-image BF16 forward/backward infrastructure acceptance. This command
does not perform an optimizer step and does not save an adapter or checkpoint.

Options:
  --member-id ID        Must be member1 (default: member1)
  --project-root PATH   Project checkout (default: detected repository root)
  --workspace-root PATH Shared runtime root (default: /workspace/phase3)
  --model-path PATH     Existing Stage 1 model snapshot
  --image PATH          Real contentment_05000 image (required)
  --report-dir PATH     New immutable attempt directory (required)
  -h, --help            Show this help
EOF
}

die() {
  printf 'member1_acceptance.sh: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --member-id) (($# >= 2)) || die "--member-id requires a value"; MEMBER_ID="$2"; shift 2 ;;
    --project-root) (($# >= 2)) || die "--project-root requires a value"; PROJECT_ROOT="$2"; shift 2 ;;
    --workspace-root) (($# >= 2)) || die "--workspace-root requires a value"; RUNTIME_ROOT="$2"; shift 2 ;;
    --model-path) (($# >= 2)) || die "--model-path requires a value"; MODEL_PATH="$2"; shift 2 ;;
    --image) (($# >= 2)) || die "--image requires a value"; IMAGE_PATH="$2"; shift 2 ;;
    --report-dir) (($# >= 2)) || die "--report-dir requires a value"; REPORT_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${MEMBER_ID}" == "member1" ]] || die "technical acceptance is owned by member1"
[[ "${PROJECT_ROOT}" == /* && "${RUNTIME_ROOT}" == /* && "${MODEL_PATH}" == /* ]] || \
  die "project, workspace, and model paths must be absolute"
[[ -n "${IMAGE_PATH}" && "${IMAGE_PATH}" == /* ]] || die "--image must be an absolute path"
[[ -n "${REPORT_DIR}" && "${REPORT_DIR}" == /* ]] || die "--report-dir must be an absolute path"
[[ -f "${IMAGE_PATH}" ]] || die "image not found: ${IMAGE_PATH}"
[[ -f "${PROJECT_ROOT}/phase3/configs/member1-smoke.json" ]] || die "Phase 3 config is missing"
command -v flock >/dev/null 2>&1 || die "flock is required"
command -v realpath >/dev/null 2>&1 || die "realpath is required"

PROJECT_ROOT="$(realpath -e -- "${PROJECT_ROOT}")"
RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT}")"
MODEL_PATH="$(realpath -m -- "${MODEL_PATH}")"
IMAGE_PATH="$(realpath -e -- "${IMAGE_PATH}")"
REPORT_DIR="$(realpath -m -- "${REPORT_DIR}")"
[[ "${RUNTIME_ROOT}" == "/workspace/phase3" ]] || \
  die "formal acceptance requires --workspace-root /workspace/phase3"
[[ "${MODEL_PATH}" == "/workspace/models/Venus-Q-Stage1" ]] || \
  die "formal acceptance requires --model-path /workspace/models/Venus-Q-Stage1"

readonly ENV_DIR="${RUNTIME_ROOT}/envs/venus-phase3"
readonly UPSTREAM_ROOT="${RUNTIME_ROOT}/upstream"
readonly QWEN_DIR="${UPSTREAM_ROOT}/Qwen-VL-finetune"
readonly LOCK_FILE="${RUNTIME_ROOT}/locks/shared-writer.lock"
readonly EXPECTED_REPORT_PARENT="${RUNTIME_ROOT}/reports/member1"
readonly FIXTURE_DIR="${RUNTIME_ROOT}/smoke"
readonly FIXTURE_PATH="${FIXTURE_DIR}/contentment_05000.json"
readonly FIXTURE_MANIFEST="${FIXTURE_DIR}/contentment_05000.manifest.json"

[[ -x "${ENV_DIR}/bin/python" ]] || die "environment is missing; run bootstrap.sh first"
[[ -d "${QWEN_DIR}/.git" ]] || die "pinned Qwen checkout is missing; run bootstrap.sh first"
[[ "$(dirname -- "${REPORT_DIR}")" == "${EXPECTED_REPORT_PARENT}" ]] || \
  die "--report-dir must be a direct child of ${EXPECTED_REPORT_PARENT}"
[[ "$(basename -- "${REPORT_DIR}")" =~ ^attempt-[0-9]{3,}$ ]] || \
  die "--report-dir basename must match attempt-NNN (for example, attempt-001)"
[[ ! -e "${REPORT_DIR}" ]] || die "report attempt already exists; choose a new attempt directory"

mkdir -p -- "${RUNTIME_ROOT}/locks"

exec 9>>"${LOCK_FILE}"
flock --nonblock 9 || die "shared workspace is busy; another writer holds ${LOCK_FILE}"
printf '%s pid=%s member=%s action=acceptance\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "${MEMBER_ID}" >&9
[[ ! -e "${REPORT_DIR}" ]] || die "report attempt was created by another process"
mkdir -p -- "${EXPECTED_REPORT_PARENT}" "${FIXTURE_DIR}"
mkdir -- "${REPORT_DIR}"

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/phase3/tools/preflight.py" \
  --mode runtime \
  --project-root "${PROJECT_ROOT}" \
  --workspace-root "${RUNTIME_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --upstream-root "${UPSTREAM_ROOT}" \
  --output "${REPORT_DIR}/preflight-runtime.json"

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/phase3/tools/build_smoke_fixture.py" \
  --project-root "${PROJECT_ROOT}" \
  --image "${IMAGE_PATH}" \
  --output "${FIXTURE_PATH}" \
  --manifest "${FIXTURE_MANIFEST}"

cp -- "${FIXTURE_PATH}" "${REPORT_DIR}/technical-fixture.json"
cp -- "${FIXTURE_MANIFEST}" "${REPORT_DIR}/technical-fixture-manifest.json"
LC_ALL=C "${ENV_DIR}/bin/python" -m pip freeze --all \
  | sed '/^[[:space:]]*$/d' \
  | LC_ALL=C sort \
  >"${REPORT_DIR}/environment.freeze.txt"

{
  printf '%s  %s\n' "$(sha256sum "${REPORT_DIR}/technical-fixture.json" | awk '{print $1}')" \
    'technical-fixture.json'
  printf '%s  %s\n' "$(sha256sum "${REPORT_DIR}/technical-fixture-manifest.json" | awk '{print $1}')" \
    'technical-fixture-manifest.json'
} >"${REPORT_DIR}/fixture-artifacts.sha256"

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/phase3/tools/smoke_backward.py" \
  --fixture "${FIXTURE_PATH}" \
  --model-dir "${MODEL_PATH}" \
  --upstream-dir "${QWEN_DIR}" \
  --config "${PROJECT_ROOT}/phase3/configs/member1-smoke.json" \
  --source-lock "${PROJECT_ROOT}/phase3/configs/source-lock.json" \
  --report "${REPORT_DIR}/smoke-backward.json" \
  --member-id "${MEMBER_ID}" \
  --cache-dir "${RUNTIME_ROOT}/cache"

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/phase3/tools/report.py" aggregate \
  --preflight "${REPORT_DIR}/preflight-runtime.json" \
  --smoke "${REPORT_DIR}/smoke-backward.json" \
  --output "${REPORT_DIR}/member1-handoff.json" \
  --project-root "${PROJECT_ROOT}" \
  --fixture-manifest "${REPORT_DIR}/technical-fixture-manifest.json" \
  --environment-freeze "${REPORT_DIR}/environment.freeze.txt" \
  --member-id "${MEMBER_ID}" \
  --multi-user-status coordination_required

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/phase3/tools/report.py" checksums \
  --input "${REPORT_DIR}/preflight-runtime.json" \
  --input "${REPORT_DIR}/smoke-backward.json" \
  --input "${REPORT_DIR}/member1-handoff.json" \
  --input "${REPORT_DIR}/technical-fixture.json" \
  --input "${REPORT_DIR}/technical-fixture-manifest.json" \
  --input "${REPORT_DIR}/environment.freeze.txt" \
  --input "${REPORT_DIR}/fixture-artifacts.sha256" \
  --output "${REPORT_DIR}/checksums.json"

(
  cd -- "${REPORT_DIR}"
  sha256sum preflight-runtime.json smoke-backward.json member1-handoff.json \
    technical-fixture.json technical-fixture-manifest.json environment.freeze.txt \
    fixture-artifacts.sha256 checksums.json \
    >checksums.sha256
)

chmod -R a-w -- "${REPORT_DIR}"
printf 'Member 1 technical acceptance passed. Immutable evidence: %s\n' "${REPORT_DIR}"
printf 'This result does not authorize full-data training or formal hyperparameters.\n'
