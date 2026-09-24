#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT_DEFAULT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"

PROJECT_ROOT="${PROJECT_ROOT_DEFAULT}"
RUNTIME_ROOT="/workspace/phase3"
MODEL_PATH="/workspace/models/Venus-Q-Stage1"
MEMBER_ID="member1"

usage() {
  cat <<'EOF'
Usage: bootstrap.sh [options]

Options:
  --member-id ID        Caller identity (default: member1)
  --project-root PATH   Project checkout (default: detected repository root)
  --workspace-root PATH Shared runtime root (default: /workspace/phase3)
  --model-path PATH     Existing Stage 1 model snapshot
  -h, --help            Show this help
EOF
}

die() {
  printf 'bootstrap.sh: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --member-id) (($# >= 2)) || die "--member-id requires a value"; MEMBER_ID="$2"; shift 2 ;;
    --project-root) (($# >= 2)) || die "--project-root requires a value"; PROJECT_ROOT="$2"; shift 2 ;;
    --workspace-root) (($# >= 2)) || die "--workspace-root requires a value"; RUNTIME_ROOT="$2"; shift 2 ;;
    --model-path) (($# >= 2)) || die "--model-path requires a value"; MODEL_PATH="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${MEMBER_ID}" =~ ^[a-z0-9][a-z0-9_-]*$ ]] || die "invalid member ID: ${MEMBER_ID}"
[[ "${MEMBER_ID}" == "member1" ]] || die "shared environment bootstrap is owned by member1"
[[ "${PROJECT_ROOT}" == /* ]] || die "--project-root must be absolute"
[[ "${RUNTIME_ROOT}" == /* ]] || die "--workspace-root must be absolute"
[[ "${MODEL_PATH}" == /* ]] || die "--model-path must be absolute"
[[ -f "${PROJECT_ROOT}/phase3/configs/source-lock.json" ]] || die "not a Phase 3 project checkout: ${PROJECT_ROOT}"
command -v flock >/dev/null 2>&1 || die "flock is required for shared-workspace coordination"
command -v realpath >/dev/null 2>&1 || die "realpath is required"

PROJECT_ROOT="$(realpath -e -- "${PROJECT_ROOT}")"
RUNTIME_ROOT="$(realpath -m -- "${RUNTIME_ROOT}")"
MODEL_PATH="$(realpath -m -- "${MODEL_PATH}")"

readonly ENV_DIR="${RUNTIME_ROOT}/envs/venus-phase3"
readonly CACHE_DIR="${RUNTIME_ROOT}/cache"
readonly UPSTREAM_ROOT="${RUNTIME_ROOT}/upstream"
readonly MEMBER_ROOT="${RUNTIME_ROOT}/members/${MEMBER_ID}"
readonly BOOTSTRAP_REPORT_DIR="${RUNTIME_ROOT}/reports/${MEMBER_ID}/bootstrap"
readonly LOCK_FILE="${RUNTIME_ROOT}/locks/shared-writer.lock"

mkdir -p -- "${RUNTIME_ROOT}/locks"
chmod 2775 -- "${RUNTIME_ROOT}" "${RUNTIME_ROOT}/locks"
exec 9>>"${LOCK_FILE}"
flock --nonblock 9 || die "shared workspace is busy; another writer holds ${LOCK_FILE}"
printf '%s pid=%s member=%s action=bootstrap\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$$" "${MEMBER_ID}" >&9

mkdir -p -- \
  "${RUNTIME_ROOT}/envs" "${UPSTREAM_ROOT}" "${CACHE_DIR}" \
  "${RUNTIME_ROOT}/data" "${RUNTIME_ROOT}/smoke" \
  "${RUNTIME_ROOT}/members" "${RUNTIME_ROOT}/runs" \
  "${RUNTIME_ROOT}/reports" "${RUNTIME_ROOT}/locks" \
  "${MEMBER_ROOT}" "${RUNTIME_ROOT}/runs/${MEMBER_ID}" "${BOOTSTRAP_REPORT_DIR}"

for team_member in member1 member2 member3 member4 member5 member6; do
  mkdir -p -- \
    "${RUNTIME_ROOT}/members/${team_member}" \
    "${RUNTIME_ROOT}/runs/${team_member}" \
    "${RUNTIME_ROOT}/reports/${team_member}"
done
chmod 2775 -- \
  "${RUNTIME_ROOT}" "${RUNTIME_ROOT}/envs" "${UPSTREAM_ROOT}" "${CACHE_DIR}" \
  "${RUNTIME_ROOT}/data" "${RUNTIME_ROOT}/smoke" \
  "${RUNTIME_ROOT}/members" "${RUNTIME_ROOT}/runs" \
  "${RUNTIME_ROOT}/reports" "${RUNTIME_ROOT}/locks"

bash "${PROJECT_ROOT}/phase3/environment/create_env.sh" \
  --runtime-root "${RUNTIME_ROOT}" \
  --env-dir "${ENV_DIR}" \
  --cache-dir "${CACHE_DIR}"

bash "${PROJECT_ROOT}/phase3/scripts/fetch_upstream.sh" \
  --runtime-root "${RUNTIME_ROOT}"

"${ENV_DIR}/bin/python" "${PROJECT_ROOT}/phase3/tools/preflight.py" \
  --mode static \
  --project-root "${PROJECT_ROOT}" \
  --workspace-root "${RUNTIME_ROOT}" \
  --model-path "${MODEL_PATH}" \
  --upstream-root "${UPSTREAM_ROOT}" \
  --output "${BOOTSTRAP_REPORT_DIR}/preflight-static.json"

"${ENV_DIR}/bin/python" -m pip freeze --all \
  >"${BOOTSTRAP_REPORT_DIR}/requirements.freeze.txt"

(
  cd -- "${BOOTSTRAP_REPORT_DIR}"
  sha256sum preflight-static.json requirements.freeze.txt >checksums.sha256
)

printf 'Bootstrap complete. Static evidence: %s\n' "${BOOTSTRAP_REPORT_DIR}"
printf 'GPU acceptance has not run; use member1_acceptance.sh on the selected 48 GB GPU.\n'
