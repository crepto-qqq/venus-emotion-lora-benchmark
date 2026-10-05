#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly REQUIREMENTS_FILE="${SCRIPT_DIR}/requirements.txt"
readonly CONSTRAINTS_FILE="${SCRIPT_DIR}/constraints-runpod-cu118.txt"
readonly EXPECTED_PYTHON_VERSION="3.10.13"
readonly EXPECTED_PIP_VERSION="23.2.1"
readonly MINIFORGE_RELEASE="24.7.1-2"
readonly MINIFORGE_INSTALLER="Miniforge3-${MINIFORGE_RELEASE}-Linux-x86_64.sh"

RUNTIME_ROOT="/workspace/phase3"
ENV_DIR=""
CACHE_DIR=""

usage() {
  cat <<'EOF'
Usage: create_env.sh [options]

Create or verify the shared, pinned Phase 3 Python environment.

Options:
  --runtime-root PATH  Shared Phase 3 runtime root (default: /workspace/phase3)
  --env-dir PATH       Environment prefix (default: <runtime-root>/envs/venus-phase3)
  --cache-dir PATH     Package/cache directory (default: <runtime-root>/cache)
  -h, --help           Show this help
EOF
}

die() {
  printf 'create_env.sh: %s\n' "$*" >&2
  exit 1
}

require_absolute_path() {
  [[ "$2" == /* ]] || die "$1 must be an absolute path: $2"
}

while (($#)); do
  case "$1" in
    --runtime-root)
      (($# >= 2)) || die "--runtime-root requires a value"
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --env-dir)
      (($# >= 2)) || die "--env-dir requires a value"
      ENV_DIR="$2"
      shift 2
      ;;
    --cache-dir)
      (($# >= 2)) || die "--cache-dir requires a value"
      CACHE_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

ENV_DIR="${ENV_DIR:-${RUNTIME_ROOT}/envs/venus-phase3}"
CACHE_DIR="${CACHE_DIR:-${RUNTIME_ROOT}/cache}"
require_absolute_path "--runtime-root" "${RUNTIME_ROOT}"
require_absolute_path "--env-dir" "${ENV_DIR}"
require_absolute_path "--cache-dir" "${CACHE_DIR}"

[[ -f "${REQUIREMENTS_FILE}" ]] || die "missing ${REQUIREMENTS_FILE}"
[[ -f "${CONSTRAINTS_FILE}" ]] || die "missing ${CONSTRAINTS_FILE}"
[[ "$(uname -s)" == "Linux" ]] || die "the pinned GPU environment must be created on Linux"

mkdir -p -- "${RUNTIME_ROOT}/envs" "${RUNTIME_ROOT}/tools" "${CACHE_DIR}/pip" "${CACHE_DIR}/conda"

find_conda() {
  local candidate
  for candidate in \
    "${CONDA_EXE:-}" \
    "/workspace/miniforge3/bin/conda" \
    "/workspace/miniconda3/bin/conda" \
    "/opt/conda/bin/conda" \
    "${RUNTIME_ROOT}/tools/miniforge3/bin/conda"; do
    if [[ -n "${candidate}" && -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done
  command -v conda 2>/dev/null || return 1
}

install_miniforge() {
  local install_dir="${RUNTIME_ROOT}/tools/miniforge3"
  local download_dir="${RUNTIME_ROOT}/tools/downloads"
  local installer="${download_dir}/${MINIFORGE_INSTALLER}"
  local checksum_file="${installer}.sha256"
  local base_url="https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_RELEASE}"

  [[ "$(uname -m)" == "x86_64" ]] || die "automatic Miniforge installation supports x86_64 only"
  command -v curl >/dev/null 2>&1 || die "curl is required to install Miniforge"
  command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required to verify Miniforge"
  [[ ! -e "${install_dir}" ]] || die "incomplete Miniforge path exists: ${install_dir}"

  mkdir -p -- "${download_dir}"
  curl --fail --location --retry 3 --output "${installer}" \
    "${base_url}/${MINIFORGE_INSTALLER}"
  curl --fail --location --retry 3 --output "${checksum_file}" \
    "${base_url}/${MINIFORGE_INSTALLER}.sha256"
  local expected_sha actual_sha
  expected_sha="$(awk 'NR == 1 {print $1}' "${checksum_file}")"
  actual_sha="$(sha256sum "${installer}" | awk '{print $1}')"
  [[ "${expected_sha}" =~ ^[0-9a-fA-F]{64}$ ]] || die "invalid Miniforge checksum file"
  [[ "${actual_sha,,}" == "${expected_sha,,}" ]] || die "Miniforge checksum verification failed"
  # install_miniforge is called through command substitution so its stdout must
  # contain only the final conda path.  Send installer progress to stderr.
  bash "${installer}" -b -p "${install_dir}" >&2
  [[ -x "${install_dir}/bin/conda" ]] || die "Miniforge installation did not create conda"
  printf '%s\n' "${install_dir}/bin/conda"
}

CONDA_BIN="$(find_conda || true)"
if [[ -z "${CONDA_BIN}" ]]; then
  CONDA_BIN="$(install_miniforge)"
fi

export CONDA_PKGS_DIRS="${CACHE_DIR}/conda"
export PIP_CACHE_DIR="${CACHE_DIR}/pip"
export PYTHONNOUSERSITE=1

if [[ -e "${ENV_DIR}" && ! -x "${ENV_DIR}/bin/python" ]]; then
  die "environment path exists but is incomplete; inspect it before retrying: ${ENV_DIR}"
fi

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
  "${CONDA_BIN}" create --yes --prefix "${ENV_DIR}" \
    "python=${EXPECTED_PYTHON_VERSION}" "pip=${EXPECTED_PIP_VERSION}"
fi

actual_python="$(${ENV_DIR}/bin/python -c 'import platform; print(platform.python_version())')"
[[ "${actual_python}" == "${EXPECTED_PYTHON_VERSION}" ]] || \
  die "expected Python ${EXPECTED_PYTHON_VERSION}, found ${actual_python} in ${ENV_DIR}"

"${ENV_DIR}/bin/python" -m pip install --disable-pip-version-check \
  "pip==${EXPECTED_PIP_VERSION}"

DS_BUILD_OPS=0 "${ENV_DIR}/bin/python" -m pip install \
  --requirement "${REQUIREMENTS_FILE}" \
  --constraint "${CONSTRAINTS_FILE}"

"${ENV_DIR}/bin/python" -m pip check

actual_pip="$(${ENV_DIR}/bin/python -m pip --version | awk '{print $2}')"
[[ "${actual_pip}" == "${EXPECTED_PIP_VERSION}" ]] || \
  die "expected pip ${EXPECTED_PIP_VERSION}, found ${actual_pip}"

printf 'Phase 3 environment ready: %s (Python %s, pip %s)\n' \
  "${ENV_DIR}" "${actual_python}" "${actual_pip}"
