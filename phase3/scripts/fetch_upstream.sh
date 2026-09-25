#!/usr/bin/env bash
set -Eeuo pipefail
umask 0002

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd -P)"
readonly QWEN_REPOSITORY="https://github.com/cognitedata/Qwen-VL-finetune.git"
readonly QWEN_COMMIT="efa37ba284d56192b246d9b4ed5d3668c1abd163"
readonly VENUS_REPOSITORY="https://github.com/PKU-ICST-MIPL/Venus_CVPR2026.git"
readonly VENUS_COMMIT="44e2971b7fe41f88a88c7e0ed8bf4c85eeb0155f"
readonly PATCH_RELATIVE_PATH="phase3/upstream/patches/qwen-vl-finetune-efa37ba-phase3.patch"
readonly PATCH_SHA256="be9e14a60f4f2189a108aa1f46a7384616bfdd527ca6c0a41a4c175a92172eba"

RUNTIME_ROOT="/workspace/phase3"

usage() {
  cat <<'EOF'
Usage: fetch_upstream.sh [--runtime-root PATH]

Fetch the two exact upstream source revisions and apply the reviewed Qwen patch.
EOF
}

die() {
  printf 'fetch_upstream.sh: %s\n' "$*" >&2
  exit 1
}

while (($#)); do
  case "$1" in
    --runtime-root)
      (($# >= 2)) || die "--runtime-root requires a value"
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *) die "unknown argument: $1" ;;
  esac
done

[[ "${RUNTIME_ROOT}" == /* ]] || die "--runtime-root must be an absolute path"
command -v git >/dev/null 2>&1 || die "git is required"
command -v sha256sum >/dev/null 2>&1 || die "sha256sum is required"

readonly UPSTREAM_ROOT="${RUNTIME_ROOT}/upstream"
readonly QWEN_DIR="${UPSTREAM_ROOT}/Qwen-VL-finetune"
readonly VENUS_DIR="${UPSTREAM_ROOT}/Venus_CVPR2026"
readonly PATCH_FILE="${PROJECT_ROOT}/${PATCH_RELATIVE_PATH}"

[[ -f "${PATCH_FILE}" ]] || die "missing reviewed patch: ${PATCH_FILE}"
observed_patch_sha="$(sha256sum "${PATCH_FILE}" | awk '{print $1}')"
[[ "${observed_patch_sha}" == "${PATCH_SHA256}" ]] || \
  die "reviewed patch checksum mismatch: ${observed_patch_sha}"

mkdir -p -- "${UPSTREAM_ROOT}"

clone_exact() {
  local repository="$1"
  local commit="$2"
  local destination="$3"
  local label="$4"
  local temporary="${UPSTREAM_ROOT}/.${label}.clone.$$"

  if [[ -e "${destination}" ]]; then
    [[ -d "${destination}/.git" ]] || die "existing path is not a Git checkout: ${destination}"
    return 0
  fi

  [[ "${temporary}" == "${UPSTREAM_ROOT}/."*".clone."* ]] || die "unsafe temporary path"
  rm -rf -- "${temporary}"
  git init --quiet "${temporary}"
  git -C "${temporary}" remote add origin "${repository}"
  git -C "${temporary}" fetch --quiet --depth 1 origin "${commit}"
  git -C "${temporary}" checkout --quiet --detach FETCH_HEAD
  [[ "$(git -C "${temporary}" rev-parse HEAD)" == "${commit}" ]] || die "failed to fetch ${label} at ${commit}"
  mv -- "${temporary}" "${destination}"
}

clone_exact "${QWEN_REPOSITORY}" "${QWEN_COMMIT}" "${QWEN_DIR}" "qwen"
clone_exact "${VENUS_REPOSITORY}" "${VENUS_COMMIT}" "${VENUS_DIR}" "venus"

[[ "$(git -C "${QWEN_DIR}" rev-parse HEAD)" == "${QWEN_COMMIT}" ]] || \
  die "Qwen checkout is not at the pinned commit"
[[ "$(git -C "${VENUS_DIR}" rev-parse HEAD)" == "${VENUS_COMMIT}" ]] || \
  die "Venus checkout is not at the pinned commit"
[[ "$(git -C "${QWEN_DIR}" remote get-url origin)" == "${QWEN_REPOSITORY}" ]] || \
  die "Qwen origin does not match the pinned repository"
[[ "$(git -C "${VENUS_DIR}" remote get-url origin)" == "${VENUS_REPOSITORY}" ]] || \
  die "Venus origin does not match the pinned repository"

if git -C "${QWEN_DIR}" apply --check "${PATCH_FILE}" 2>/dev/null; then
  git -C "${QWEN_DIR}" apply "${PATCH_FILE}"
elif ! git -C "${QWEN_DIR}" apply --reverse --check "${PATCH_FILE}" 2>/dev/null; then
  die "Qwen checkout is neither clean-patchable nor already patched"
fi

git -C "${QWEN_DIR}" diff --check
if ! cmp -s <(git -C "${QWEN_DIR}" diff --binary --no-ext-diff HEAD -- finetune.py) "${PATCH_FILE}"; then
  die "Qwen working-tree change does not exactly match the reviewed patch"
fi

qwen_status="$(git -C "${QWEN_DIR}" status --porcelain --untracked-files=all)"
[[ "${qwen_status}" == " M finetune.py" ]] || \
  die "unexpected Qwen working-tree content: ${qwen_status:-<clean>}"

venus_status="$(git -C "${VENUS_DIR}" status --porcelain --untracked-files=all)"
[[ -z "${venus_status}" ]] || die "Venus checkout contains unreviewed changes"

printf 'Pinned upstream sources ready under %s\n' "${UPSTREAM_ROOT}"
printf '  Qwen-VL-finetune: %s + reviewed patch %s\n' "${QWEN_COMMIT}" "${PATCH_SHA256}"
printf '  Venus_CVPR2026:   %s\n' "${VENUS_COMMIT}"
