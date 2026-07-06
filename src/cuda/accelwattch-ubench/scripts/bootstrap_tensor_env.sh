#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "${script_dir}/.." && pwd)"

if [[ -n "${CUDA_HOME:-}" ]]; then
  cuda_home="${CUDA_HOME}"
elif [[ -n "${CUDA_PATH:-}" ]]; then
  cuda_home="${CUDA_PATH}"
elif command -v nvcc >/dev/null 2>&1; then
  nvcc_path="$(command -v nvcc)"
  cuda_home="$(cd -- "$(dirname -- "${nvcc_path}")/.." && pwd)"
elif [[ -x /usr/local/cuda/bin/nvcc ]]; then
  cuda_home="/usr/local/cuda"
else
  echo "Unable to find nvcc. Set CUDA_HOME or add nvcc to PATH." >&2
  return 1 2>/dev/null || exit 1
fi

export ACCELWATTCH_UBENCH_ROOT="${repo_root}"
export CUDA_HOME="${cuda_home}"
export CUDA_PATH="${cuda_home}"
export NVCC="${cuda_home}/bin/nvcc"
export PATH="${cuda_home}/bin:${PATH}"
export LD_LIBRARY_PATH="${cuda_home}/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  "${NVCC}" --version
  printf 'ACCELWATTCH_UBENCH_ROOT=%s\n' "${ACCELWATTCH_UBENCH_ROOT}"
  printf 'CUDA_HOME=%s\n' "${CUDA_HOME}"
  printf 'NVCC=%s\n' "${NVCC}"
fi
