#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=bootstrap_tensor_env.sh
source "${script_dir}/bootstrap_tensor_env.sh"

iterations="${1:-1}"
benchmark_dir="${ACCELWATTCH_UBENCH_ROOT}/tensor_benchmarks/TENSOR"
binary="${ACCELWATTCH_UBENCH_ROOT}/bin/linux/release/TENSOR"

if ! command -v ncu >/dev/null 2>&1; then
  echo "Unable to find ncu. Install Nsight Compute or add it to PATH." >&2
  exit 1
fi

make -C "${benchmark_dir}"
exec ncu --target-processes all --set full "${binary}" "${iterations}"
