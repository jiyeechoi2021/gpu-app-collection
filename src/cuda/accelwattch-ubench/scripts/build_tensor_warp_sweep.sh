#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
root_dir="$(cd -- "${script_dir}/.." && pwd)"
benchmark_dir="${root_dir}/tensor_benchmarks/TENSOR"
original_source="${benchmark_dir}/tensorcore.cu"
original_makefile="${benchmark_dir}/Makefile"
generated_dir="${root_dir}/generated/tensor_warp_sweep"
validation_script="${script_dir}/validate_tensor_warp_variants.py"
output_dir=""
cuda_sm="70"
nvcc=""
cuobjdump=""
ptxas=""
self_check=false

readonly canonicalizer_version="task006b-sass-canonicalizer-v1"
readonly build_id_schema_version="task006b-build-identity-v1"
readonly expected_original_sha256="02e382f4746e0216574faba8ed431461c553f041855aae4ce136f5835554eed2"
readonly expected_makefile_sha256="afca3279374d4fae7068a47fa9dc47b741ebd78745bf2a80b59c5d985105c9d8"
readonly binary_execution_allowed="false"

readonly -a build_passes=(pass1 pass2)
readonly -a build_identities=(T0_make T0_direct T1 T2 T3 T4)
readonly -a primary_variant_identities=(T0_make T1 T2 T3 T4)
readonly -a recognized_options=(
  --cuda-sm --nvcc --cuobjdump --ptxas --output-dir --self-check --help -h
)
readonly -a manifest_resource_fields=(
  register_count_per_thread
  ptxas_spill_store_bytes
  ptxas_spill_load_bytes
  stack_frame_bytes
  constant_memory_bytes_by_bank
  binary_size_bytes
  ptxas_output_lines
  resource_parse_status
  resource_parse_reason
)
readonly -a build_id_input_fields=(
  schema_version
  source_sha256_by_variant
  protected_original_source_sha256
  protected_makefile_sha256
  maintained_build_script_sha256
  nvcc_executable_sha256
  cuobjdump_executable_sha256
  ptxas_executable_sha256
  nvcc_version
  target_architecture
  normalized_compile_flags
  normalized_include_paths
  normalized_linker_flags
  canonicalizer_version
)
readonly -a mismatch_classifications=(
  identical
  binary_only_difference
  raw_sass_only_difference
  canonical_sass_difference
  resource_difference
  effective_option_difference
  t0_build_path_difference
  pass_reproducibility_failure
  cross_variant_device_code_difference
  tool_or_source_identity_difference
  compiler_warning_difference
  evidence_missing
)

usage() {
  printf '%s\n' \
    'Usage: build_tensor_warp_sweep.sh [--cuda-sm 70] [--nvcc PATH]' \
    '                                  [--cuobjdump PATH] [--ptxas PATH]' \
    '                                  [--output-dir PATH] [--self-check]' \
    '' \
    'Builds T0_make, T0_direct, and T1-T4 twice, then records build-path,' \
    'reproducibility, SASS, resource, and identity evidence. It never' \
    'executes a generated binary. --self-check performs no CUDA work.'
}

contains_exactly() {
  local expected="$1"
  shift
  [[ " $* " == " ${expected} " ]]
}

run_self_check() {
  local failed=0
  local required

  contains_exactly "pass1 pass2" "${build_passes[@]}" || {
    printf 'self-check: build passes are not exactly pass1/pass2\n' >&2
    failed=1
  }
  contains_exactly "T0_make T0_direct T1 T2 T3 T4" "${build_identities[@]}" || {
    printf 'self-check: build matrix is incomplete\n' >&2
    failed=1
  }
  contains_exactly "T0_make T1 T2 T3 T4" "${primary_variant_identities[@]}" || {
    printf 'self-check: primary T0-T4 matrix is incomplete\n' >&2
    failed=1
  }
  [[ "${canonicalizer_version}" == "task006b-sass-canonicalizer-v1" ]] || failed=1
  [[ "${build_id_schema_version}" == "task006b-build-identity-v1" ]] || failed=1
  [[ "${binary_execution_allowed}" == "false" ]] || failed=1

  for required in \
    --cuda-sm --nvcc --cuobjdump --ptxas --output-dir --self-check --help; do
    [[ " ${recognized_options[*]} " == *" ${required} "* ]] || {
      printf 'self-check: missing recognized option %s\n' "${required}" >&2
      failed=1
    }
  done
  for required in \
    register_count_per_thread ptxas_spill_store_bytes \
    ptxas_spill_load_bytes stack_frame_bytes constant_memory_bytes_by_bank \
    binary_size_bytes ptxas_output_lines resource_parse_status; do
    [[ " ${manifest_resource_fields[*]} " == *" ${required} "* ]] || {
      printf 'self-check: missing manifest resource field %s\n' "${required}" >&2
      failed=1
    }
  done
  for required in \
    source_sha256_by_variant protected_original_source_sha256 \
    protected_makefile_sha256 maintained_build_script_sha256 \
    nvcc_executable_sha256 cuobjdump_executable_sha256 \
    ptxas_executable_sha256 nvcc_version target_architecture \
    normalized_compile_flags normalized_include_paths normalized_linker_flags \
    canonicalizer_version; do
    [[ " ${build_id_input_fields[*]} " == *" ${required} "* ]] || {
      printf 'self-check: missing build-id input %s\n' "${required}" >&2
      failed=1
    }
  done
  for required in \
    identical binary_only_difference raw_sass_only_difference \
    canonical_sass_difference resource_difference effective_option_difference \
    t0_build_path_difference pass_reproducibility_failure \
    cross_variant_device_code_difference tool_or_source_identity_difference \
    compiler_warning_difference evidence_missing; do
    [[ " ${mismatch_classifications[*]} " == *" ${required} "* ]] || {
      printf 'self-check: missing mismatch classification %s\n' "${required}" >&2
      failed=1
    }
  done

  if grep -nE '^[[:space:]]*"?\$\{[^}]*binary[^}]*\}"?[[:space:]]*($|[|>&])' \
      "${BASH_SOURCE[0]}" >/dev/null; then
    printf 'self-check: possible direct binary execution command found\n' >&2
    failed=1
  fi
  if grep -nE '^[[:space:]]*(ncu|nsys|nvidia-smi|run_tensor(_warp_sweep)?)([[:space:]]|$)' \
      "${BASH_SOURCE[0]}" >/dev/null; then
    printf 'self-check: prohibited runtime/profiling command found\n' >&2
    failed=1
  fi
  if ! python3 - "${BASH_SOURCE[0]}" <<'PYSELF'
import pathlib
import re
import sys

source = pathlib.Path(sys.argv[1]).read_text(encoding="utf-8")
blocks = re.findall(r"<<'PY'\n(.*?)\nPY$", source, flags=re.MULTILINE | re.DOTALL)
if len(blocks) != 2:
    raise SystemExit("expected exactly two embedded Python blocks")
for index, block in enumerate(blocks, start=1):
    compile(block, f"embedded-task006b-python-{index}", "exec")
PYSELF
  then
    printf 'self-check: embedded Python syntax check failed\n' >&2
    failed=1
  fi

  if ((failed)); then
    printf 'Task006B static self-check: FAIL\n' >&2
    return 1
  fi
  printf 'Task006B static self-check: PASS\n'
  printf 'CUDA compilation invoked: false\n'
  printf 'Generated binary execution allowed: false\n'
}

while (($#)); do
  case "$1" in
    --cuda-sm)
      [[ $# -ge 2 ]] || { printf '%s\n' '--cuda-sm requires a value' >&2; exit 2; }
      cuda_sm="$2"
      shift 2
      ;;
    --nvcc)
      [[ $# -ge 2 ]] || { printf '%s\n' '--nvcc requires a value' >&2; exit 2; }
      nvcc="$2"
      shift 2
      ;;
    --cuobjdump)
      [[ $# -ge 2 ]] || { printf '%s\n' '--cuobjdump requires a value' >&2; exit 2; }
      cuobjdump="$2"
      shift 2
      ;;
    --ptxas)
      [[ $# -ge 2 ]] || { printf '%s\n' '--ptxas requires a value' >&2; exit 2; }
      ptxas="$2"
      shift 2
      ;;
    --output-dir)
      [[ $# -ge 2 ]] || { printf '%s\n' '--output-dir requires a value' >&2; exit 2; }
      output_dir="$2"
      shift 2
      ;;
    --self-check)
      self_check=true
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ "${self_check}" == true ]]; then
  run_self_check
  exit
fi

if [[ ! "${cuda_sm}" =~ ^[0-9]+$ ]]; then
  printf 'CUDA architecture must be numeric, received: %s\n' "${cuda_sm}" >&2
  exit 2
fi

if [[ -z "${output_dir}" ]]; then
  output_dir="${generated_dir}/build_validation/sm_${cuda_sm}"
fi
output_dir="$(realpath -m -- "${output_dir}")"
generated_dir="$(realpath -m -- "${generated_dir}")"
case "${output_dir}" in
  "${generated_dir}"/*) ;;
  *)
    printf 'Build-validation output must be inside %s\n' "${generated_dir}" >&2
    exit 2
    ;;
esac

mkdir -p "${generated_dir}/manifests"
python3 "${validation_script}" \
  --generated-dir "${generated_dir}" \
  --output-json "${generated_dir}/manifests/static_validation.json" \
  > /dev/null

actual_original_sha256="$(sha256sum "${original_source}" | awk '{print $1}')"
actual_makefile_sha256="$(sha256sum "${original_makefile}" | awk '{print $1}')"
if [[ "${actual_original_sha256}" != "${expected_original_sha256}" ]]; then
  printf 'Original source hash differs from the reviewed Task006A baseline.\n' >&2
  printf 'Expected: %s\nActual:   %s\n' \
    "${expected_original_sha256}" "${actual_original_sha256}" >&2
  exit 1
fi
if [[ "${actual_makefile_sha256}" != "${expected_makefile_sha256}" ]]; then
  printf 'Original Makefile hash differs from the reviewed Task006A baseline.\n' >&2
  printf 'Expected: %s\nActual:   %s\n' \
    "${expected_makefile_sha256}" "${actual_makefile_sha256}" >&2
  exit 1
fi

if [[ -z "${nvcc}" ]]; then
  nvcc="$(command -v nvcc || true)"
fi
if [[ -z "${nvcc}" && -x /usr/local/cuda/bin/nvcc ]]; then
  nvcc="/usr/local/cuda/bin/nvcc"
fi

availability_manifest="${generated_dir}/manifests/build_validation_availability.json"
if [[ -z "${nvcc}" || ! -x "${nvcc}" ]]; then
  python3 - "${availability_manifest}" "${cuda_sm}" \
    "${expected_original_sha256}" "${expected_makefile_sha256}" <<'PY'
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
data = {
    "schema_version": 2,
    "record_kind": "tool_availability",
    "build_availability": "nvcc_unavailable",
    "build_validation_status": "not_performed",
    "benchmark_executed": False,
    "cuda_target": "sm_" + sys.argv[2],
    "reviewed_original_source_sha256": sys.argv[3],
    "reviewed_makefile_sha256": sys.argv[4],
    "variants": [],
}
for variant_id in ("T0", "T1", "T2", "T3", "T4"):
    data["variants"].append({
        "variant_id": variant_id,
        "build_status": "not_performed",
        "compiler_version": None,
        "cuda_version": None,
        "nvcc_command": None,
        "ptxas_output_lines": None,
        "register_count_per_thread": None,
        "ptxas_spill_store_bytes": None,
        "ptxas_spill_load_bytes": None,
        "stack_frame_bytes": None,
        "constant_memory_bytes_by_bank": None,
        "binary_size_bytes": None,
        "binary_sha256": None,
        "raw_wmma_sass_sha256": None,
        "canonical_wmma_sass_sha256": None,
    })
path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  printf 'Task006A static validation passed.\n'
  printf 'Task006B build validation not performed: nvcc is unavailable.\n'
  printf 'Availability record: %s\n' "${availability_manifest}"
  exit 0
fi

nvcc="$(cd -- "$(dirname -- "${nvcc}")" && pwd)/$(basename -- "${nvcc}")"
cuda_home="$(cd -- "$(dirname -- "${nvcc}")/.." && pwd)"
if [[ -z "${cuobjdump}" && -x "${cuda_home}/bin/cuobjdump" ]]; then
  cuobjdump="${cuda_home}/bin/cuobjdump"
fi
if [[ -z "${cuobjdump}" ]]; then
  cuobjdump="$(command -v cuobjdump || true)"
fi
if [[ -z "${cuobjdump}" || ! -x "${cuobjdump}" ]]; then
  printf 'Task006B requires cuobjdump. No build was started.\n' >&2
  exit 1
fi
cuobjdump="$(cd -- "$(dirname -- "${cuobjdump}")" && pwd)/$(basename -- "${cuobjdump}")"
if [[ -z "${ptxas}" && -x "${cuda_home}/bin/ptxas" ]]; then
  ptxas="${cuda_home}/bin/ptxas"
elif [[ -z "${ptxas}" ]]; then
  ptxas="$(command -v ptxas || true)"
fi
if [[ -n "${ptxas}" ]]; then
  [[ -x "${ptxas}" ]] || { printf 'Specified ptxas is not executable: %s\n' "${ptxas}" >&2; exit 1; }
  ptxas="$(cd -- "$(dirname -- "${ptxas}")" && pwd)/$(basename -- "${ptxas}")"
fi

nvccflags=(
  -O3
  -std=c++17
  --expt-relaxed-constexpr
  "-arch=sm_${cuda_sm}"
  "-I${cuda_home}/include"
  -Xptxas=-v
)
ldflags=("-L${cuda_home}/lib64" -lcublas -lcurand)

if [[ -e "${output_dir}" ]] && \
    find "${output_dir}" -mindepth 1 -print -quit | grep -q .; then
  printf 'Refusing to overwrite an existing build-validation directory: %s\n' \
    "${output_dir}" >&2
  printf 'Choose a new --output-dir for a new validation attempt.\n' >&2
  exit 1
fi
mkdir -p "${output_dir}"

record_command() {
  local path="$1"
  shift
  {
    printf '%q ' "$@"
    printf '\n'
  } > "${path}"
}

run_logged() {
  local log_path="$1"
  local status_path="$2"
  shift 2
  local status
  if "$@" > "${log_path}" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "${status}" > "${status_path}"
  return 0
}

run_logged_append() {
  local log_path="$1"
  local status_path="$2"
  shift 2
  local status
  if "$@" >> "${log_path}" 2>&1; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "${status}" > "${status_path}"
  return 0
}

run_split_output() {
  local stdout_path="$1"
  local stderr_path="$2"
  local status_path="$3"
  shift 3
  local status
  if "$@" > "${stdout_path}" 2> "${stderr_path}"; then
    status=0
  else
    status=$?
  fi
  printf '%s\n' "${status}" > "${status_path}"
  return 0
}

build_direct_identity() {
  local identity="$1"
  local source="$2"
  local binary_dir="$3"
  local object_dir="$4"
  local log_dir="$5"
  local command_dir="$6"
  local status_dir="$7"
  local object_path="${object_dir}/${identity}.o"
  local binary_path="${binary_dir}/TENSOR_${identity}"
  local log_path="${log_dir}/${identity}.log"
  local compile_status
  local -a compile_command link_command

  compile_command=("${nvcc}" "${nvccflags[@]}" -c "${source}" -o "${object_path}")
  link_command=("${nvcc}" "${nvccflags[@]}" "${object_path}" -o "${binary_path}" "${ldflags[@]}")
  record_command "${command_dir}/${identity}_compile.argv" "${compile_command[@]}"
  record_command "${command_dir}/${identity}_link.argv" "${link_command[@]}"
  : > "${log_path}"
  run_logged_append "${log_path}" "${status_dir}/${identity}_compile.status" \
    "${compile_command[@]}"
  compile_status="$(<"${status_dir}/${identity}_compile.status")"
  if [[ "${compile_status}" == 0 ]]; then
    run_logged_append "${log_path}" "${status_dir}/${identity}_link.status" \
      "${link_command[@]}"
  else
    printf 'not_run_compile_failed\n' > "${status_dir}/${identity}_link.status"
  fi
}

inspect_identity() {
  local identity="$1"
  local binary_path="$2"
  local inspection_dir="$3"
  local command_dir="$4"
  local status_dir="$5"
  local -a sass_command resource_command

  if [[ ! -s "${binary_path}" ]]; then
    printf 'not_run_binary_missing\n' > "${status_dir}/${identity}_sass.status"
    printf 'not_run_binary_missing\n' > "${status_dir}/${identity}_resource.status"
    return
  fi
  sass_command=("${cuobjdump}" --dump-sass "${binary_path}")
  resource_command=("${cuobjdump}" --dump-resource-usage "${binary_path}")
  record_command "${command_dir}/${identity}_cuobjdump_sass.argv" "${sass_command[@]}"
  record_command "${command_dir}/${identity}_cuobjdump_resource.argv" "${resource_command[@]}"
  run_split_output "${inspection_dir}/${identity}_complete.sass" \
    "${inspection_dir}/${identity}_complete.sass.stderr.txt" \
    "${status_dir}/${identity}_sass.status" "${sass_command[@]}"
  run_split_output "${inspection_dir}/${identity}_complete.resources.txt" \
    "${inspection_dir}/${identity}_complete.resources.stderr.txt" \
    "${status_dir}/${identity}_resource.status" "${resource_command[@]}"
}

build_pass() {
  local pass="$1"
  local pass_dir="${output_dir}/${pass}"
  local binary_dir="${pass_dir}/binaries"
  local object_dir="${pass_dir}/objects"
  local log_dir="${pass_dir}/logs"
  local command_dir="${pass_dir}/commands"
  local inspection_dir="${pass_dir}/inspection"
  local status_dir="${pass_dir}/status"
  local variant source binary_path dry_status
  local -a make_args make_dry_run make_build

  mkdir -p "${binary_dir}" "${object_dir}" "${log_dir}" \
    "${command_dir}" "${inspection_dir}" "${status_dir}"

  make_args=(
    -C "${benchmark_dir}" release
    "EXECUTABLE=TENSOR_T0_make"
    "BINDIR=${binary_dir}"
    "OBJDIR=${object_dir}/T0_make"
    "NVCC=${nvcc}"
    "CUDA_HOME=${cuda_home}"
    "CUDA_SM=${cuda_sm}"
    "NVCCFLAGS=${nvccflags[*]}"
    "LDLIBS=${ldflags[*]}"
  )
  make_dry_run=(make -n "${make_args[@]}")
  make_build=(make "${make_args[@]}")
  record_command "${command_dir}/T0_make_make_dry_run.argv" "${make_dry_run[@]}"
  record_command "${command_dir}/T0_make_make_build.argv" "${make_build[@]}"
  run_logged "${command_dir}/T0_make_make_dry_run.stdout" \
    "${status_dir}/T0_make_dry_run.status" "${make_dry_run[@]}"
  dry_status="$(<"${status_dir}/T0_make_dry_run.status")"
  if [[ "${dry_status}" == 0 ]]; then
    run_logged "${log_dir}/T0_make.log" "${status_dir}/T0_make_build.status" \
      "${make_build[@]}"
  else
    printf 'not_run_dry_run_failed\n' > "${status_dir}/T0_make_build.status"
    : > "${log_dir}/T0_make.log"
  fi

  build_direct_identity T0_direct "${original_source}" "${binary_dir}" \
    "${object_dir}" "${log_dir}" "${command_dir}" "${status_dir}"
  for variant in T1 T2 T3 T4; do
    source="${generated_dir}/sources/TENSOR_${variant}.cu"
    build_direct_identity "${variant}" "${source}" "${binary_dir}" \
      "${object_dir}" "${log_dir}" "${command_dir}" "${status_dir}"
  done

  for variant in "${build_identities[@]}"; do
    binary_path="${binary_dir}/TENSOR_${variant}"
    inspect_identity "${variant}" "${binary_path}" "${inspection_dir}" \
      "${command_dir}" "${status_dir}"
  done
}

for pass in "${build_passes[@]}"; do
  build_pass "${pass}"
done

python3 - "${output_dir}" "${generated_dir}" "${root_dir}" \
  "${BASH_SOURCE[0]}" "${nvcc}" "${cuobjdump}" "${ptxas}" "${cuda_sm}" \
  "${canonicalizer_version}" "${build_id_schema_version}" \
  "${expected_original_sha256}" "${expected_makefile_sha256}" <<'PY'
import hashlib
import json
import pathlib
import re
import shlex
import subprocess
import sys

(
    output_dir_arg,
    generated_dir_arg,
    root_dir_arg,
    script_path_arg,
    nvcc_arg,
    cuobjdump_arg,
    ptxas_arg,
    cuda_sm,
    canonicalizer_version,
    build_id_schema_version,
    reviewed_original_sha256,
    reviewed_makefile_sha256,
) = sys.argv[1:]

output_dir = pathlib.Path(output_dir_arg)
generated_dir = pathlib.Path(generated_dir_arg)
root_dir = pathlib.Path(root_dir_arg)
script_path = pathlib.Path(script_path_arg)
nvcc = pathlib.Path(nvcc_arg)
cuobjdump = pathlib.Path(cuobjdump_arg)
ptxas = pathlib.Path(ptxas_arg) if ptxas_arg else None
original_source = root_dir / "tensor_benchmarks/TENSOR/tensorcore.cu"
original_makefile = root_dir / "tensor_benchmarks/TENSOR/Makefile"
generation_manifest_path = generated_dir / "manifests/generation_manifest.json"
generation = json.loads(generation_manifest_path.read_text(encoding="utf-8"))
source_records = {item["variant_id"]: item for item in generation["variants"]}
passes = ("pass1", "pass2")
identities = ("T0_make", "T0_direct", "T1", "T2", "T3", "T4")
primary_identities = ("T0_make", "T1", "T2", "T3", "T4")


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha256_bytes(path.read_bytes())


def read_status(path):
    if not path.exists():
        return None
    value = path.read_text(encoding="utf-8").strip()
    return int(value) if value.isdigit() else value


def rel(path):
    return str(path.relative_to(generated_dir))


def read_argv(path):
    if not path.exists():
        return None
    return shlex.split(path.read_text(encoding="utf-8"))


def commands_from_make_output(path):
    if not path.exists():
        return []
    commands = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            argv = shlex.split(line)
        except ValueError:
            continue
        if argv and pathlib.Path(argv[0]).name == pathlib.Path(nvcc).name:
            commands.append(argv)
    return commands


def split_compile_link(commands):
    compile_commands = [argv for argv in commands if "-c" in argv]
    link_commands = [argv for argv in commands if "-c" not in argv]
    if len(compile_commands) != 1 or len(link_commands) != 1:
        return None, None
    return compile_commands[0], link_commands[0]


def normalized_path(value):
    return str(pathlib.Path(value).resolve(strict=False))


def normalize_compile_argv(argv):
    if not argv:
        return None
    options = []
    include_paths = []
    normalized_argv = [normalized_path(argv[0])]
    target_architecture = None
    source_path = None
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "-o" and index + 1 < len(argv):
            normalized_argv.extend(("-o", "<OUTPUT>"))
            index += 2
            continue
        if token == "-c":
            normalized_argv.append(token)
            index += 1
            continue
        if token == "-I" and index + 1 < len(argv):
            path = normalized_path(argv[index + 1])
            include_paths.append(path)
            normalized_argv.extend(("-I", path))
            index += 2
            continue
        if token.startswith("-I") and token != "-I":
            path = normalized_path(token[2:])
            include_paths.append(path)
            normalized_argv.append("-I" + path)
            index += 1
            continue
        if token.startswith("-arch="):
            target_architecture = token.split("=", 1)[1]
            options.append(token)
            normalized_argv.append(token)
            index += 1
            continue
        if not token.startswith("-") and token.endswith(".cu"):
            source_path = normalized_path(token)
            normalized_argv.append("<SOURCE>")
            index += 1
            continue
        options.append(token)
        normalized_argv.append(token)
        index += 1
    return {
        "compiler_executable": normalized_path(argv[0]),
        "source_path": source_path,
        "target_architecture": target_architecture,
        "normalized_compile_flags": options,
        "normalized_include_paths": include_paths,
        "normalized_argv": normalized_argv,
    }


def normalize_link_argv(argv):
    if not argv:
        return None
    linker_flags = []
    normalized_argv = [normalized_path(argv[0])]
    index = 1
    while index < len(argv):
        token = argv[index]
        if token == "-o" and index + 1 < len(argv):
            normalized_argv.extend(("-o", "<OUTPUT>"))
            index += 2
            continue
        if token == "-L" and index + 1 < len(argv):
            normalized = normalized_path(argv[index + 1])
            linker_flags.extend(("-L", normalized))
            normalized_argv.extend(("-L", normalized))
            index += 2
            continue
        if token.startswith("-L") and token != "-L":
            normalized = "-L" + normalized_path(token[2:])
            linker_flags.append(normalized)
            normalized_argv.append(normalized)
            index += 1
            continue
        if not token.startswith("-") and token.endswith((".o", ".obj")):
            normalized_argv.append("<OBJECT>")
            index += 1
            continue
        if token.startswith("-"):
            linker_flags.append(token)
        normalized_argv.append(token)
        index += 1
    return {
        "compiler_executable": normalized_path(argv[0]),
        "normalized_linker_flags": linker_flags,
        "normalized_argv": normalized_argv,
    }


def effective_configuration(compile_argv, link_argv):
    compile_record = normalize_compile_argv(compile_argv)
    link_record = normalize_link_argv(link_argv)
    if compile_record is None or link_record is None:
        return None
    return {
        "compiler_executable": compile_record["compiler_executable"],
        "compiler_executable_sha256": sha256(nvcc),
        "target_architecture": compile_record["target_architecture"],
        "normalized_compile_flags": compile_record["normalized_compile_flags"],
        "normalized_include_paths": compile_record["normalized_include_paths"],
        "normalized_linker_flags": link_record["normalized_linker_flags"],
    }


def extract_function_bytes(data, function_name):
    lines = data.splitlines(keepends=True)
    start = None
    end = len(lines)
    for index, line in enumerate(lines):
        text = line.decode("utf-8", errors="replace")
        is_header = "Function" in text and (
            text.lstrip().startswith("Function") or "Function :" in text
        )
        if not is_header:
            continue
        if start is None and function_name in text:
            start = index
        elif start is not None:
            end = index
            break
    if start is None:
        return None
    return b"".join(lines[start:end])


def canonicalize_sass(raw_bytes):
    # v1 deliberately performs only presentation normalization:
    # CRLF/CR become LF through splitlines(), trailing/repeated horizontal
    # whitespace is normalized, the cuobjdump Function header and blank lines
    # are omitted, and only the leading /*instruction-address*/ display column
    # is removed. Predicates, opcode/modifiers, operands, branch targets,
    # instruction encodings/control comments, and instruction order remain.
    canonical = []
    for source_line in raw_bytes.decode("utf-8", errors="replace").splitlines():
        line = source_line.rstrip()
        if not line.strip():
            continue
        if "Function" in line and (
            line.lstrip().startswith("Function") or "Function :" in line
        ):
            continue
        line = re.sub(r"^\s*/\*[0-9A-Fa-f]+\*/\s*", "", line)
        line = re.sub(r"[ \t]+", " ", line).strip()
        canonical.append(line)
    return ("\n".join(canonical) + "\n").encode("utf-8")


def parse_ptxas(log_text):
    ptxas_lines = [line for line in log_text.splitlines() if "ptxas" in line]
    compiler_warnings = [
        line for line in log_text.splitlines() if "warning" in line.lower()
    ]
    selected = []
    in_wmma = False
    for line in ptxas_lines:
        if "Compiling entry function" in line:
            in_wmma = "wmma_example" in line
        if in_wmma:
            selected.append(line)
    joined = "\n".join(selected)

    def parsed(pattern):
        match = re.search(pattern, joined)
        return int(match.group(1)) if match else None

    constants = {}
    for count, bank in re.findall(r"(\d+) bytes cmem\[(\d+)\]", joined):
        constants[bank] = int(count)
    values = {
        "register_count_per_thread": parsed(r"Used (\d+) registers"),
        "ptxas_spill_store_bytes": parsed(r"(\d+) bytes spill stores"),
        "ptxas_spill_load_bytes": parsed(r"(\d+) bytes spill loads"),
        "stack_frame_bytes": parsed(r"(\d+) bytes stack frame"),
        "constant_memory_bytes_by_bank": constants or None,
    }
    available = [key for key, value in values.items() if value is not None]
    values.update({
        "ptxas_output_lines": selected,
        "ptxas_warning_lines": [
            line for line in selected if "warning" in line.lower()
        ],
        "compiler_warning_lines": compiler_warnings,
        "resource_parse_status": "available" if available else "unavailable",
        "resource_parse_reason": None if available else "wmma_example ptxas fields absent",
    })
    return values


def parse_cuobjdump_resources(text):
    raw = {}
    patterns = {
        "cuobjdump_register_count": r"\bREG:(\d+)",
        "cuobjdump_stack_bytes": r"\bSTACK:(\d+)",
        "cuobjdump_shared_memory_bytes": r"\bSHARED:(\d+)",
        "cuobjdump_local_memory_bytes": r"\bLOCAL:(\d+)",
    }
    for name, pattern in patterns.items():
        match = re.search(pattern, text)
        raw[name] = int(match.group(1)) if match else None
    constants = {
        bank: int(count)
        for bank, count in re.findall(r"CONSTANT\[(\d+)\]:(\d+)", text)
    }
    raw["cuobjdump_constant_memory_bytes_by_bank"] = constants or None
    return raw


def source_for_identity(identity):
    if identity.startswith("T0_"):
        return original_source
    return generated_dir / "sources" / ("TENSOR_" + identity + ".cu")


def artifact_record(pass_name, identity):
    pass_dir = output_dir / pass_name
    command_dir = pass_dir / "commands"
    log_dir = pass_dir / "logs"
    status_dir = pass_dir / "status"
    inspection_dir = pass_dir / "inspection"
    binary = pass_dir / "binaries" / ("TENSOR_" + identity)
    log = log_dir / (identity + ".log")
    dry_run = None
    dry_compile = dry_link = None

    if identity == "T0_make":
        dry_run = command_dir / "T0_make_make_dry_run.stdout"
        dry_compile, dry_link = split_compile_link(commands_from_make_output(dry_run))
        compile_argv, link_argv = split_compile_link(commands_from_make_output(log))
        exit_statuses = {
            "make_dry_run": read_status(status_dir / "T0_make_dry_run.status"),
            "build": read_status(status_dir / "T0_make_build.status"),
        }
    else:
        compile_argv = read_argv(command_dir / (identity + "_compile.argv"))
        link_argv = read_argv(command_dir / (identity + "_link.argv"))
        exit_statuses = {
            "compile": read_status(status_dir / (identity + "_compile.status")),
            "link": read_status(status_dir / (identity + "_link.status")),
        }

    compile_normalized = normalize_compile_argv(compile_argv)
    link_normalized = normalize_link_argv(link_argv)
    effective = effective_configuration(compile_argv, link_argv)
    dry_effective = effective_configuration(dry_compile, dry_link)
    make_dry_run_matches_actual = (
        None if identity != "T0_make" or not dry_effective or not effective
        else dry_effective == effective
    )

    complete_sass = inspection_dir / (identity + "_complete.sass")
    complete_sass_stderr = inspection_dir / (identity + "_complete.sass.stderr.txt")
    complete_resources = inspection_dir / (identity + "_complete.resources.txt")
    complete_resources_stderr = inspection_dir / (identity + "_complete.resources.stderr.txt")
    raw_path = inspection_dir / (identity + "_wmma_example.raw.sass")
    canonical_path = inspection_dir / (identity + "_wmma_example.canonical.sass")
    resource_section_path = inspection_dir / (identity + "_wmma_example.resources.txt")
    raw_bytes = None
    canonical_bytes = None
    if read_status(status_dir / (identity + "_sass.status")) == 0 and complete_sass.exists():
        raw_bytes = extract_function_bytes(complete_sass.read_bytes(), "wmma_example")
        if raw_bytes is not None:
            raw_path.write_bytes(raw_bytes)
            canonical_bytes = canonicalize_sass(raw_bytes)
            canonical_path.write_bytes(canonical_bytes)

    log_text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    resources = parse_ptxas(log_text)
    resource_text = (
        complete_resources.read_text(encoding="utf-8", errors="replace")
        if complete_resources.exists() else ""
    )
    resource_section_bytes = extract_function_bytes(
        resource_text.encode("utf-8"), "wmma_example"
    ) if resource_text else None
    if resource_section_bytes is not None:
        resource_section_path.write_bytes(resource_section_bytes)
        parsed_resource_text = resource_section_bytes.decode("utf-8", errors="replace")
    else:
        parsed_resource_text = ""
    resources.update(parse_cuobjdump_resources(parsed_resource_text))
    resources["binary_size_bytes"] = binary.stat().st_size if binary.exists() else None
    resources["ptxas_evidence_artifact_path"] = rel(log) if log.exists() else None
    resources["ptxas_evidence_artifact_sha256"] = sha256(log) if log.exists() else None
    resources["cuobjdump_resource_artifact_path"] = (
        rel(complete_resources) if complete_resources.exists() else None
    )
    resources["cuobjdump_resource_artifact_sha256"] = (
        sha256(complete_resources) if complete_resources.exists() else None
    )

    statuses_are_zero = all(value == 0 for value in exit_statuses.values())
    evidence_complete = all((
        statuses_are_zero,
        binary.is_file() and binary.stat().st_size > 0,
        read_status(status_dir / (identity + "_sass.status")) == 0,
        read_status(status_dir / (identity + "_resource.status")) == 0,
        raw_bytes is not None,
        canonical_bytes is not None,
        resource_section_bytes is not None,
        resources["register_count_per_thread"] is not None,
        resources["ptxas_spill_store_bytes"] is not None,
        resources["ptxas_spill_load_bytes"] is not None,
        effective is not None,
        make_dry_run_matches_actual is not False,
    ))
    source = source_for_identity(identity)
    return {
        "build_identity": identity,
        "pass": pass_name,
        "source_path": str(source.relative_to(root_dir)),
        "source_sha256": sha256(source),
        "exit_statuses": exit_statuses,
        "build_status": "succeeded" if statuses_are_zero else "failed",
        "evidence_complete": evidence_complete,
        "binary_path": rel(binary) if binary.exists() else None,
        "binary_size_bytes": binary.stat().st_size if binary.exists() else None,
        "binary_sha256": sha256(binary) if binary.exists() else None,
        "compiler_log_path": rel(log) if log.exists() else None,
        "compiler_log_sha256": sha256(log) if log.exists() else None,
        "make_dry_run_command_path": (
            rel(command_dir / "T0_make_make_dry_run.argv")
            if identity == "T0_make" else None
        ),
        "make_dry_run_output_path": rel(dry_run) if dry_run else None,
        "make_dry_run_compile_argv": dry_compile,
        "make_dry_run_link_argv": dry_link,
        "actual_compile_argv": compile_argv,
        "actual_link_argv": link_argv,
        "normalized_compile": compile_normalized,
        "normalized_link": link_normalized,
        "normalized_effective_configuration": effective,
        "make_dry_run_effective_configuration": dry_effective,
        "make_dry_run_matches_actual": make_dry_run_matches_actual,
        "complete_sass_artifact_path": rel(complete_sass) if complete_sass.exists() else None,
        "complete_sass_artifact_sha256": sha256(complete_sass) if complete_sass.exists() else None,
        "complete_sass_stderr_path": rel(complete_sass_stderr) if complete_sass_stderr.exists() else None,
        "complete_sass_stderr_sha256": sha256(complete_sass_stderr) if complete_sass_stderr.exists() else None,
        "raw_wmma_sass_artifact_path": rel(raw_path) if raw_path.exists() else None,
        "raw_wmma_sass_sha256": sha256_bytes(raw_bytes) if raw_bytes is not None else None,
        "canonical_wmma_sass_artifact_path": rel(canonical_path) if canonical_path.exists() else None,
        "canonical_wmma_sass_sha256": (
            sha256_bytes(canonical_bytes) if canonical_bytes is not None else None
        ),
        "canonicalizer_version": canonicalizer_version,
        "complete_resource_artifact_path": rel(complete_resources) if complete_resources.exists() else None,
        "complete_resource_artifact_sha256": sha256(complete_resources) if complete_resources.exists() else None,
        "complete_resource_stderr_path": rel(complete_resources_stderr) if complete_resources_stderr.exists() else None,
        "complete_resource_stderr_sha256": sha256(complete_resources_stderr) if complete_resources_stderr.exists() else None,
        "wmma_resource_artifact_path": rel(resource_section_path) if resource_section_path.exists() else None,
        "wmma_resource_artifact_sha256": sha256(resource_section_path) if resource_section_path.exists() else None,
        "resources": resources,
        "cuobjdump_sass_exit_status": read_status(status_dir / (identity + "_sass.status")),
        "cuobjdump_resource_exit_status": read_status(status_dir / (identity + "_resource.status")),
    }


def resource_signature(record):
    resources = record["resources"]
    keys = (
        "register_count_per_thread",
        "ptxas_spill_store_bytes",
        "ptxas_spill_load_bytes",
        "stack_frame_bytes",
        "constant_memory_bytes_by_bank",
        "cuobjdump_register_count",
        "cuobjdump_stack_bytes",
        "cuobjdump_shared_memory_bytes",
        "cuobjdump_local_memory_bytes",
        "cuobjdump_constant_memory_bytes_by_bank",
    )
    return {key: resources.get(key) for key in keys}


def warning_signature(record):
    normalized = []
    for line in record["resources"]["compiler_warning_lines"]:
        position = line.lower().find("warning")
        normalized.append(line[position:].strip() if position >= 0 else line.strip())
    return normalized


def compare_pair(
    left,
    right,
    compare_binary=True,
    require_equal_exit_statuses=True,
    compare_source_identity=True,
):
    checks = {
        "build_exit_status_equal": left["exit_statuses"] == right["exit_statuses"],
        "both_builds_succeeded": (
            left["build_status"] == "succeeded" and right["build_status"] == "succeeded"
        ),
        "binary_sha256_equal": left["binary_sha256"] == right["binary_sha256"],
        "raw_sass_sha256_equal": (
            left["raw_wmma_sass_sha256"] == right["raw_wmma_sass_sha256"]
        ),
        "canonical_sass_sha256_equal": (
            left["canonical_wmma_sass_sha256"] == right["canonical_wmma_sass_sha256"]
        ),
        "resources_equal": resource_signature(left) == resource_signature(right),
        "compiler_warnings_equal": warning_signature(left) == warning_signature(right),
        "effective_options_equal": (
            left["normalized_effective_configuration"]
            == right["normalized_effective_configuration"]
        ),
        "source_identity_equal": left["source_sha256"] == right["source_sha256"],
        "evidence_complete": left["evidence_complete"] and right["evidence_complete"],
    }
    if not checks["evidence_complete"]:
        classification = "evidence_missing"
    elif not checks["both_builds_succeeded"] or (
        require_equal_exit_statuses and not checks["build_exit_status_equal"]
    ):
        classification = "pass_reproducibility_failure"
    elif compare_source_identity and not checks["source_identity_equal"]:
        classification = "tool_or_source_identity_difference"
    elif not checks["effective_options_equal"]:
        classification = "effective_option_difference"
    elif not checks["canonical_sass_sha256_equal"]:
        classification = "canonical_sass_difference"
    elif not checks["resources_equal"]:
        classification = "resource_difference"
    elif not checks["compiler_warnings_equal"]:
        classification = "compiler_warning_difference"
    elif not checks["raw_sass_sha256_equal"]:
        classification = "raw_sass_only_difference"
    elif compare_binary and not checks["binary_sha256_equal"]:
        classification = "binary_only_difference"
    else:
        classification = "identical"
    return {"checks": checks, "classification": classification}


records = {
    pass_name: {
        identity: artifact_record(pass_name, identity) for identity in identities
    }
    for pass_name in passes
}

compiler_version_process = subprocess.run(
    [str(nvcc), "--version"], check=False, capture_output=True, text=True
)
compiler_version = compiler_version_process.stdout + compiler_version_process.stderr
cuda_match = re.search(r"release\s+([^,\s]+)", compiler_version)
cuda_version = cuda_match.group(1) if cuda_match else None

reference_config = records["pass1"]["T0_direct"]["normalized_effective_configuration"]
if reference_config is None:
    normalized_compile_flags = None
    normalized_include_paths = None
    normalized_linker_flags = None
else:
    normalized_compile_flags = reference_config["normalized_compile_flags"]
    normalized_include_paths = reference_config["normalized_include_paths"]
    normalized_linker_flags = reference_config["normalized_linker_flags"]

source_sha256_by_variant = {
    variant: source_records[variant]["source_sha256"]
    for variant in ("T0", "T1", "T2", "T3", "T4")
}
build_identity_inputs = {
    "schema_version": build_id_schema_version,
    "source_sha256_by_variant": source_sha256_by_variant,
    "protected_original_source_sha256": sha256(original_source),
    "protected_makefile_sha256": sha256(original_makefile),
    "maintained_build_script_sha256": sha256(script_path),
    "nvcc_executable_sha256": sha256(nvcc),
    "cuobjdump_executable_sha256": sha256(cuobjdump),
    "ptxas_executable_sha256": sha256(ptxas) if ptxas is not None else None,
    "nvcc_version": compiler_version,
    "target_architecture": "sm_" + cuda_sm,
    "normalized_compile_flags": normalized_compile_flags,
    "normalized_include_paths": normalized_include_paths,
    "normalized_linker_flags": normalized_linker_flags,
    "canonicalizer_version": canonicalizer_version,
}
canonical_identity_json = json.dumps(
    build_identity_inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True
)
identity_path = output_dir / "build_identity_inputs.canonical.json"
identity_path.write_text(canonical_identity_json, encoding="utf-8")
build_id = sha256_bytes(canonical_identity_json.encode("utf-8"))
for pass_name in passes:
    for identity in identities:
        records[pass_name][identity]["build_id"] = build_id

pass_comparisons = {}
review_reasons = []
all_classifications = []
if compiler_version_process.returncode != 0 or not compiler_version.strip():
    review_reasons.append("evidence_missing")
    all_classifications.append("evidence_missing")
for identity in identities:
    comparison = compare_pair(records["pass1"][identity], records["pass2"][identity])
    detailed = comparison["classification"]
    if detailed not in ("identical", "raw_sass_only_difference"):
        comparison["gate_classification"] = "pass_reproducibility_failure"
        review_reasons.append("pass_reproducibility_failure")
        all_classifications.append("pass_reproducibility_failure")
    else:
        comparison["gate_classification"] = detailed
    all_classifications.append(detailed)
    pass_comparisons[identity] = comparison

t0_build_path_comparisons = {}
for pass_name in passes:
    comparison = compare_pair(
        records[pass_name]["T0_make"],
        records[pass_name]["T0_direct"],
        compare_binary=False,
        require_equal_exit_statuses=False,
    )
    detailed = comparison["classification"]
    if detailed not in ("identical", "raw_sass_only_difference", "binary_only_difference"):
        comparison["gate_classification"] = "t0_build_path_difference"
        review_reasons.append("t0_build_path_difference")
        all_classifications.append("t0_build_path_difference")
    else:
        comparison["gate_classification"] = detailed
    all_classifications.append(detailed)
    t0_build_path_comparisons[pass_name] = comparison

cross_variant_comparisons = {}
baseline = records["pass1"]["T0_make"]
for identity in primary_identities[1:]:
    comparison = compare_pair(
        baseline,
        records["pass1"][identity],
        compare_binary=False,
        require_equal_exit_statuses=False,
        compare_source_identity=False,
    )
    detailed = comparison["classification"]
    if detailed == "canonical_sass_difference":
        comparison["gate_classification"] = "cross_variant_device_code_difference"
        review_reasons.append("cross_variant_device_code_difference")
        all_classifications.append("cross_variant_device_code_difference")
    else:
        comparison["gate_classification"] = detailed
        if detailed not in ("identical", "raw_sass_only_difference", "binary_only_difference"):
            review_reasons.append(detailed)
    all_classifications.append(detailed)
    cross_variant_comparisons["T0_make_vs_" + identity] = comparison

for pass_name in passes:
    for identity in identities:
        record = records[pass_name][identity]
        variant_id = "T0" if identity.startswith("T0_") else identity
        if record["source_sha256"] != source_records[variant_id]["source_sha256"]:
            review_reasons.append("tool_or_source_identity_difference")
        if record["normalized_effective_configuration"] != reference_config:
            review_reasons.append("effective_option_difference")
        if identity == "T0_make" and record["make_dry_run_matches_actual"] is not True:
            review_reasons.append("effective_option_difference")

if sha256(original_source) != reviewed_original_sha256:
    review_reasons.append("tool_or_source_identity_difference")
if sha256(original_makefile) != reviewed_makefile_sha256:
    review_reasons.append("tool_or_source_identity_difference")

warning_sets = {}
for identity in identities:
    warning_sets[identity] = warning_signature(records["pass1"][identity])
if len({json.dumps(value, sort_keys=True) for value in warning_sets.values()}) != 1:
    review_reasons.append("compiler_warning_difference")
    all_classifications.append("compiler_warning_difference")

review_reasons = sorted(set(review_reasons))
all_classifications = sorted(set(all_classifications))
review_required = bool(review_reasons)
status = "review_required" if review_required else "passed"

table_lines = [
    "| Build identity | Registers (count/thread) | Spill loads (ptxas bytes) | Spill stores (ptxas bytes) | Binary size (bytes) | Pass comparison |",
    "| --- | ---: | ---: | ---: | ---: | --- |",
]
for identity in identities:
    record = records["pass1"][identity]
    resource = record["resources"]
    table_lines.append(
        "| {identity} | {registers} | {loads} | {stores} | {size} | {comparison} |".format(
            identity=identity,
            registers=resource["register_count_per_thread"],
            loads=resource["ptxas_spill_load_bytes"],
            stores=resource["ptxas_spill_store_bytes"],
            size=record["binary_size_bytes"],
            comparison=pass_comparisons[identity]["gate_classification"],
        )
    )
(output_dir / "build_comparison.md").write_text(
    "\n".join(table_lines) + "\n", encoding="utf-8"
)

manifest = {
    "schema_version": 2,
    "record_kind": "task006b_build_validation",
    "build_validation_status": status,
    "review_required": review_required,
    "review_reasons": review_reasons,
    "mismatch_classifications_observed": all_classifications,
    "documented_nonzero_exit_status": 3,
    "benchmark_executed": False,
    "build_id": build_id,
    "build_identity_formula": "sha256(canonical_json(build_identity_inputs))",
    "build_identity_inputs_path": rel(identity_path),
    "build_identity_inputs_sha256": sha256(identity_path),
    "build_identity_inputs": build_identity_inputs,
    "canonicalizer_version": canonicalizer_version,
    "cuda_target": "sm_" + cuda_sm,
    "nvcc_path": str(nvcc),
    "cuobjdump_path": str(cuobjdump),
    "ptxas_path": str(ptxas) if ptxas is not None else None,
    "nvcc_version": compiler_version,
    "nvcc_version_exit_status": compiler_version_process.returncode,
    "cuda_version": cuda_version,
    "protected_inputs": {
        "original_source_sha256": sha256(original_source),
        "original_source_reviewed_hash_matched": sha256(original_source) == reviewed_original_sha256,
        "makefile_sha256": sha256(original_makefile),
        "makefile_reviewed_hash_matched": sha256(original_makefile) == reviewed_makefile_sha256,
    },
    "passes": records,
    "comparisons": {
        "pass_reproducibility": pass_comparisons,
        "t0_build_path_equivalence": t0_build_path_comparisons,
        "cross_variant_compiler_comparability": cross_variant_comparisons,
    },
}
manifest_path = output_dir / "build_validation_manifest.json"
manifest_path.write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
)
PY

build_status="$(python3 -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["build_validation_status"])' \
  "${output_dir}/build_validation_manifest.json")"
if [[ "${build_status}" == "review_required" ]]; then
  printf 'Task006B evidence requires researcher review.\n' >&2
  printf 'Manifest: %s/build_validation_manifest.json\n' "${output_dir}" >&2
  exit 3
fi
printf 'Task006B build validation passed without executing a benchmark.\n'
printf 'Manifest: %s/build_validation_manifest.json\n' "${output_dir}"
