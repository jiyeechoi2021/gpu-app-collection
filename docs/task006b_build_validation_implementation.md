# Task006B: Build Validation for Experiment A

## Scope and current status

Task006B validates only whether the researcher-reviewed Task006A T0-T4 sources
compile reproducibly under one recorded CUDA toolchain and whether the compiled
`wmma_example` evidence is comparable. It never executes a generated binary.
It contains no benchmark runner, power measurement, NCU/Nsight profiling,
runtime normalization, energy calculation, or scientific conclusion.

The corrected implementation has passed static syntax and self-check review
only. It has **not** yet been run as a CUDA build and Task006B remains
**incomplete** pending a separately authorized GPU build and researcher review.

The earlier `build_validation_availability.json` was generated on the separate
local WSL host `DESKTOP-95EE45S`. That host had neither a Linux `nvcc` on
`PATH` nor `/usr/local/cuda/bin/nvcc`, so the record correctly says
`build_status=not_performed`. It is a tool-availability record, not a successful
build manifest and not evidence about the intended GPU container.

The intended GPU environment was diagnosed separately as a
`Tesla V100-SXM2-32GB` with CUDA 12.1 `nvcc`, `cuobjdump`, and `ptxas` under
`/usr/local/cuda/bin`; that directory was also present on `PATH`. No CUDA build
was performed during that diagnosis or during this implementation correction.

## Maintained implementation

The Task006B entry point is:

```text
src/cuda/accelwattch-ubench/scripts/build_tensor_warp_sweep.sh
```

The maintained script provides two modes:

```bash
# Static-only schema/matrix/invariant review; does not require CUDA.
src/cuda/accelwattch-ubench/scripts/build_tensor_warp_sweep.sh --self-check

# Future CUDA build-validation command; not run as part of this correction.
src/cuda/accelwattch-ubench/scripts/build_tensor_warp_sweep.sh \
  --cuda-sm 70 \
  --nvcc /usr/local/cuda/bin/nvcc \
  --cuobjdump /usr/local/cuda/bin/cuobjdump \
  --ptxas /usr/local/cuda/bin/ptxas
```

`--self-check` verifies without tool discovery or generated output that:

- the recognized CLI options are present;
- the build matrix is exactly `T0_make`, `T0_direct`, and T1-T4;
- the passes are exactly `pass1` and `pass2`;
- the primary experiment matrix is `T0_make`, T1, T2, T3, and T4;
- the canonicalizer version and build-identity schema are defined;
- required unit-explicit manifest fields are defined;
- all required mismatch classifications are defined; and
- no direct generated-binary, runtime, or profiler command is present.

## Protected inputs and preflight

Before any CUDA tool is invoked, the build path:

1. runs the Task006A static validator;
2. requires the original `tensorcore.cu` SHA-256 to equal the reviewed value;
3. requires the original TENSOR Makefile SHA-256 to equal the reviewed value;
4. resolves one `nvcc` and its CUDA root;
5. resolves one `cuobjdump`; and
6. records `ptxas` executable identity when it is available.

The protected hashes are:

```text
tensor_benchmarks/TENSOR/tensorcore.cu
02e382f4746e0216574faba8ed431461c553f041855aae4ce136f5835554eed2

tensor_benchmarks/TENSOR/Makefile
afca3279374d4fae7068a47fa9dc47b741ebd78745bf2a80b59c5d985105c9d8
```

Neither protected file is edited or copied back into the benchmark directory.
T0 uses the original source directly. T1-T4 use only the deterministic
Task006A generated sources.

## Build matrix and compiler settings

Every build identity is produced independently in both `pass1` and `pass2`:

| Build identity | Source | Build path | Role |
| --- | --- | --- | --- |
| `T0_make` | protected original `tensorcore.cu` | protected original Makefile | Primary T0 experiment binary |
| `T0_direct` | protected original `tensorcore.cu` | explicit direct `nvcc` compile/link | Auxiliary build-path reference only |
| `T1` | Task006A `TENSOR_T1.cu` | explicit direct `nvcc` compile/link | Primary variant |
| `T2` | Task006A `TENSOR_T2.cu` | explicit direct `nvcc` compile/link | Primary variant |
| `T3` | Task006A `TENSOR_T3.cu` | explicit direct `nvcc` compile/link | Primary variant |
| `T4` | Task006A `TENSOR_T4.cu` | explicit direct `nvcc` compile/link | Primary variant |

`T0_direct` does not replace `T0_make` and is never included as a sixth
scientific variant. It exists only to test whether the protected Makefile path
and the explicit direct path produce equivalent CUDA device evidence.

The default target is V100 `sm_70`. All identities and passes share:

```text
-O3
-std=c++17
--expt-relaxed-constexpr
-arch=sm_70
-I<resolved CUDA root>/include
-Xptxas=-v
```

Every link shares:

```text
-L<resolved CUDA root>/lib64
-lcublas
-lcurand
```

The Make invocation overrides only its output directories/name, resolved
toolchain, target, and the common compiler/link arrays. Outputs from
`T0_make` and `T0_direct` have distinct object and binary names and cannot
overwrite each other.

## Effective-command preservation and normalization

For each pass, the script preserves:

- the exact shell-escaped original-Makefile dry-run request;
- complete dry-run stdout/stderr and exit status;
- the exact shell-escaped actual Make build request;
- the complete actual Make log and exit status;
- the parsed actual Makefile `nvcc` compile and link argv;
- the direct-reference compile and link argv and their exit statuses; and
- normalized compile argv, link argv, compiler options, include paths, linker
  options, and target architecture.

The actual Make log, not only the earlier diagnostic `make -n` observation, is
the authoritative source for the effective `T0_make` compile and link argv.
The same invocation's dry-run output must agree with the actual effective
configuration.

Normalization removes only:

- the output path following `-o`;
- the one compilation source-path spelling after recording and hashing the
  resolved source; and
- the one link object path.

Include and library paths are resolved to deterministic absolute spellings.
No optimization option, architecture option, language option, preprocessor
definition, device option, include-path ordering, linker option, library-path
ordering, or linked library is removed. The normalized effective configuration
also records the resolved compiler and compiler executable SHA-256.

An absent command, an actual/dry-run disagreement, or an effective
configuration difference is evidence requiring researcher review.

## Raw and canonical `wmma_example` SASS

Every successfully inspected build preserves:

```text
complete cuobjdump SASS stdout
complete cuobjdump SASS stderr
byte-preserved wmma_example raw SASS section
versioned canonical wmma_example SASS section
SHA-256 for every artifact
canonicalizer_version=task006b-sass-canonicalizer-v1
```

Canonicalizer v1 is deliberately conservative. In source order, it:

1. normalizes CRLF/CR/LF line boundaries to LF;
2. removes trailing whitespace and blank presentation lines;
3. removes only the `cuobjdump` `Function` section header;
4. removes only a leading `/*instruction-address*/` display column; and
5. collapses repeated horizontal presentation whitespace.

It does **not** reorder instructions or operands, rename registers, rewrite
branch targets, remove labels, discard predicates, strip opcode modifiers, or
remove trailing instruction-encoding/control comments. Predicates, opcodes,
operands, control/encoding evidence, branch targets, labels, and instruction
order therefore remain comparison inputs. A future transformation requires a
new canonicalizer version and researcher review.

Raw equality and canonical equality are always recorded separately. A raw-only
difference is non-semantic only when canonical SASS, parsed resources, and
effective options all match.

## Compiler and resource evidence

`-Xptxas=-v` exposes compiler-reported information. The parser scopes ptxas
lines to the `wmma_example` entry-function block and retains the complete raw
ptxas lines and warnings. It records at least:

```text
register_count_per_thread
ptxas_spill_store_bytes
ptxas_spill_load_bytes
stack_frame_bytes
constant_memory_bytes_by_bank
binary_size_bytes
ptxas_output_lines
resource_parse_status
resource_parse_reason
```

The `ptxas_spill_*_bytes` names intentionally retain the neutral compiler-
reported byte quantity. The implementation does not relabel it as dynamic
traffic or infer a per-thread/total scope that the captured compiler line does
not explicitly establish. In particular, it is not a measurement of runtime
spill-memory traffic.

The complete `cuobjdump --dump-resource-usage` stdout and stderr are preserved.
The `wmma_example` resource section is also isolated and hashed before parsing.
Additional fields retain tool-qualified, unit-explicit names such as
`cuobjdump_stack_bytes` and `cuobjdump_shared_memory_bytes`; no unreported
scope is invented. Missing/unparseable values remain JSON `null` with a status
and reason and cannot be silently converted to zero.

## Content-based build identity

The build input identity schema is
`task006b-build-identity-v1`. The script writes canonical JSON with sorted keys
and compact separators, then computes:

```text
build_id = sha256(canonical_json(build_identity_inputs))
```

The inputs are:

```text
T0-T4 source SHA-256 values
protected original-source SHA-256
protected Makefile SHA-256
maintained build-script SHA-256
nvcc executable SHA-256
cuobjdump executable SHA-256
ptxas executable SHA-256 when available
full nvcc version output
target architecture
normalized compile flags
normalized include paths
normalized linker flags
canonicalizer version
```

No timestamp, hostname, output directory, pass name, or run identifier enters
`build_id`. Thus pass1 and pass2 share one build ID when their validated inputs
are the same. Raw attempt directories and evidence remain separate and are
never overwritten.

## Comparison gates

### Pass reproducibility

The following pairs are compared independently:

```text
T0_make pass1 vs pass2
T0_direct pass1 vs pass2
T1 pass1 vs pass2
T2 pass1 vs pass2
T3 pass1 vs pass2
T4 pass1 vs pass2
```

Each comparison records build exit status, full executable hash, raw SASS hash,
canonical SASS hash, parsed resource values, source identity, evidence
completeness, and normalized effective options. An executable-only difference
between reproducibility passes is recorded as `binary_only_difference` and
also fails the strict pass-reproducibility gate. A raw-only difference does not
force review when the canonical/resource/option evidence matches.

### T0 build-path equivalence

Within each pass, `T0_make` is compared against `T0_direct`. Both must have
successful builds and complete evidence. The gate compares normalized effective
options, raw SASS, canonical SASS, parsed resources, and protected source
identity.

The complete host executable hashes are recorded, but equality is not required
for this build-path gate because link/build-path metadata can differ without a
CUDA device-code difference.

### Cross-variant compiler comparability

Pass1 primary identities `T0_make`, T1, T2, T3, and T4 are compared for:

- canonical `wmma_example` SASS;
- raw SASS independently;
- parsed resource values;
- target architecture; and
- normalized compiler/linker configuration.

T1-T4 intentionally have different controlled source hashes, so source-hash
equality is not a cross-variant gate. Their actual source hashes are still
recorded in the build identity and each artifact record.

## Mismatch classification and stopping gate

The manifest distinguishes at least:

```text
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
```

`raw_sass_only_difference` requires raw SASS inequality together with matching
canonical SASS, resources, and effective options. It is recorded independently
and does not alone force review.

Any unresolved semantic difference, missing evidence, tool/source identity
problem, asymmetric compiler warning, T0 build-path failure, or pass-
reproducibility failure sets:

```text
review_required=true
build_validation_status=review_required
```

The script exits with documented status `3` for researcher review. It cannot
print an automatic PASS in that state. Setup/preflight/tool failures use a
nonzero failure exit. A clean automated `passed` status still requires explicit
researcher review before Task007.

## Output preservation

The default future output layout is:

```text
generated/tensor_warp_sweep/build_validation/sm_70/
├── pass1/
│   ├── binaries/TENSOR_T0_make, TENSOR_T0_direct, TENSOR_T1 ... TENSOR_T4
│   ├── objects/
│   ├── commands/
│   ├── logs/
│   ├── inspection/
│   └── status/
├── pass2/
│   └── same isolated evidence layout
├── build_identity_inputs.canonical.json
├── build_comparison.md
└── build_validation_manifest.json
```

The script refuses to overwrite a nonempty output directory. Failed command
logs and explicit exit-status files remain raw evidence. A retry must select a
new `--output-dir`.

## Scientific limits and stopping point

Task006B can establish only build-input identity, compilation reproducibility,
and static compiler/device-code comparability under the recorded toolchain. It
does not establish numerical correctness, runtime validity, occupancy,
performance, power behavior, Tensor Core energy, or energy per operation.

No generated binary is invoked anywhere in the maintained implementation.
After a future CUDA build, stop for researcher review regardless of whether the
automated status is `passed` or `review_required`. Do not run T0-T4, NCU,
Nsight Systems, NVML/power collection, a benchmark, or Task007 as part of
Task006B.
