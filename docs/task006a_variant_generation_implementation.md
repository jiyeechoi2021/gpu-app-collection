# Task006A: Reproducible Variant Generation for Experiment A

## Scope

Task006A proves only that controlled T0-T4 source variants for Experiment A can
be generated reproducibly while preserving the original AccelWattch TENSOR
source.

The maintained scientific artifact consists of:

1. the immutable original `tensorcore.cu`;
2. one authoritative CSV variant specification;
3. deterministic generator version `task006-generator-v1`; and
4. a static validator.

This task stops after static validation. It does not build or execute any
variant and contains no `nvcc`, ptxas, binary, power, NCU, extraction,
normalization, analysis, or result-reporting implementation.

## Files created

| Path | Purpose |
| --- | --- |
| `src/cuda/accelwattch-ubench/configs/tensor_warp_sweep.csv` | Authoritative machine-readable Experiment A variant specification. |
| `src/cuda/accelwattch-ubench/scripts/generate_tensor_warp_variants.py` | Generates T1-T4 sources, unified diffs, hashes, and provenance from the original source and CSV. T0 resolves directly to the original. |
| `src/cuda/accelwattch-ubench/scripts/validate_tensor_warp_variants.py` | Validates source preservation, approved changes, exact Experiment A geometry, arithmetic, hashes, and deterministic regeneration. |
| `docs/task006a_variant_generation_implementation.md` | Records Task006A design, commands, outputs, and validation result. |

The root `.gitignore` was extended only to exclude:

```text
src/cuda/accelwattch-ubench/generated/
```

Generated sources, diffs, and manifests therefore remain derived artifacts and
cannot be confused with maintained benchmark source.

The pre-existing `build_tensor_warp_sweep.sh` is not part of Task006A, was
restored to its pre-Task006 contents, and was not invoked. Task006A adds no build
behavior.

## Authoritative variant specification

| Variant | Block dimensions | Grid dimensions | Threads/block | Warps/block | Blocks | Total threads | Total warps | Requested work/T0 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 | `(128,4,1)` | `(16,16,1)` | 512 | 16 | 256 | 131,072 | 4,096 | 1 |
| T1 | `(128,2,1)` | `(16,16,1)` | 256 | 8 | 256 | 65,536 | 2,048 | 0.5 |
| T2 | `(128,1,1)` | `(16,16,1)` | 128 | 4 | 256 | 32,768 | 1,024 | 0.25 |
| T3 | `(64,1,1)` | `(16,16,1)` | 64 | 2 | 256 | 16,384 | 512 | 0.125 |
| T4 | `(32,1,1)` | `(16,16,1)` | 32 | 1 | 256 | 8,192 | 256 | 0.0625 |

The CSV contains these required fields:

```text
variant_id
blockDim.x, blockDim.y, blockDim.z
gridDim.x, gridDim.y, gridDim.z
threads_per_block, warps_per_block
total_blocks, total_launched_threads, total_launched_warps
requested_work_relative_to_T0
```

The validator enforces the exact T0-T4 block/grid tuples, not merely internally
consistent arithmetic. A different grid or block value is rejected even if its
derived totals are mathematically valid.

## Deterministic generation procedure

The generator:

1. reads the original source as bytes and records its SHA-256;
2. reads and validates the CSV using only Python's standard library;
3. verifies each expected original launch expression occurs exactly once;
4. resolves T0 to the original source without creating a T0 copy;
5. produces T1-T4 by replacing only the two block assignments and two grid
   expressions;
6. writes one canonical unified diff for each generated variant;
7. hashes every generated source and diff;
8. writes a deterministic JSON generation manifest without timestamps or
   machine-specific absolute paths;
9. verifies the original source and Makefile bytes remain unchanged; and
10. refuses to write inside the original benchmark directory or overwrite a
    non-identical generated artifact.

The approved substitutions are limited to:

```text
blockDim.x = <specified x>;
blockDim.y = <specified y>;
gridDim.x = 16;
gridDim.y = 16;
```

`blockDim.z` and `gridDim.z` remain 1 through the original CUDA `dim3` default.
No kernel body, fragment operation, loop, input, copy, event, or cleanup source
is altered.

## Commands

Run from the repository root.

Generate:

```bash
src/cuda/accelwattch-ubench/scripts/generate_tensor_warp_variants.py
```

Validate and preserve the result:

```bash
src/cuda/accelwattch-ubench/scripts/validate_tensor_warp_variants.py \
  --output-json \
  src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/manifests/static_validation.json
```

Neither command invokes a compiler or requires a GPU or CUDA installation.

## Generated artifacts

```text
src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/
├── sources/
│   ├── TENSOR_T1.cu
│   ├── TENSOR_T2.cu
│   ├── TENSOR_T3.cu
│   └── TENSOR_T4.cu
├── diffs/
│   ├── TENSOR_T1.diff
│   ├── TENSOR_T2.diff
│   ├── TENSOR_T3.diff
│   └── TENSOR_T4.diff
└── manifests/
    ├── generation_manifest.json
    └── static_validation.json
```

There is intentionally no generated T0 source or T0 diff. Its manifest entry
has `source_kind=immutable_original`, the original repository-relative source
path, and the original source hash.

The generation manifest records:

- original source path and SHA-256;
- original Makefile SHA-256 as a preservation guard;
- variant specification path and SHA-256;
- generator version;
- block/grid and derived launch values;
- generated source paths and SHA-256 values;
- unified diff paths and SHA-256 values.

## Static validation

Static validation passed the following checks:

1. the original source and Makefile remained byte-for-byte unchanged before and
   after generation and validation;
2. T0 resolved exactly to the original source and hash;
3. all expected source patterns occurred exactly once;
4. T1-T4 matched independently reconstructed expected outputs containing only
   approved launch substitutions;
5. every stored unified diff matched the canonical reconstructed diff;
6. every grid was exactly `(16,16,1)`;
7. threads/block equaled the product of block dimensions;
8. every block contained a positive whole number of warps and no more than
   1,024 threads;
9. warps/block, block count, total threads, total warps, and relative requested
   work were arithmetically correct;
10. source, diff, specification, and manifest hashes matched;
11. two independent temporary generations were byte-for-byte identical;
12. an intentionally altered T4 grid was rejected; and
13. an attempted output path inside the original benchmark directory was
    rejected.

Result:

```text
status: passed
generator_version: task006-generator-v1
```

## Source hashes

The original hashes remained unchanged throughout Task006A:

```text
tensorcore.cu
02e382f4746e0216574faba8ed431461c553f041855aae4ce136f5835554eed2

TENSOR/Makefile
afca3279374d4fae7068a47fa9dc47b741ebd78745bf2a80b59c5d985105c9d8
```

Generated source hashes:

| Variant | SHA-256 |
| --- | --- |
| T1 | `a0c3c54b0da9c915b5c423d2a53d06cb46c33853dd1bdb120bf4daab906bf79e` |
| T2 | `3f4114e9af50744feafac7c623208ae00b2dbe82edf9d0d07305da509f5aecf5` |
| T3 | `dc03d131d6c4d9f079e7b9ebf309c22d2995f749361e68ac1558c25406be9894` |
| T4 | `f43854b0dd6fe42188e732b77c09f66535ebc7c1ec021265bb0955633c4fd52d` |

The manifest is authoritative if a future generator version intentionally
changes output formatting.

## Assumptions and limitations

- Exact source-pattern matching is intentional. An upstream source change
  causes a failure requiring review instead of a guessed transformation.
- Original comments describing `128x4` and 16 warps remain unchanged in T1-T4.
  Editing comments would add differences unrelated to execution semantics; the
  CSV and manifests are authoritative for variant geometry.
- Static source equality does not establish identical compiler output,
  registers, spills, SASS, occupancy, or runtime behavior.
- No generated source has been compiled or executed as part of Task006A.
- Numerical correctness and the original benchmark's overlapping C stores are
  unchanged and outside this source-generation proof.

## Task006A stopping point

Task006A is complete when the researcher reviews the variant specification,
canonical diffs, hashes, and passing static-validation result.

Task006B must not begin until that review approves Task006A. Build commands,
compiler capture, ptxas/resource parsing, binary hashes, and any build-time SASS
checks belong to Task006B. Power measurement and all later experimental stages
remain outside both the Task006A implementation and this review gate.
