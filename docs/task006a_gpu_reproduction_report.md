# Task006A GPU Reproduction Report

- Old commit: `9c97dc08cd5501099abc308018ec8fbdbacae031`
- New commit: `f97f83391e2d093fd4157721cb626ee0e9220c53`
- Branch: `dev`
- Remote: `origin` -> `https://github.com/jiyeechoi2021/gpu-app-collection.git`
- Update method: `git fetch origin`, then `git merge --ff-only origin/dev`
- Required maintained files: all 8 present
- Generator status: PASSED
- Generator version: `task006-generator-v1`
- Static validation status: PASSED
- Original source SHA-256: `02e382f4746e0216574faba8ed431461c553f041855aae4ce136f5835554eed2` (MATCH)
- Original Makefile SHA-256: `afca3279374d4fae7068a47fa9dc47b741ebd78745bf2a80b59c5d985105c9d8` (MATCH)

## Generated artifacts

- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/sources/TENSOR_T1.cu`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/sources/TENSOR_T2.cu`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/sources/TENSOR_T3.cu`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/sources/TENSOR_T4.cu`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/diffs/TENSOR_T1.diff`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/diffs/TENSOR_T2.diff`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/diffs/TENSOR_T3.diff`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/diffs/TENSOR_T4.diff`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/manifests/generation_manifest.json`
- `src/cuda/accelwattch-ubench/generated/tensor_warp_sweep/manifests/static_validation.json`

- All 10 generated artifacts checked and ignored: YES
- `NOT_IGNORED` artifacts found: NO
- Repository-maintained files modified: NO
- Final git status: clean
- Ready for Task006B code update: YES

Task006B was not started. No build, `nvcc`, `ptxas`, `cuobjdump`, NCU, or
benchmark was executed.
