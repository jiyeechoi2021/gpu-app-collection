# Original AccelWattch TENSOR Benchmark Characterization

## Scope and evidence boundary

This report reconstructs the execution of the unmodified benchmark in
`src/cuda/accelwattch-ubench/tensor_benchmarks/TENSOR/tensorcore.cu`. It is a
source-code characterization only. It does not use SASS, profiler counters,
runtime traces, or power measurements.

Statements about calls, launch geometry, ordering, memory sizes, and source
operations are **confirmed from source code**. Statements about dominant
hardware units and qualitative board-power importance are **plausible
hypotheses**. They require runtime measurement or profiling for confirmation.

The qualitative importance labels assume an external board-power observer and
mean expected contribution to GPU board activity during the stated stage:

- **negligible:** little or no GPU work is requested;
- **small:** short runtime-management or cleanup activity;
- **moderate:** substantive but bounded allocation, transfer, or conversion
  activity;
- **dominant:** expected to govern a sufficiently long run because its duration
  scales with the user-provided iteration count.

These labels are not energy estimates. Their effect on an average depends on
the actual measurement window and sampling interval. In particular, CPU-only
initialization can lengthen a whole-process window while the GPU is idle without
itself creating active GPU power.

## 1. Benchmark constants and invocation

The program requires exactly one command-line argument, parsed with `atoi()` as
the WMMA loop iteration count. The source performs no range or positivity check.
If the argument count is not two, it prints usage and exits before CUDA setup.

The fixed matrix dimensions are:

| Symbol | Value |
| --- | ---: |
| `MATRIX_M` | 1024 |
| `MATRIX_N` | 1024 |
| `MATRIX_K` | 1024 |
| `WMMA_M` | 16 |
| `WMMA_N` | 16 |
| `WMMA_K` | 16 |

The direct Makefile compiles `tensorcore.cu` with `nvcc`, optimization level
`-O3`, C++17, relaxed `constexpr`, and an `sm_${CUDA_SM}` target. It links
cuBLAS and cuRAND, but the benchmark invokes neither a cuBLAS operation nor a
cuRAND operation. A `curandGenerator_t gen` variable is declared but never
created or used. Random host inputs are produced by the C library `rand()`.

## 2. Host-side execution flow

### 2.1 Argument handling and declarations

1. Require one iteration-count argument.
2. Convert it to `int` with `atoi()`.
3. Declare device pointers, one host-result pointer, an unused cuRAND generator
   handle, and two CUDA event handles.

No GPU work is requested before the event creation.

**Board-power importance: negligible.** This is CPU-only control flow.

### 2.2 CUDA event initialization

The host creates `startWMMA` and `stopWMMA` with two calls to
`cudaEventCreate()`.

**Board-power importance: small.** Event creation is runtime setup, not a
compute workload. First CUDA-runtime interaction may also trigger context
initialization, but the exact runtime behavior is not established by source
alone.

### 2.3 Device memory allocation

The host performs six `cudaMalloc()` calls:

| Allocation | Type | Elements | Bytes | MiB |
| --- | --- | ---: | ---: | ---: |
| `a_fp32` | `float` | 1,048,576 | 4,194,304 | 4 |
| `b_fp32` | `float` | 1,048,576 | 4,194,304 | 4 |
| `a_fp16` | `half` | 1,048,576 | 2,097,152 | 2 |
| `b_fp16` | `half` | 1,048,576 | 2,097,152 | 2 |
| `c` | `float` | 1,048,576 | 4,194,304 | 4 |
| `c_wmma` | `float` | 1,048,576 | 4,194,304 | 4 |
| **Total** |  |  | **20,971,520** | **20** |

**Board-power importance: small.** Allocation may cause CUDA and memory-manager
activity but does not explicitly initialize all 20 MiB with a kernel or copy.
The exact cost is runtime-dependent.

### 2.4 Host memory allocation and initialization

The host allocates four 4 MiB buffers using `malloc()`:

- `c_host_wmma`, the final device-to-host destination;
- `a_fp32_h`, the FP32 A input;
- `b_fp32_h`, the FP32 B input;
- `c_h`, the FP32 accumulator input.

Total host allocation is 16 MiB. `RandomInit_fp()` then fills `a_fp32_h`,
`b_fp32_h`, and `c_h`, executing 3,145,728 calls to `rand()` and converting each
result to `float`. `c_host_wmma` is not initialized by the host.

**Board-power importance: negligible active GPU contribution.** This work is
CPU-only. Under a whole-process GPU-energy window it can contribute idle-board
residency and dilute average power, so its duration must not be confused with
the CUDA-event WMMA interval.

### 2.5 Host-to-device copies

Three synchronous API calls copy the initialized FP32 arrays to the GPU in this
source order:

1. `c_h` to `c`: 4 MiB;
2. `a_fp32_h` to `a_fp32`: 4 MiB;
3. `b_fp32_h` to `b_fp32`: 4 MiB.

Total explicit host-to-device traffic is 12 MiB.

**Board-power importance: moderate.** These bounded transfers exercise the
copy path, memory subsystem, and DRAM, but their work does not grow with the
WMMA iteration count.

### 2.6 FP32-to-FP16 conversion launches

The host launches `convertFp32ToFp16` twice in the default stream:

1. convert `a_fp32` into `a_fp16`;
2. convert `b_fp32` into `b_fp16`.

There is no explicit launch-error check or synchronization between the two
launch statements. Default-stream ordering makes the second launch follow the
first. Each launch converts 1,048,576 elements.

**Board-power importance: moderate.** The two full-array conversion kernels
exercise general CUDA cores and global memory. They are finite preparation work
outside the WMMA event interval.

### 2.7 Device-to-device initialization of the accumulator

The host calls `cudaMemcpy()` to copy 4 MiB from `c` to `c_wmma`. Because it is
issued after both conversion kernels in the default stream, the later WMMA work
cannot pass the conversions or this copy. The source contains no independent
stream.

**Board-power importance: moderate.** This is a bounded device-memory transfer
outside the WMMA event interval.

### 2.8 Launch-geometry calculation and console output

The host prints `M`, `N`, and `K`, defines a block of `(128, 4, 1)`, and computes
the grid:

```text
grid.x = ceil(1024 / (16 * 128 / 32)) = ceil(1024 / 64) = 16
grid.y = ceil(1024 / (16 * 4))        = ceil(1024 / 64) = 16
```

The resulting launch has 256 blocks, 512 threads or 16 warps per block,
131,072 threads, and 4,096 launched warps.

**Board-power importance: negligible.** Geometry calculation and printing are
host work.

### 2.9 WMMA timing and launch

The host enqueues these operations in order:

1. record `startWMMA`;
2. launch `wmma_example<<<(16,16,1),(128,4,1)>>>`;
3. record `stopWMMA`.

There is no explicit `cudaDeviceSynchronize()` and no immediate kernel launch
error query. Both events and the kernel use the default stream, so the event
timestamps bracket the `wmma_example` launch in stream order. The start event is
after all preparation work; the stop event is before the result copy.

**Board-power importance: dominant for sufficiently large positive iteration
counts.** This is the only stage whose instruction work scales directly with
the argument `iterations`. The label is a hypothesis about duration and board
activity, not an estimate of Tensor energy. At zero or very small iteration
counts, setup, fragment loads/stores, and other stages can instead be material.

### 2.10 Device-to-host result copy and synchronization

After recording `stopWMMA`, the host calls synchronous `cudaMemcpy()` to copy
the complete 4 MiB `c_wmma` allocation into `c_host_wmma`. Default-stream
ordering places this copy after the stop event. The host cannot proceed past the
synchronous result copy until the relevant transfer completes; consequently,
the preceding default-stream WMMA launch and stop event have completed before
the later timing query.

The copied result is not checked, printed, or otherwise validated.

**Board-power importance: moderate.** The transfer exercises device memory and
the device-to-host copy path, but it is outside the CUDA-event interval and does
not scale with `iterations`.

### 2.11 Elapsed-time query

The host calls `cudaEventElapsedTime(&wmmaTime, startWMMA, stopWMMA)` and prints
`wmma took ... ms`. The reported value is the elapsed default-stream time
between the two event records. It excludes the earlier allocations, host input
generation, host-to-device copies, conversion kernels, device-to-device copy,
and the later device-to-host copy.

It includes all work inside `wmma_example`: its fragment setup and loads, loop,
and fragment store. It is therefore a whole-`wmma_example` kernel time, not a
measurement of `mma_sync` alone.

**Board-power importance: negligible.** This is a host-side timing query after
the measured kernel has completed.

### 2.12 Cleanup

The host then:

1. destroys both CUDA events;
2. frees all six device allocations;
3. frees `c_host_wmma`;
4. calls `cudaDeviceReset()`;
5. returns success.

The source does **not** free `a_fp32_h`, `b_fp32_h`, or `c_h`; the operating
system reclaims those host allocations at process exit. The reset destroys the
CUDA context after device work has completed.

**Board-power importance: small.** Device frees and context reset can cause
short runtime and memory-manager activity, but no benchmark compute remains.

## 3. Complete CUDA kernel inventory

There are two distinct kernel functions and three launches.

### 3.1 `convertFp32ToFp16` — A conversion

| Property | Source-confirmed characterization |
| --- | --- |
| Purpose | Convert `a_fp32` to `a_fp16`, one element per in-range thread |
| Input/output | Read 1,048,576 FP32 values; write 1,048,576 FP16 values |
| Grid dimensions | `(4096, 1, 1)` because `(1,048,576 + 255) / 256 = 4096` |
| Block dimensions | `(256, 1, 1)` |
| Threads/warps | 1,048,576 threads; 32,768 launched warps |
| Dynamic shared memory | 0 bytes; no third launch-configuration argument |
| Source-declared static shared memory | None |
| Expected dominant hardware units | Plausible: global load/store paths, conversion/general arithmetic pipeline, caches, and DRAM |
| WMMA use | None |
| Relative power importance | Moderate preparation activity |

Every launched thread is in range because the element count is exactly
divisible by 256. Each thread calculates a linear index, loads one `float`,
converts it through assignment to `half`, and stores one result.

### 3.2 `convertFp32ToFp16` — B conversion

| Property | Source-confirmed characterization |
| --- | --- |
| Purpose | Convert `b_fp32` to `b_fp16`, one element per in-range thread |
| Input/output | Read 1,048,576 FP32 values; write 1,048,576 FP16 values |
| Grid dimensions | `(4096, 1, 1)` |
| Block dimensions | `(256, 1, 1)` |
| Threads/warps | 1,048,576 threads; 32,768 launched warps |
| Dynamic shared memory | 0 bytes |
| Source-declared static shared memory | None |
| Expected dominant hardware units | Plausible: global load/store paths, conversion/general arithmetic pipeline, caches, and DRAM |
| WMMA use | None |
| Relative power importance | Moderate preparation activity |

This is the same kernel and geometry as the A conversion, with different input
and output pointers.

### 3.3 `wmma_example` — Tensor workload

| Property | Source-confirmed characterization |
| --- | --- |
| Purpose | Load one A fragment, one B fragment, and one C fragment per warp; repeat warp-level matrix multiply-accumulate `iterations` times; store the accumulator fragment |
| Grid dimensions | `(16, 16, 1)` = 256 blocks |
| Block dimensions | `(128, 4, 1)` = 512 threads = 16 warps |
| Total launch | 131,072 threads = 4,096 warps |
| Dynamic shared memory | 0 bytes; no third launch-configuration argument |
| Source-declared static shared memory | None |
| Expected dominant hardware units | Plausible during loads/store: global load/store paths, caches, register file; during loop: Tensor Core/HMMA execution paths, register file, warp schedulers, instruction issue/fetch, and clock/control distribution |
| Relative power importance | Dominant for a sufficiently long positive iteration count |

The kernel does not read `blockIdx`, `threadIdx`, a warp ID, or any derived
index. Consequently, all 4,096 warps use the same base pointers for A, B, and C.
Each warp loads and operates on the same logical leading fragments and stores
to the same base C tile. The launch geometry therefore replicates work rather
than assigning distinct output tiles in the implemented kernel body. The final
stores overlap without inter-warp or inter-block ordering, so output correctness
is not established by the source.

Although the source declares no shared memory, exact compiler-generated local,
register, spill, or implicit instruction behavior is SASS-dependent and cannot
be confirmed from this source-only analysis.

## 4. Tensor kernel boundaries

### 4.1 Whole-kernel boundary

The Tensor workload kernel begins when the default stream executes the
`wmma_example` launch after `startWMMA` and ends when that kernel completes
before `stopWMMA` is timestamped. The CUDA events therefore time the entire
kernel, not only its loop.

### 4.2 WMMA operations before the Tensor loop

Inside each warp of `wmma_example`, WMMA API use begins with fragment
declarations and the three collective loads:

1. declare FP16 column-major `a_frag`;
2. declare FP16 column-major `b_frag`;
3. declare FP32 accumulator `c_frag`;
4. `load_matrix_sync(a_frag, a, lda)`;
5. `load_matrix_sync(b_frag, b, ldb)`;
6. `load_matrix_sync(c_frag, c, ldc, mem_col_major)`.

These fragment loads are WMMA-related setup but are outside the repeated Tensor
multiply-accumulate loop. Logically, each warp requests a 16x16 FP16 A tile, a
16x16 FP16 B tile, and a 16x16 FP32 C tile. Their exact machine-instruction
sequence and cache/DRAM behavior are SASS- and runtime-dependent.

**Relative power importance: small to moderate for a long loop; potentially
material for very small iteration counts.** This is a hypothesis based on the
loads occurring once per warp rather than once per iteration.

### 4.3 Repeated Tensor loop

The repeated Tensor work begins at the first execution of:

```text
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)
```

and ends after the last executed loop iteration. The loop runs exactly
`iterations` times when the parsed value is positive. Each call uses `c_frag`
as both accumulator input and output, forming a loop-carried dependency within
each warp. A `#pragma unroll 100` precedes the runtime-count loop, but the actual
unrolling, HMMA instruction count, register allocation, and scheduling require
compiler or SASS evidence and are not established here.

**Relative power importance: dominant for sufficiently large positive
`iterations`.** Expected Tensor execution, register access, scheduling, and
instruction issue repeat with the loop. No energy per WMMA, HMMA, or FLOP is
estimated.

### 4.4 Work after the Tensor loop but inside the kernel

After the loop, every warp executes:

```text
wmma::store_matrix_sync(c, c_frag, ldc, wmma::mem_col_major)
```

This ends the kernel's WMMA work by storing the accumulator fragment to global
memory. Because every warp receives the same `c` pointer and no index offset,
all stores target the same logical base tile.

**Relative power importance: small to moderate for a long loop; potentially
material for very small iteration counts.** It is a one-time-per-warp store,
not iteration-scaled work.

### 4.5 Work outside the Tensor kernel

The following GPU-related work is outside both the `mma_sync` loop and the
CUDA-event WMMA timing interval:

- CUDA context/runtime and event setup;
- six device allocations;
- 12 MiB of host-to-device FP32 copies;
- two FP32-to-FP16 conversion kernel launches;
- the 4 MiB device-to-device C initialization;
- the 4 MiB device-to-host result copy;
- event destruction, device frees, and device reset.

Within the event interval but outside the repeated `mma_sync` loop are the three
fragment loads and the final fragment store.

## 5. Execution timeline

```text
Host: parse iteration count
  |
  v
Create CUDA events
  |
  v
cudaMalloc x6 (20 MiB device memory)
  |
  v
Host malloc x4 (16 MiB) + CPU random initialization of A, B, C
  |
  v
cudaMemcpy H->D: C (4 MiB)
  |
  v
cudaMemcpy H->D: A FP32 (4 MiB)
  |
  v
cudaMemcpy H->D: B FP32 (4 MiB)
  |
  v
convertFp32ToFp16(A): <<<4096, 256>>>
  |
  v
convertFp32ToFp16(B): <<<4096, 256>>>
  |
  v
cudaMemcpy D->D: C to C_WMMA (4 MiB)
  |
  v
Print dimensions and calculate grid/block geometry
  |
  v
Record startWMMA event                         <--- CUDA-event interval begins
  |
  v
wmma_example: <<<(16,16,1), (128,4,1)>>>
  |
  +--> Per warp: load A fragment               [inside timed kernel, outside loop]
  +--> Per warp: load B fragment               [inside timed kernel, outside loop]
  +--> Per warp: load C fragment               [inside timed kernel, outside loop]
  +--> Repeat mma_sync `iterations` times      [Tensor loop]
  +--> Per warp: store C fragment              [inside timed kernel, outside loop]
  |
  v
Record stopWMMA event                          <--- CUDA-event interval ends
  |
  v
cudaMemcpy D->H: C_WMMA (4 MiB; host-blocking completion point)
  |
  v
Query and print CUDA-event elapsed time
  |
  v
Destroy events -> cudaFree x6 -> free result host buffer
  |
  v
cudaDeviceReset -> process exit
```

All CUDA operations shown use the default stream because the source creates no
other stream and supplies no stream argument. The two kernel launch statements
are asynchronous with respect to immediate host continuation. The synchronous
copy APIs and final device-to-host result copy provide the relevant host-side
completion behavior; there is no explicit `cudaDeviceSynchronize()`.

## 6. Stage-by-stage qualitative board-power assessment

| Stage | GPU work requested | Importance | Evidence status and rationale |
| --- | --- | --- | --- |
| Argument parsing and declarations | None | Negligible | Source-confirmed CPU work |
| CUDA event/context setup | Runtime setup | Small | Source-confirmed calls; exact context cost unresolved |
| Six `cudaMalloc` calls | Device memory management | Small | Source-confirmed; runtime cost unresolved |
| Host allocation/random initialization | None directly | Negligible | CPU-only; may extend an external process window at idle board power |
| Three H→D copies, 12 MiB | Copy engine/memory traffic | Moderate | Bounded source-confirmed transfers |
| A FP32→FP16 conversion | 1,048,576 element conversions | Moderate | Bounded kernel; hardware mix is a profiling hypothesis |
| B FP32→FP16 conversion | 1,048,576 element conversions | Moderate | Same as A conversion |
| C D→D copy, 4 MiB | Device memory transfer | Moderate | Bounded source-confirmed transfer |
| Geometry calculation/printing | None directly | Negligible | CPU-only |
| WMMA fragment loads | Three collective loads per launched warp | Small–moderate | Once per warp; relative weight depends on iterations/cache behavior |
| Repeated `mma_sync` loop | Iteration-scaled warp-level MMA work | Dominant for long positive runs | Source confirms scaling; actual HMMA lowering/activity needs SASS/profile evidence |
| WMMA fragment store | One collective store per launched warp | Small–moderate | Once per warp; overlapping target addresses |
| C D→H copy, 4 MiB | Copy engine/memory traffic | Moderate | Outside CUDA-event interval but inside whole-process observation |
| Event timing query/printing | Minimal runtime/CPU work | Negligible | Occurs after completion |
| Event/device cleanup/reset | Runtime/memory management | Small | Short teardown expected; runtime-dependent |

The categories are not additive and do not quantify joules. A sampler restricted
to the CUDA-event interval observes only `wmma_example`, including its fragment
loads, repeated loop, and store. A whole-process sampler can observe every GPU
stage and idle periods during CPU work. Those measurement boundaries must not
be compared as though they represented the same quantity.

## 7. Confirmed facts, hypotheses, and unresolved items

### Confirmed from source code

- The program launches two conversion kernels and one WMMA kernel.
- Both conversion launches use 4,096 blocks of 256 threads.
- The WMMA launch uses a 16x16 grid and a 128x4 block: 16 warps per block and
  4,096 total launched warps.
- `wmma_example` contains no thread, warp, or block indexing.
- Each warp loads A, B, and C fragments once, executes `mma_sync` in the loop,
  and stores C once.
- The `mma_sync` loop contains no explicit source-level memory operation other
  than fragment/register semantics of the intrinsic.
- The CUDA events bracket the whole `wmma_example` kernel but not preparation or
  the final device-to-host copy.
- No dynamic or source-declared static shared memory is used by either kernel.
- The copied result is not validated.
- The original host input buffers are not explicitly freed.

### Plausible hypotheses requiring profiling or runtime evidence

- The long loop primarily drives Tensor execution paths, registers, schedulers,
  and instruction issue.
- The conversion kernels primarily drive global memory and conversion/general
  arithmetic paths.
- `wmma_example` dominates board activity when the iteration count makes its
  duration large relative to preparation and teardown.
- Identical fragment addresses may create substantial cache reuse and
  overlapping-store behavior.

### Unresolved from source alone

- The number and form of generated HMMA/SASS instructions per `mma_sync`.
- Actual compiler unrolling of the runtime-count loop.
- Register count, occupancy, spills, issue rate, and dependency stalls.
- Cache hit rates, DRAM transactions, and effective conversion bandwidth.
- Actual clocks, thermal state, throttling, and board power.
- The duration at which the repeated WMMA loop becomes dominant for the
  measurement system in use.
- Whether any observed output is numerically meaningful in the presence of
  overlapping unsynchronized stores.

No intrinsic Tensor Core energy or effective energy per HMMA/FLOP can be
derived from this source-only timeline.
