# Task 005: Execution-Context Variable Decomposition fo[text](task005_execution_context_variable_decomposition.md)r the TENSOR Warp-Count Study

## Scope and scientific question

This report analyzes the planned T0-T4 configurations of the original
AccelWattch `TENSOR` benchmark. It does not modify or generate source, build
scripts, or pipeline code.

The immediate question is:

> Does the fixed-grid T0-T4 sweep isolate warps per block, or does changing the
> block size also change other execution properties that prevent a pure
> warps-per-block interpretation?

The answer is that it does **not** isolate warps per block. With a fixed 256
block grid, reducing warps per block necessarily reduces total launched
threads, total launched warps, fragment operations, dynamically requested
warp-level WMMA calls, and overlapping stores. It may also change residency,
occupancy, scheduling, achieved Tensor throughput, clocks, temperature, runtime,
and board power. The correct interpretation is a coupled warp-supply sweep.

No quantity in this report is an estimate of pure Tensor Core energy.

## Sequential Experimental Logic

The study proceeds through three distinct roles:

- **Characterization is the reference stage.** Its purpose is to establish the
  original TENSOR execution flow, measurement boundary, source-level behavior,
  and unresolved runtime properties. It supplies the reference needed to state
  a controlled hypothesis but is not itself a variant experiment.
- **Experiment A is the first hypothesis test.** It is the fixed-grid coupled
  block-and-global warp-supply sweep. Its purpose is to test whether reducing
  the original benchmark's redundant warp supply and total replicated warp work
  produces a reproducible change in runtime, board power, board energy, and
  execution activity.
- **Experiment B is a conditional mechanism test.** It holds total launched
  warp work constant while changing block packaging. It should be performed
  only if Experiment A shows a meaningful and reproducible response and block
  packaging versus total work remains an unresolved mechanism.

```text
Characterization
    ↓
Experiment A: Hypothesis Test
    ↓
Review validity and effect size
    ↓
Experiment B: Mechanism Test, only if justified
```

Experiment B is contingent upon the outcome of Experiment A. It must not be
implemented merely because it is technically feasible or already specified in
this report. Researcher review must first determine that Experiment A's change
is significant in the scientific sense relevant to the study, reproducible,
valid, and in need of mechanistic decomposition.

## Evidence classes used in this report

- **Confirmed from source:** directly established by the original kernel or the
  stated variant transformations.
- **Analytically derived:** arithmetic consequence of the fixed grid, block
  dimensions, and positive iteration count.
- **Compiler/SASS dependent:** requires compiled resource reports or machine
  code to establish.
- **Runtime dependent:** requires execution, profiling, or power telemetry.
- **Hypothesis:** a scientifically plausible relationship to be tested, not a
  measured fact.

## 1. Source-level controlled variable

### 1.1 Intended source-level change

The intended controlled quantity is the number of threads, equivalently whole
warps, in each `wmma_example` thread block. It is set through the host-side
`dim3 blockDim` assignments used for that kernel launch:

| Variant | `blockDim.x` | `blockDim.y` | Threads/block | Warps/block |
| --- | ---: | ---: | ---: | ---: |
| T0 | 128 | 4 | 512 | 16 |
| T1 | 128 | 2 | 256 | 8 |
| T2 | 128 | 1 | 128 | 4 |
| T3 | 64 | 1 | 64 | 2 |
| T4 | 32 | 1 | 32 | 1 |

The block shape is itself coupled to the intended quantity: T0-T2 reduce
`blockDim.y`, while T2-T4 reduce `blockDim.x`. Thus even at source level the
sweep is not five values of one scalar parameter encoded through an otherwise
identical block shape. It changes the two-dimensional launch shape as necessary
to realize 16, 8, 4, 2, and 1 whole warps.

Because `wmma_example` does not read `threadIdx`, `blockIdx`, `blockDim`, or
`gridDim`, the block-coordinate changes do not change per-warp addresses or
branches in the kernel source. They change how many identical warp instances
are launched and how those warps are packaged into blocks.

### 1.2 Source-identity audit

| Component | Remains source-identical? | Evidence and qualification |
| --- | --- | --- |
| WMMA fragment declarations | Yes | The A, B, and accumulator fragment types and shapes remain unchanged. |
| A/B/C fragment loads | Yes | The three `load_matrix_sync` calls, pointers, leading dimensions, layouts, and fragment types remain unchanged. |
| `mma_sync` loop body | Yes | The loop body remains one `wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)` call. |
| Iteration count | Yes, for matched runs | The same runtime argument is passed to every variant. Run manifests must verify equality. |
| Unroll directive | Yes | `#pragma unroll 100` remains unchanged. Actual compiler unrolling still requires SASS evidence. |
| Fragment store | Yes | The same `store_matrix_sync` call, pointer, leading dimension, and layout remain unchanged. |
| Input matrices | Yes, structurally | Dimensions, allocations, initialization procedure, and input pointers remain identical. Exact random values should also match because the program uses the same deterministic unseeded `rand()` call sequence under the same C runtime, but input hashes are needed to prove byte identity. |
| Grid dimensions | Same runtime value, not necessarily source-text identical | Every planned launch is fixed at `(16,16,1)`. T0 derives this with formulas in the original source; the planned generated variants pin it to 16 to prevent formula reevaluation after changing `blockDim`. |
| Memory addresses used by each warp | Yes at kernel-source semantics | No thread/block index offsets are applied. Every launched warp receives the same A, B, and C base pointers and uses the same leading dimensions. Physical/cache behavior is runtime-dependent. |
| Host-side setup and cleanup | Yes except launch-geometry assignments | Allocation, initialization, copies, events, result copy, frees, and reset remain unchanged. The intended block assignments and required fixed-grid assignments differ. |
| Conversion kernels | Yes | Kernel body, two launches, grid `(4096,1,1)`, block `(256,1,1)`, inputs, and outputs remain unchanged. |
| CUDA-event timing boundary | Yes | `startWMMA` remains immediately before `wmma_example`; `stopWMMA` remains immediately after it in default-stream order. |

The source-identical loop does not guarantee identical SASS across separately
compiled variants. Launch-bound knowledge, compiler decisions, register count,
instruction layout, and metadata must be checked rather than assumed.

## 2. Variables that necessarily change with warps per block

Let:

- `I` be the positive runtime iteration count;
- `B = 16 * 16 = 256` blocks;
- `W_b` be warps per block;
- `W = B * W_b` be total launched warps.

Every warp collectively executes three source-level fragment loads, one
fragment store, and `I` source-level `mma_sync` calls. “WMMA calls” below means
dynamically requested warp-level executions of the source intrinsic, not the
number of generated HMMA instructions. The latter is not known without SASS or
profiler evidence.

| Variant | Threads/block | Warps/block | Blocks | Total threads | Total warps | Collective fragment loads (`3W`) | Collective fragment stores (`W`) | Warp-level `mma_sync` calls (`W*I`) | Requested work relative to T0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| T0 | 512 | 16 | 256 | 131,072 | 4,096 | 12,288 | 4,096 | `4,096I` | 1 = 100% |
| T1 | 256 | 8 | 256 | 65,536 | 2,048 | 6,144 | 2,048 | `2,048I` | 1/2 = 50% |
| T2 | 128 | 4 | 256 | 32,768 | 1,024 | 3,072 | 1,024 | `1,024I` | 1/4 = 25% |
| T3 | 64 | 2 | 256 | 16,384 | 512 | 1,536 | 512 | `512I` | 1/8 = 12.5% |
| T4 | 32 | 1 | 256 | 8,192 | 256 | 768 | 256 | `256I` | 1/16 = 6.25% |

“Requested work relative to T0” refers to warp-replicated source work inside
`wmma_example`: fragment operations and positive-iteration loop calls. It does
not mean useful matrix work, because all warps use the same base addresses and
do not partition the matrices. It also does not include the unchanged host
setup, conversion kernels, copies, or cleanup.

Keeping the grid fixed therefore changes two primary quantities together:

1. **warps per block**, from 16 to 1; and
2. **total launched warp work**, from 4,096 to 256 warps, including a 16-fold
   reduction in dynamically requested loop calls and one-time fragment work.

It also necessarily changes total launched threads and the number of
overlapping fragment stores to the common C tile. These are analytical facts,
not runtime hypotheses.

If `I <= 0`, the loop executes no source-level `mma_sync` calls and the relative
loop-work interpretation is inapplicable; fragment loads and stores still
execute. Study runs must therefore require and record a positive matched `I`.

## 3. Runtime execution-context variables likely to change

The classifications below apply across T0-T4 under a matched build environment,
GPU, iteration count, and device policy. “Necessarily changes” is reserved for
properties forced by launch arithmetic or source semantics. Exact achieved
values remain measurements unless stated otherwise.

| Execution property | Classification | Analysis |
| --- | --- | --- |
| Resident blocks per SM | **May change**; exact value cannot be determined without build/SASS and device-limit evidence | Smaller blocks consume fewer threads and, if registers/thread are comparable, fewer registers/block, potentially allowing more blocks to reside. The architectural resident-block cap, register allocation, and allocation granularity can create plateaus. |
| Resident warps per SM | **May change** | More resident blocks do not guarantee more resident warps: block size falls as possible block residency rises. Thread, register, block, and warp caps jointly determine theoretical residency. |
| Achieved occupancy | **May change**; runtime evidence required | It depends on theoretical limits plus scheduling, kernel duration, tail effects, and measurement definition. It cannot be inferred from total launched warps alone. |
| Active warp supply | **Necessarily changes globally; may change per SM over time** | Total launched supply falls exactly 4096→256. Instantaneous active/resident warps per SM depend on allocation limits, SM count, waves, and duration. |
| Scheduler eligibility | **May change** | Fewer resident warps can reduce choices for each scheduler, while dependency stalls in the accumulator chain affect eligibility. Exact eligible-warps behavior requires profiling. |
| Issue-active behavior | **May change** | It can fall if fewer eligible warps expose dependency latency, remain saturated over a range, or change with instruction/front-end effects. NCU evidence is needed. |
| Latency hiding | **May change** | Each warp has a loop-carried accumulator dependency. The ability to cover it with other warps depends on resident and eligible warp supply, not only block size. |
| Tensor instruction throughput | **May change** | Total requested loop calls necessarily change; achieved HMMA/s may remain near saturation for some variants and fall below a warp-supply threshold. Exact lowering and throughput require SASS plus runtime/profile evidence. |
| Register-file activity | **Necessarily changes in total requested warp work; achieved rate may change** | Fewer warps request fragment operations and loop iterations. Per-warp fragment semantics remain identical, but registers/thread, spills, bank behavior, and activity rate are compiler/runtime dependent. |
| Instruction-fetch/control activity | **Necessarily changes in aggregate requested dynamic warp work; achieved rate may change** | Fewer warps execute the same source instructions. Static code should be similar, but generated instruction sequence, unrolling, launch/block management, and fetch rate need SASS/runtime evidence. |
| Cache and DRAM traffic | **May change** | Logical collective loads/stores necessarily scale with total warps, but all warps reuse common addresses. Cache hits, request merging, evictions, write behavior, and physical DRAM traffic cannot be derived from logical operation counts. |
| Overlapping stores to common C address | **Necessarily changes in count; runtime interaction may change** | Collective stores fall 4096→256 and target the same logical tile. Ordering, overlap, cache write behavior, and observed traffic are runtime-dependent. |
| Kernel duration | **May change** | Requested work falls 16-fold, but achieved throughput, launch overhead, wave count, clock, and fixed fragment setup/store costs also matter. Duration must be measured by the unchanged CUDA-event boundary. |
| SM clock and power state | **May change** | Different activity and duration can produce different boost/P-state behavior even under the same configured policy. They must be sampled in power runs. |
| Thermal state | **May change** | Power, duration, run order, initial temperature, fan policy, and cooldown interact. Randomization/interleaving and temperature records are required. |

### 3.1 Properties expected to remain approximately constant by design

The following should remain approximately constant if build and run validation
succeeds:

- per-warp source instruction and dependency structure;
- per-warp logical A, B, and C fragment shapes and addresses;
- positive loop iteration count;
- static matrix allocations and host/device preparation work;
- conversion-kernel work;
- event timing boundary;
- GPU identity, configured power limit, clock policy, and software stack;
- per-thread register demand, **only if compiled SASS/resource reports confirm
  it**.

“Approximately constant” is not “proven equal.” Inputs, binaries, build flags,
device policy, initial thermal state, and relevant metrics must be recorded.

### 3.2 Properties that cannot be determined from source arithmetic

Exact registers/thread, spills, theoretical occupancy, resident blocks/warps,
generated HMMA count per source call, achieved occupancy, eligible warps,
issue-active rate, HMMA/s, cache/DRAM transactions, clocks, temperature, runtime,
power, and energy cannot be established from the sweep table alone.

## 4. Confounding analysis

### 4.1 What the fixed-grid sweep does and does not isolate

| Candidate interpretation | Is it isolated? | Reason |
| --- | --- | --- |
| A. Warps per block | **No** | Total launched warps and work change in direct proportion, while block shape and block resource packaging also change. |
| B. Total launched warps | **No** | Total launched warps change, but so do warps/block, threads/block, block resource footprint, possible residency, and scheduling context. |
| C. Active warps per SM | **No** | Global warp supply is known, but instantaneous active/resident warps per SM require runtime evidence and may plateau or vary nonlinearly. |
| D. Tensor throughput | **No** | Requested source calls change, but achieved HMMA/s is an outcome affected by residency, eligibility, latency hiding, clocks, and generated SASS. |
| E. Combined warp-supply effect | **Yes, as the experimental treatment** | The sweep deliberately reduces block-local and grid-wide warp supply together while preserving per-warp source work and a fixed block grid. |

The scientifically correct name for the first sweep is:

> **Fixed-grid coupled block-and-global warp-supply sweep**

A shorter acceptable label is **fixed-grid warp-supply sweep**. It must not be
reported as a pure warps-per-block experiment. Its treatment is the coupled
reduction in warps/block and total replicated warp work; residency, throughput,
clock, thermal, and power responses are outcomes or mediators.

## 5. Two-stage experiment design

### 5.1 Experiment A: fixed-grid coupled warp-supply sweep

**Design**

- Grid remains `(16,16,1)`, or 256 blocks.
- Blocks contain 16, 8, 4, 2, or 1 warp.
- Total launched warps and total replicated work are allowed to fall from 4,096
  to 256.
- Per-warp kernel source, memory addresses, dependency chain, fragment
  operations, and positive iteration count remain matched.

**Hypothesis that Experiment A can test**

> Under a fixed 256-block grid and matched per-warp WMMA work, reducing the
> combined block-local and global supply of redundant WMMA warps changes
> measured WMMA-kernel runtime, achieved execution activity, and uninstrumented
> V100 board-power behavior.

This tests whether the original benchmark's broad redundant warp supply is
associated with its measured power behavior and where saturation or scheduling
thresholds appear. It does not determine whether the cause is warps/block,
total work, resident warps, achieved Tensor throughput, or their interaction.

**Primary responses**

- CUDA-event `wmma_example` runtime from unprofiled power executions;
- sampled V100 board power and explicitly bounded board energy from unprofiled
  executions;
- clock and temperature during those executions;
- separate explanatory profile metrics when NCU is available.

### 5.2 Experiment B: constant-total-warp block-packaging sweep

**Design**

- Change block size exactly as in T0-T4.
- Increase block count so total launched warps remain 4,096.
- Preserve per-warp source operations, addresses, and iteration count.
- Treat grid/block count as an intentional coupled design variable, not as an
  unchanged control.

Required total blocks are:

```text
blocks = 4096 total warps / warps_per_block
```

Candidate balanced two-dimensional grids are:

| Variant-equivalent block shape | Warps/block | Required blocks | Candidate grid | Total warps | Total threads |
| --- | ---: | ---: | --- | ---: | ---: |
| `(128,4,1)` | 16 | 256 | `(16,16,1)` | 4,096 | 131,072 |
| `(128,2,1)` | 8 | 512 | `(16,32,1)` or `(32,16,1)` | 4,096 | 131,072 |
| `(128,1,1)` | 4 | 1,024 | `(32,32,1)` | 4,096 | 131,072 |
| `(64,1,1)` | 2 | 2,048 | `(32,64,1)` or `(64,32,1)` | 4,096 | 131,072 |
| `(32,1,1)` | 1 | 4,096 | `(64,64,1)` | 4,096 | 131,072 |

These launch sizes are arithmetically feasible with ordinary CUDA grid limits,
and the current kernel does not index the grid, so added blocks continue the
same redundant per-warp work. Technical feasibility still requires a build and
smoke validation on the target V100. Exact two-dimensional orientation should
be fixed before generation; because the kernel ignores indices, orientation is
not expected to alter kernel-source semantics, but that expectation remains to
be validated rather than varied within one experiment.

Experiment B holds constant:

- total launched threads and warps;
- total source-level fragment loads and stores;
- total warp-level `mma_sync` calls at matched `I`;
- per-warp addresses and dependency structure.

It necessarily changes:

- warps and threads per block;
- total block count and grid dimensions;
- block scheduling/allocation events and tail-wave structure;
- possible resident blocks/warps and achieved occupancy.

**Hypothesis that Experiment B can test**

> With total launched warp work held constant at 4,096 warps, changing how
> those identical warps are packaged into blocks changes residency, scheduler
> supply, achieved Tensor throughput, runtime, and V100 board-power behavior.

Experiment B is closer to a block-packaging or block-local warp-supply test. It
still does not directly isolate active warps per SM, because block count,
resource allocation granularity, scheduling waves, and residency can change.

### 5.3 Why both experiments are needed

Experiment A asks whether reducing the original benchmark's overall redundant
warp supply changes behavior. Experiment B asks whether block packaging matters
when the total requested warp work is held constant. Comparing their patterns
can narrow the explanation:

- a strong A effect but weak B effect supports total global work/supply as the
  larger association;
- effects in both suggest block-local residency/scheduling context also matters;
- neither result alone proves a causal mechanism without the corresponding
  execution metrics.

The experiments must remain separate datasets with distinct design labels.

## 6. Expected power interpretation

All interpretations below require a declared boundary. Mean sampled process-
window board power, kernel-window board power, process-window energy, and
CUDA-event kernel time are different responses. Power runs must remain
uninstrumented; NCU results are explanatory evidence from separate executions.

### 6.1 Board power decreases monotonically from T0 to T4

**Can conclude after validation:** measured board power is monotonically
associated with the coupled reduction in block-local and global warp supply
under Experiment A's conditions.

**Cannot conclude:** power fell specifically because of warps/block, active
warps/SM, or Tensor Core activity. Total requested work also fell, and clocks,
duration, temperature, issue behavior, and memory activity may mediate the
result. It is not evidence of intrinsic Tensor energy per operation.

### 6.2 Board power remains nearly constant

**Can conclude after defining “nearly”:** over this sweep and measurement
boundary, reducing requested warp supply did not produce a resolved change in
the chosen board-power statistic.

**Possible hypotheses:** all variants maintain enough instantaneous work to
saturate the relevant execution/power state; the board remains in the same
clock/P-state; fixed/common board contributions dominate; or the sampler lacks
temporal resolution. Runtime and energy may still differ substantially.

**Cannot conclude:** warps do not affect execution, Tensor throughput is equal,
or energy per operation is constant.

### 6.3 Board power decreases but kernel runtime increases

**Can conclude after validation:** lower instantaneous board power coincides
with longer measured kernel duration for some lower-supply variants.

This is consistent with reduced latency hiding or throughput, but that mechanism
requires occupancy, eligible-warp, issue-active, and HMMA/s evidence. Integrated
energy can decrease, remain constant, or increase because energy depends on the
power-time trajectory. It must be calculated from the declared measurement
boundary rather than inferred from mean power.

### 6.4 Board power tracks HMMA throughput

**Can conclude:** in matched but separate executions, board power covaries with
measured HMMA throughput across variants, provided build ID, GPU UUID,
iterations, launch configuration, clock/power policy, and thermal conditions
are comparable.

**Supports but does not prove:** achieved Tensor-workload activity is an
important mediator of board power.

**Cannot conclude:** the slope is intrinsic energy per HMMA, other activity is
constant, or Tensor Core switching alone caused the difference. Register,
scheduler, issue, clock, leakage, and common SM activity can covary with HMMA/s.

### 6.5 Board power does not track occupancy

**Can conclude:** the selected occupancy metric alone does not explain the
cross-variant power pattern under the measured conditions.

**Possible hypotheses:** occupancy is on a plateau while eligible warps or issue
rate changes; Tensor throughput, clocks, or duration matter more; average
occupancy hides temporal structure; or profiling and power executions are not
sufficiently matched.

**Cannot conclude:** occupancy is irrelevant to latency hiding or Tensor
throughput.

### 6.6 Board power shows a non-monotonic peak

**Can conclude:** the power response to the coupled warp-supply treatment is
non-monotonic and cannot be summarized by a single proportional relationship.

**Candidate hypotheses:** a residency/allocation threshold, improved scheduling
at an intermediate block size, clock boosting at lower load, thermal or power
limiting at higher load, instruction/cache effects, tail-wave behavior, or
sampling/run-order artifacts. The peak must be replicated with interleaved run
order and explained using clocks, temperature, throttle state, runtime, and
separate NCU metrics.

**Cannot conclude:** an intermediate block size has intrinsically more
energy-intensive Tensor operations.

## 7. Evidence required

“NCU required” means NCU or an equivalently authoritative profiler is needed
for achieved runtime counters. Power and NCU executions remain separate.

| Variable | Why it matters | Source-level derivable? | Build/SASS required? | Runtime measurement required? | NCU required? | Power-run required? | Minimum evidence needed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Threads and warps per block | Defines treatment and resource packaging | Yes | No | Launch validation desirable | No | No | Variant specification plus emitted launch metadata |
| Total launched warps | Defines global warp supply and source-work multiplier | Yes | No | Launch validation desirable | No | No | Fixed block count × warps/block |
| Registers per thread | Determines register footprint/block and residency; may differ after compilation | No | Yes | No | No | No | `ptxas` resource report and preferably binary/SASS metadata for every build |
| Generated instruction sequence | Tests whether per-warp machine work is actually matched | No | Yes | No | No | No | Disassembly/diff of `wmma_example` for T0-T4, including loop/unrolling and spills |
| Theoretical occupancy | Establishes resident block/warp limits from resources | Partly | Yes | Device properties needed | NCU occupancy section or occupancy calculator useful | No | Registers/thread, shared memory, threads/block, allocation granularities, V100 limits |
| Resident blocks/warps limit | Defines available block-local scheduler supply | Partly | Yes | Device limits needed | NCU useful | No | Resource usage plus V100 residency limits and calculation |
| Achieved occupancy | Shows realized active warps relative to hardware maximum | No | Build identity required | Yes | Yes | No | Per-kernel achieved occupancy from matched profile executions |
| Active warps | Directly characterizes instantaneous/average warp supply | Total launched only | Build identity required | Yes | Yes | No | Active warps per cycle/SM or equivalent, with metric definition |
| Eligible warps/scheduler | Tests latency-hiding explanation for dependency chain | No | SASS dependency context useful | Yes | Yes | No | Eligible-warps-per-scheduler/cycle metric for `wmma_example` |
| `issue_active` | Measures instruction-issue utilization | No | Build identity required | Yes | Yes | No | V100-supported issue-active metric with unit and scope |
| Tensor instruction count | Validates denominator and that generated Tensor work scales as expected | Source `mma_sync` calls only | Yes | Preferably yes | Yes for executed count | No | HMMA SASS mapping plus executed Tensor instruction metric, explicitly per warp or thread as defined |
| HMMA/s | Measures achieved Tensor instruction throughput | No | HMMA definition/SASS required | Yes | Yes | No | Validated executed HMMA count divided by profile-kernel duration, with formula and profile ID |
| Runtime | Determines throughput and power-duration tradeoff | Event boundary known | No | Yes | No for primary runtime | Yes, unprofiled | CUDA-event `wmma_example` time from every power run; NCU runtime kept explanatory only |
| Sampled board power | Primary observed response | No | No | Yes | No | Yes | Timestamped `nvidia-smi` power samples, GPU UUID, units, window, sample interval, run ID |
| Integrated board energy | Determines total energy over a declared boundary | Formula only | No | Yes | No | Yes | Trapezoidal integration of source power samples over explicit timestamps, formula/version/run ID |
| SM clock | Detects DVFS mediation and comparability failures | No | No | Yes | Not necessarily | Yes | Timestamped SM clock aligned with power samples; configured policy and throttle state |
| Temperature | Detects thermal/run-order confounding | No | No | Yes | No | Yes | Initial, in-window, and final temperature with run order and cooldown metadata |
| Power/P-state/throttle policy | Establishes device-state comparability | No | No | Yes | No | Yes | GPU snapshots and in-run samples where available |
| Memory activity | Tests whether scaled fragment operations alter cache/DRAM behavior | Logical operations only | SASS helps identify memory instructions | Yes | Yes | No | DRAM bytes/throughput and relevant cache-sector metrics for `wmma_example` |
| Fragment load/store count | Quantifies logical one-time warp work | Yes at source-collective level | SASS required for machine instructions | Profiler needed for executed transactions | Optional for logical count; yes for physical activity | No | `3W` loads and `W` stores, clearly labeled source-level; profiler metrics for physical traffic |
| Kernel correctness/completion | Prevents partial or failed work entering denominators | No meaningful oracle in current source | Build identity required | Yes | No | Yes | Successful exit, launch/runtime error evidence where available, parseable event time; limitation that numerical result is unchecked |

If NCU permission is unavailable, Experiment A can still produce valid power,
clock, thermal, and runtime evidence. It cannot then establish achieved
occupancy, eligible/active warps, issue activity, executed Tensor instruction
count, HMMA/s, or physical memory activity. Interpretations must remain at the
coupled-treatment level.

## 8. Recommendation

### 8.1 Perform Experiment A first

**Recommendation requiring researcher approval:** perform Experiment A first as
the smallest experiment that tests the current source-based hypothesis that the
original benchmark's unusually large redundant warp supply is associated with
high board power.

It is already defined, preserves the original benchmark, changes a narrow
launch property, and can reveal whether the response is monotonic, saturated,
thresholded, or non-monotonic. It must be labeled as confounded with total
launched warp work by design.

### 8.2 Minimum measurements for Experiment A

The minimum primary measurement set is:

1. variant ID, block/grid dimensions, total threads, total warps, positive
   iteration count, build ID, source/variant diff, source hash, and binary hash;
2. unprofiled CUDA-event `wmma_example` runtime for every run;
3. timestamped board power with explicit process/kernel observation boundary
   and source run ID;
4. integrated board energy over that same explicit boundary, with timestamps,
   unit, formula, and formula version;
5. timestamped SM clock, temperature, performance state, and throttle reasons;
6. GPU model/UUID, driver/toolchain, power limit, clock policy, run order,
   repetition, initial temperature, and validity reasons.

The minimum explanatory set, collected separately when NCU permission exists,
is registers/thread and SASS comparison plus achieved occupancy, active/eligible
warps, issue-active, executed Tensor instruction count, HMMA/s, and memory
activity. Without it, the experiment still tests an association but cannot
identify the execution mechanism.

### 8.3 Validation before interpretation

Before interpreting Experiment A:

- confirm original-source immutability and deterministic variant diffs/hashes;
- confirm T0-T4 contain only intended block changes and the required fixed-grid
  preservation;
- compare compiler resource reports and SASS so unexpected machine-code,
  register, spill, or unrolling differences are exposed;
- confirm identical positive iteration count, matrix/input identity or
  equivalent initialization, event boundary, build environment, GPU UUID,
  power limit, and clock policy;
- confirm raw power and NCU records come from separate executions;
- verify sufficient sampling coverage relative to kernel/process duration;
- inspect clock, P-state, throttle, initial temperature, run order, and thermal
  drift;
- retain all failed/invalid runs and let the researcher decide scientific
  exclusions;
- state that the treatment couples warps/block with total launched work;
- avoid HMMA/FLOP normalization until the SASS/profiler denominator is defined
  and validated.

### 8.4 Should Experiment B immediately follow?

**Recommendation requiring researcher approval:** design Experiment B now, but
do not automatically generate or run it before reviewing Experiment A's valid
evidence.

Experiment B should follow only if Experiment A shows a significant or
meaningful and reproducible power, energy, runtime, or execution-activity
response **and** block packaging versus total work remains a mechanism that the
result cannot resolve. A null, negligible, irreproducible, or invalid Experiment
A result does not justify implementing Experiment B.

Experiment B must therefore not be implemented immediately or automatically.
If Experiment A has inadequate duration or sampling, uncontrolled clock or
thermal behavior, unexpected SASS/resource differences, or another validity
failure, those problems must first be resolved at the appropriate stage and
Experiment A reviewed again. The researcher, not the pipeline, decides whether
its effect size and reproducibility justify the mechanism test.

### 8.5 Conclusions by evidence class

**Confirmed from source**

- Per-warp WMMA fragment declarations, loads, loop body, dependency chain, and
  store can remain identical across T0-T4.
- Every warp uses the same base A, B, and C addresses.
- Host preparation, conversions, event boundary, result copy, and cleanup can
  remain identical.

**Analytically derived**

- At fixed 256 blocks, total warps fall from 4,096 to 256 with warps/block.
- Fragment calls, stores, and positive-iteration warp-level `mma_sync` calls
  fall in the same 1, 1/2, 1/4, 1/8, 1/16 sequence.
- Experiment B requires 256, 512, 1,024, 2,048, and 4,096 blocks to preserve
  4,096 total warps.

**Compiler/SASS dependent**

- Register allocation, spills, exact unrolling, machine instruction sequence,
  and HMMA count per source call.

**Runtime dependent**

- Resident/active/eligible warps, achieved occupancy, issue activity, HMMA/s,
  memory traffic, kernel duration, clocks, temperature, board power, and board
  energy.

**Hypothesis**

- Reducing combined block-local and global warp supply will reduce latency
  hiding, achieved Tensor activity, and board power after some scheduling or
  saturation threshold.
- Experiment B can distinguish much of the total-work association from the
  block-packaging association, but neither experiment alone isolates intrinsic
  Tensor Core energy.

## Final concise decision summary

### 1. Controlled variable

The intended source-level variable is `wmma_example` threads/warps per block,
implemented through `blockDim.x` and `blockDim.y`, with grid fixed at 256 blocks
and per-warp source work held constant.

### 2. Coupled variables

Total threads, total launched warps, fragment loads/stores, dynamically
requested warp-level `mma_sync` calls, and overlapping common-address stores
necessarily change. Residency, occupancy, eligibility, issue rate, latency
hiding, HMMA/s, memory traffic, runtime, clocks, thermal state, power, and energy
may also change.

### 3. Correct scientific name of Experiment A

**Fixed-grid coupled block-and-global warp-supply sweep.**

### 4. Minimum measurement set

Unprofiled CUDA-event runtime; timestamped board power and integrated board
energy with explicit boundary/formula/run ID; SM clock, temperature, P-state and
throttle state; launch/build/source identities; iterations; run order and
validity. Separately collect SASS/resources and, when permitted, NCU occupancy,
active/eligible warps, issue activity, Tensor instruction count/HMMA/s, and
memory activity.

### 5. Decision gate before code generation

The researcher must approve the coupled-sweep name and hypothesis, positive
iteration count, measurement boundary, repetition/order/thermal policy,
variant and grid specifications, minimum evidence set, and the rule that no
HMMA/FLOP energy normalization occurs before its denominator is validated.
Only then should Experiment A generation proceed. Experiment B remains a
contingent, separate mechanism test. It must not be generated or implemented
unless researcher review finds a valid, meaningful, and reproducible Experiment
A response and confirms that block packaging versus total work remains an
unresolved mechanism requiring decomposition.
