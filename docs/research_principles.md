# Research Principles for GPU Energy Characterization

## 1. Purpose

This project exists to answer scientific questions about GPU energy behavior.

The framework, scripts, metadata, and automation are supporting tools. They are not the primary research objective.

The current immediate objective is to explain the measured power behavior of the AccelWattch TENSOR microbenchmark on NVIDIA V100 and to establish a defensible basis for estimating effective energy per HMMA or per FLOP.

Generality should emerge from repeated validated experiments. It must not delay or obscure the current scientific objective.

---

## 2. Research Goal First, Framework Second

Every implementation task must be traceable to a concrete research question.

Before creating a new abstraction, script, interface, or directory, ask:

1. Which research question does this support?
2. Which experiment requires it now?
3. What evidence will it produce?
4. Can the same question be answered with a simpler implementation?

Do not build infrastructure solely because it may be useful in the future.

---

## 3. One Controlled Variable per Experiment

Each experimental variant should change only one intended factor whenever technically possible.

Examples include:

* warps per block;
* total launched warps;
* grid dimensions;
* fragment initialization;
* loop unrolling;
* dependency structure;
* register pressure;
* memory activity;
* instruction mix.

All other known factors must remain unchanged or be explicitly recorded.

If one source change unintentionally affects multiple execution properties, the experiment must be labeled as confounded rather than interpreted as a single-factor result.

---

## 4. Preserve the Original Benchmark

The original AccelWattch TENSOR benchmark is immutable reference evidence.

It must not be modified in place.

Controlled variants must be produced through:

* copied source files;
* mechanically generated sources;
* compile-time parameters;
* isolated patches; or
* reproducible generation scripts.

Every variant must record:

* the original source hash;
* the variant source hash;
* the exact intended change;
* the generated diff;
* the build command;
* the resulting binary hash.

---

## 5. Reproducible Variant Generation

A benchmark variant must be regenerable from:

* the original source revision;
* a versioned variant specification;
* a deterministic generation procedure.

Manual source edits that cannot be reproduced are not acceptable as final experimental artifacts.

Variant names must describe the controlled factor rather than an informal historical label.

Recommended examples:

* `tensor_warp16`
* `tensor_warp8`
* `tensor_warp4`
* `tensor_unroll4`
* `tensor_depchain1`

Avoid names whose meaning depends on conversation history.

---

## 6. Power Measurement and Profiling Are Separate Experiments

Primary board-power measurements and NCU profiling must never be treated as the same execution.

Power runs:

* must execute without NCU instrumentation;
* produce the primary timing, power, and energy evidence;
* use the normal benchmark binary and measurement window.

Profile runs:

* collect explanatory architectural metrics;
* may incur replay and instrumentation overhead;
* must not provide runtime or power values for the primary energy calculation.

Power and profile records may be compared only when they share the same:

* `variant_id`;
* `build_id`;
* GPU model and UUID;
* iteration count;
* launch configuration;
* relevant clock and power policy.

They must retain separate `run_id` and `profile_id` values.

---

## 7. Preserve Raw Evidence

Raw measurement and profiler outputs are immutable.

The pipeline may generate:

* extracted data;
* normalized data;
* summaries;
* figures;
* reports.

It must never rewrite or silently clean the original raw evidence.

Invalid and failed runs must remain recorded with explicit reason codes.

A retry is a new run, not a replacement.

---

## 8. Every Derived Value Must Be Traceable

Every calculated quantity must identify:

* the source run or runs;
* the source columns or metrics;
* the unit;
* the formula;
* the formula version;
* the validity conditions.

Examples include:

* mean sampled board power;
* integrated process-window board energy;
* HMMA throughput;
* energy per launched warp;
* energy per HMMA;
* energy per FLOP.

No derived value may appear only as an unexplained number in a report.

---

## 9. Do Not Confuse Measurement Boundaries

The energy boundary must always be stated explicitly.

Possible boundaries include:

* full GPU board power;
* process-window GPU board energy;
* kernel-window GPU board energy;
* idle-subtracted dynamic board energy;
* modeled component energy;
* effective energy per operation.

These quantities are not interchangeable.

In particular, measured board energy per HMMA is not automatically equivalent to intrinsic Tensor Core switching energy.

---

## 10. Do Not Infer Pure Tensor Energy Prematurely

A Tensor-heavy benchmark includes unavoidable common contributions such as:

* scheduler activity;
* register-file activity;
* instruction issue and control;
* active SM state;
* clock-distribution activity;
* static and leakage power;
* loop and dependency-management overhead;
* limited setup and completion activity.

Therefore, a measured value must initially be described as effective Tensor-workload energy.

It may be interpreted as isolated Tensor energy only after the assumptions of the isolation method have been experimentally validated.

---

## 11. Differential Subtraction Requires Matched Common Activity

A non-Tensor benchmark is not automatically a valid subtraction baseline.

Before differential subtraction is accepted, the candidate baseline should be evaluated for similarity in:

* occupancy;
* active warps;
* issue-active behavior;
* scheduler pressure;
* register usage;
* dependency pattern;
* loop and branch structure;
* memory activity;
* clock and thermal state;
* execution duration.

If these common activities are not sufficiently matched, the subtraction result must not be labeled as Tensor-only energy.

---

## 12. Source-Level Equality Does Not Imply Execution Equality

Two benchmarks using the same `wmma::mma_sync()` call may still differ in:

* block and grid configuration;
* active warps per SM;
* compiler unrolling;
* register allocation;
* SASS instruction scheduling;
* dependency latency hiding;
* instruction throughput;
* fragment load and store behavior;
* clocks, temperature, and power state.

The following levels must be distinguished:

1. source-code behavior;
2. generated SASS;
3. runtime execution context;
4. measured board-power behavior.

Claims must state which level supports them.

---

## 13. Separate Confirmed Facts from Hypotheses

Every technical conclusion should be labeled as one of:

* confirmed from source code;
* confirmed from compiler or SASS output;
* confirmed from runtime measurement;
* supported by profiling data;
* plausible hypothesis;
* unresolved.

Do not present an architectural hypothesis as a measured fact.

When evidence is insufficient, state that it is not estimable from the available data.

---

## 14. Normalize Only After Validating the Denominator

Before calculating energy per HMMA or per FLOP, verify:

* the definition of one HMMA;
* the number of HMMA instructions produced by one WMMA call;
* whether counts are source-level, SASS-level, profiler-level, or model-level;
* the number of warps executing the operation;
* the number of iterations;
* whether failed or partial work is included;
* the FLOP convention used.

The denominator and its derivation must be included in the result.

---

## 15. Prefer Small Experiments over Large Refactors

When investigating a power difference, prefer:

* one small variant;
* one explicit hypothesis;
* one measurable response;
* one small Git commit.

Avoid combining benchmark redesign, framework refactoring, new profiling, and new analysis in one task.

A small controlled experiment that produces interpretable evidence is more valuable than a large generalized implementation.

---

## 16. The Framework Must Grow from Validated Use

The framework should be generalized only after an interface has been used successfully in a real experiment.

The current TENSOR warp-count study is the first reference implementation.

Future support for FMAD, SFU, Mix, cuBLAS, or Transformer workloads should be added only when a corresponding study begins.

Do not create unused adapters or speculative abstractions.

---

## 17. Human Scientific Judgment Remains Required

Codex may:

* inspect source code;
* generate controlled variants;
* create scripts;
* execute build and validation steps;
* extract and normalize results;
* prepare technical summaries.

Codex must not independently decide:

* whether a differential baseline is scientifically valid;
* whether a run should be excluded for scientific reasons;
* whether a measured quantity represents intrinsic component energy;
* whether a causal conclusion is justified;
* whether the research question has been conclusively answered.

Those decisions require researcher review.

---

## 18. Current Study Priority

Until explicitly changed, the immediate work priority is:

1. preserve the original AccelWattch TENSOR benchmark;
2. generate controlled TENSOR variants;
3. measure power without profiler instrumentation;
4. collect NCU data separately when available;
5. extract compact, traceable results;
6. determine which execution-context variables explain the measured power;
7. assess whether a defensible effective energy per HMMA can be calculated;
8. only then evaluate common-power isolation or differential methods.

Framework work that does not support this sequence should be deferred.

---

## 19. Definition of a Successful Experiment

An experiment is considered successful when:

* the intended controlled variable is clearly defined;
* the original source is preserved;
* the variant is reproducibly generated;
* build and binary identities are recorded;
* the run produces valid raw evidence;
* measurement and profiling boundaries are explicit;
* derived values trace to raw data and formulas;
* confounders and limitations are documented;
* the result answers or narrows a stated research question.

A successful experiment does not require the original hypothesis to be correct.

---

## 20. Governing Principle

The governing principle of this project is:

> Build only what is needed to produce reliable evidence for the current scientific question, preserve that evidence completely, and generalize only after the method has been validated.
