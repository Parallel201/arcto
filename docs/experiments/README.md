# Experiment logs

One log per optimization branch (`opt/<topic>-<yyyy-mm>`), one entry per commit. The
entries are the supporting material for the write-up: every change states **what** was
changed (files, mechanism), **why** it should help on the target architectures, the
**prediction** made before measuring, **how** it was measured (node, toolchain, inputs,
flags) and the **verdict** with the numbers. Kept commits and reverted attempts are both
logged — a reverted attempt is still evidence.

## Categories

Every entry (and every commit message, as a `Category:` line) is tagged with one of the
categories below so the optimizations can be aggregated per category when they are
written up:

| Tag | Category | Typical changes |
|---|---|---|
| **C1** | Synchronization & scheduling | spin-waits and yields, barriers, wave specialisation, work distribution between waves |
| **C2** | Wave-level primitives | shuffles → `readlane`/`readfirstlane`, DPP/hipCUB scans and reductions, ballots, match-any |
| **C3** | Memory access width & cache policy | byte → dword/dwordx4 loads and stores, alignment fast paths, coalescing, non-temporal stores |
| **C4** | LDS layout | bank-conflict-free layouts, padding, aligned queues, LDS-resident tables |
| **C5** | Occupancy, registers & launch bounds | `__launch_bounds__`/waves-per-EU, VGPR pressure, scalarisation of uniform state |
| **C6** | Launch geometry | threads per block, waves per block, chunks per block, grid sizing |
| **C7** | Compiler & build flags | optimisation flags, `-mcumode`, resource reports, per-arch flag sets |
| **C8** | Host-side & API overhead | allocations, copies, syncs, launch count, staging |

## Entry template

```
### <ID> — <title>                                   Category: Cn   Status: KEPT | REVERTED | PENDING
Commit: <sha>  (branch <name>)
Files: <paths>
Change: <what, in one or two sentences; the algorithm and the bytes are unchanged>
Why (mechanism): <architectural reason it should help on gfx90a / gfx942 / gfx1100>
Prediction: <direction and rough magnitude, written before measuring>
Measured: <node, toolchain, build flags, inputs, benchmark flags, repetitions>
Result: <numbers: median [Q1-Q3] before → after, per arch; compressed bytes identical? tests?>
Verdict: <kept / reverted and why>
```

## Gates (every entry)

- `ctest` green on the wave-size build of the node (wave32 on gfx1100, wave64 on gfx942);
- compressed bytes identical on the six `tests/data` fixtures (the "exact-bytes ladder")
  unless the entry says otherwise and reports the deviation in exact bytes;
- round trip bit-exact;
- throughput from the chunked benchmark at saturation (`-x`) and at small batch,
  ≥ 10 timed repetitions, raw per-repetition values kept (`ARCTO_PER_REP_CSV`).

## Reference environment

ROCm 7.0.1 in the project's toolchain container (`arcto_toolchain_rocm701.sif`), CMake
3.28, Release, `-D BUILD_TESTS=ON -D BUILD_BENCHMARKS=ON`; gfx1100 (RX 7900 XT, wave32
build) and gfx942 (MI300A, wave64 build).
