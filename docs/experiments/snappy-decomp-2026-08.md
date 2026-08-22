# Snappy decompression — experiment log (branch `opt/snappy-decomp-2026-08`)

Baseline: `main` @ `e3e1a1f` plus the `chore/build-and-cleanup-2026-08` and
`test/coverage-2026-08` branches (build hygiene and the Snappy coverage gate — no kernel
changes). The Snappy kernels on this baseline are the hipified nvCOMP 2.2 design: one
3-wave block per chunk (decode / prefetch / process) communicating through a `volatile`
queue in LDS; nothing in them had been tuned for AMD before this branch.

Plan (from `docs/AMD_OPTIMIZATION_MAP.md` §5, on the `docs/amd-optimization-map-2026-08`
branch): SNP-D1 → D3 → D4 → D5 → D2 → D7 → D9 → D6 → D8 → D11 → D12 → D10.

Gates: see `docs/experiments/README.md`. Benchmark: `benchmark_snappy_chunked`.

---

### SNP-D1 — yield the SIMD in the decoder's spin-waits (`s_sleep`)        Category: C1   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/device_functions.hiph`, `src/snappy/config.h`, `src/snappy/decompression_decode.hiph`,
`src/snappy/decompression_prefetch.hiph`, `src/snappy/decompression_process.hiph`
Change: the four polling loops of the three-wave decoder (decoder waiting for prefetched
bytes, decoder waiting for a free symbol slot, prefetcher waiting for ring space,
processor waiting for a batch) back off with `__builtin_amdgcn_s_sleep(n)` on AMD
instead of the inherited `clock()` no-op; `NANOSLEEP` becomes `SPIN_SLEEP` with
platform-dependent units (ns on CUDA sm_70+, s_sleep units on AMD), the literal
`NANOSLEEP(100)` becomes the named `PROCESS_SLEEP`, and the AMD amounts are tunables
`ARCTO_SNAPPY_{PREFETCH,DECODE,PROCESS}_SLEEP` (defaults 2 / 1 / 1). Algorithm, stream
format and emitted bytes unchanged (scheduling only). CUDA path unchanged.
Why (mechanism): `s_sleep n` parks the wave for ~64·n clocks and hands the SIMD's issue
slots to other resident waves. With ≈ 6–9 Snappy blocks per CU (LDS-bound) a large share
of resident waves are pollers at any moment; a tight `volatile`-LDS poll competes for
issue and LDS bandwidth with the productive waves of the other chunks on the same SIMD
(and, for the decoder lane-0 polls, with its own sibling waves). Applies to CDNA and RDNA3.
Prediction: decompression throughput up at saturation (many resident blocks) — tens of
percent plausible, near zero at small batch; compression unchanged; bytes identical;
effect should grow with the number of resident blocks (`-x`) and be visible in
`SQ_WAIT_INST_LDS` / `SQ_INSTS_LDS` going down.
Measured: (pending) lunaris gfx1100 wave32 and sdumont2nd4014 gfx942 wave64, ROCm 7.0.1
container, `benchmark_snappy_chunked` on tti_rsf_64x64x64_t050 / synth_binary / words
inputs, `-p 65536`, `-x 512` and `-x 0`, 30 reps; sweep of the three tunables
(0/0/0 = yield-only, 1/1/2 default, 2/2/4, 4/4/8).
Result: (pending)
Verdict: (pending)
