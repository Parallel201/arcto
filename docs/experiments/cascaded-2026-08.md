# Cascaded (RLE → Delta → BitPack) — experiment log (branch `opt/cascaded-2026-08`)

Baseline: `main` @ `e3e1a1f` plus `chore/build-and-cleanup-2026-08` and `test/coverage-2026-08`
(incl. `tests/test_cascaded_coverage.cpp`, the gate for this branch). The batched kernels
(`src/lowlevel/CascadedBatch.hip` → `src/CascadedKernels.hiph`) are the hipified nvCOMP 2.2
design: one 128-thread block per partition, serial over 4 KB chunks held in LDS, RLE / delta /
bit-pack stages fused in one kernel with one hipCUB collective + barrier per 128 elements;
decompression launches four width-specialised kernels per call. On CDNA (64 KB LDS per CU,
128 threads = 2 wave64) the 10–25 KB of static LDS per block leaves 1–3 waves per SIMD.

Plan (from `docs/AMD_OPTIMIZATION_MAP.md` §6): CAS-S2 → S1 → C1 → S4 → D2 → D1 → S5 → C2/C3 →
S6/S7 → S3/D5 → (D3). Measured with `benchmark_cascaded_chunked` (default opts {2 RLE, 1 delta,
bp, INT}), the exact-bytes ladder and `test_cascaded_coverage`; same protocol as the Snappy
branches (`docs/experiments/README.md`). Inputs: the TTI float field (as int32), synth_zeros
(extreme RLE) and a generated int32 mixture of runs / ramps / small-range noise.

---

### CAS-S2 — `__launch_bounds__` on the three batched Cascaded kernels                 Category: C5   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/lowlevel/CascadedBatch.hip`
Change: `cascaded_compression_kernel`, `cascaded_decompression_kernel_type_check` and
`get_decompress_size_kernel` carry `__launch_bounds__(threadblock_size)` (the template / constant
they are launched with). The HLIF kernels (`hlif_shared.hiph`) are shared with other codecs and
stay as they are for now.
Why (mechanism): without bounds hip-clang budgets registers for a 1024-thread workgroup
(`amdgpu_flat_work_group_size` 1..1024) — at most 128 VGPRs per lane on CDNA — and has no
occupancy target; with the true block size the compiler may use the registers the real block
allows and schedule accordingly. Enabler for CAS-S1; no algorithm/bytes change.
Prediction: 0–5 %; bytes identical.
Measured: (pending) `benchmark_cascaded_chunked`, exact-bytes ladder, `test_cascaded_coverage`.
Result: (pending)
Verdict: (pending)
