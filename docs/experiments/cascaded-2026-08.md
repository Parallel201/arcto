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

### CAS-S1 — 256-thread blocks on wave64 targets                                        Category: C6   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph`
Change: `cascaded_compress_threadblock_size` / `cascaded_decompress_threadblock_size` become
`(arcto::warpsize >= 64) ? 256 : 128` — 4 waves per block on every target. All device code (the
batched kernels, the HLIF wrappers and their occupancy queries, hipCUB collectives) is templated
on the constant; the element-to-thread mapping changes the schedule only, never the values
(RLE output order and bit-pack words are index-defined). wave32 and CUDA unchanged (128).
Why (mechanism): the LDS footprint of a block is fixed by the 4 KB chunk (10–25 KB per block), so
on CDNA (64 KB LDS/CU) a 128-thread block is 2 wave64 and the CU holds 2–6 blocks → 1–3 waves per
SIMD; 256 threads doubles the resident waves at the same LDS (u32: 5 blocks × 4 waves = 5 waves/
SIMD instead of 2.5), halves the number of rounds of every "per threadblock_size elements" loop
(`get_for_bitwidth`, delta/RLE decompress, RLE compress prefix sums), and halves
`num_inputs_per_thread` in `block_rle_compress` (16-B per-lane stride instead of 32 B → 4-way
instead of 8-way LDS bank conflicts).
Prediction: compression 15–30 %, decompression 10–25 % on gfx942; nil on gfx1100 (unchanged);
bytes identical on both.
Measured: (pending) gfx942 = the target; gfx1100 as a no-change check.
Result: (pending)
Verdict: (pending)

### CAS-C1 — single-pass min/max in `get_for_bitwidth` (2 collectives instead of 2 per 128 elements)   Category: C1   Status: PENDING (not yet measured)
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph`
Change: each thread folds its strided share of the array into register-resident min/max (unit-stride
LDS reads, no barriers), then one `BlockReduce(Min)` and one `BlockReduce(Max)` over the whole block
(threads without elements contribute the identity values). Replaces two block reductions + two
barriers per `threadblock_size` elements (u8: 64 collectives per array, ×3 arrays per chunk with
the default {2 RLE, bp}). Min/max are order-independent: FOR and bitwidth bit-identical.
Prediction: 20–40 % of compression time with `use_bp` on CDNA (barrier-bound at 1–3 waves/SIMD),
less on gfx1100; bytes identical.
Measured: (pending — next sweep: baseline, S2, S1, C1 on gfx942 and gfx1100)
Result: (pending)
Verdict: (pending)

### Resource evidence — gfx942 (MI300A), ROCm 7.0.1, baseline kernels (before CAS-S2/S1/C1)
`-Rpass-analysis=kernel-resource-usage`, wave64, 128-thread blocks (inherited): `cascaded_compression_kernel<int,128,4096>` SGPR 106 / VGPR 71 / LDS 13456 B per block / 9 SGPR spills / est. 7 waves per SIMD; `cascaded_decompression_kernel<4B,128,4096>` SGPR 106 / VGPR 73 / LDS 13192 B / 12 SGPR spills / est. 6 waves per SIMD; `get_decompress_size_kernel` 22 / 10 / 0 LDS / 8. The SGPR spills (scalar state of the nested RLE/delta/bit-pack passes) and the 13 KB of LDS per 128-thread block (≤4 blocks per CU by LDS) are the two structural costs to attack (CAS-S2 launch bounds already committed; CAS-D5 LDS shrink queued).
