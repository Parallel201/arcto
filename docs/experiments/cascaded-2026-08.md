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

### CAS-S4 — `BLOCK_SCAN_WARP_SCANS` for the three Cascaded block scans                 Category: C2   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph`
Change: the `hipcub::BlockScan` typedefs in `block_rle_compress` (run-count exclusive sum),
`block_rle_decompress` (run-offset exclusive sum per round) and `block_delta_decompress`
(prefix sum per round) take `ARCTO_CASCADED_BLOCK_SCAN_ALGORITHM`, which defaults to
`hipcub::BLOCK_SCAN_WARP_SCANS` on AMD (rocPRIM `using_warp_scan`) and to hipCUB/CUB's default
`BLOCK_SCAN_RAKING` elsewhere. Overridable per build.
Why (mechanism): rocPRIM's `using_warp_scan` does one in-register DPP/`ds_swizzle` scan per
wave and exchanges only the per-wave prefixes through LDS (one `ds_write` + one `ds_read` per
wave, one barrier); `reduce_then_scan` (hipCUB's mapping of RAKING) reduces per wave, scans the
wave totals in one wave, then scans again — two LDS round trips and an extra barrier per block
scan. Every RLE round (`threadblock_size` runs) and every delta round (`threadblock_size`
elements) pays one block scan, so a 4096-element chunk pays 16 (128-thread) / 8 (256-thread)
scans per delta layer. Output bytes identical (same sums, integer arithmetic).
Prediction: +2–6 % decompression on delta/RLE-heavy inputs (ints_mixed), ≈ 0 on zeros (one
run per chunk) and TTI; compression ±1 % (one scan per RLE layer); bytes identical.
Measured: (pending) `benchmark_cascaded_chunked`, exact-bytes ladder, `test_cascaded_coverage`.
Result: (pending)
Verdict: (pending)

### CAS-D5 — decompression LDS shrink: RLE count scratch aliases the dead element buffer   Category: C4   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`compute_smem_size`, `cascaded_decompression_fcn`)
Change: the bit-unpack scratch used while an RLE count array is read from global memory
(`temp_count_array`, `chunk_num_elements * 2` B) is no longer a separate LDS region for data
types of 2+ bytes: it aliases `shared_output_buffer`, which is dead at that point (the previous
layer's input, already consumed; the final-array read already used it as scratch). 1-byte types
keep the dedicated region (their count array is twice the element buffer). `compute_smem_size`
shrinks accordingly, so the batched `cascaded_decompression_kernel_type_check` and the HLIF
decompressor (both size their static `__shared__` from it) get the smaller footprint.
Static LDS per block, 4 KB chunk, before scan scratch: u8 24 688 (unchanged), u16 16 496 → 12 400,
u32 12 400 → 10 352, u64 10 368 → 9 344 B.
Why (mechanism): on CDNA (64 KB LDS per CU) resident blocks per CU for the default `int` type go
from 4 (13.2 KB incl. scan scratch) to 6 (≈11.1 KB); with 256-thread blocks that is 16 → 24
resident waves per CU (4 → 6 per SIMD), more latency hiding for the global loads and the
barrier-heavy layer pipeline. On gfx1100 (64 KB per CU in CU mode / 128 KB per WGP) the same
ratio applies. Output bytes identical (scratch contents are consumed before the buffer is reused).
Prediction: +5–15 % decompression at saturation on gfx942 for int inputs, ≈ 0 at small batch;
compression unchanged; bytes identical.
Measured: (pending) `benchmark_cascaded_chunked`, exact-bytes ladder, `test_cascaded_coverage`
(which round-trips every type through every RLE/delta/bp configuration).
Result: (pending)
Verdict: (pending)

### CAS-D2 — multi-item block scan in delta decompression                               Category: C1   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_delta_decompress`)
Change: each round scans `threadblock_size * ARCTO_CASCADED_DELTA_ITEMS` elements with
`BlockScan::ExclusiveScan(T (&)[ITEMS], T (&)[ITEMS], initial, Sum, aggregate)` (blocked
arrangement: ITEMS consecutive elements per thread), carrying `initial_value += aggregate`
across rounds exactly as before. Default ITEMS = 4 on AMD, 1 on the CUDA backend (nvCOMP's
original one-element-per-thread rounds); overridable per build (`-DARCTO_CASCADED_DELTA_ITEMS=N`).
Why (mechanism): a 4 KB chunk of 32-bit values paid 16 (128 threads) / 8 (256 threads) block
scans + barriers per delta layer; with ITEMS = 4 it pays 4 / 2. The per-thread slab is 16 B of
LDS (four `ds_read_b32` at a 16-B lane stride, or one `ds_read_b128` when the compiler can prove
alignment); the scan itself is the same wave-level primitive with a 4-element serial prefix per
lane first. Integer sums in any association give identical output bytes.
Prediction: +3–10 % decompression on delta-using inputs (default opts have one delta layer),
≈ 0 on zeros; bytes identical. Try ITEMS = 2 / 8 as flag variants.
Measured: (pending) `benchmark_cascaded_chunked`, exact-bytes ladder, `test_cascaded_coverage`.
Result: (pending)
Verdict: (pending)

### CAS-D1 — load-balanced RLE expansion in `block_rle_decompress`                        Category: C1   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_rle_decompress`)
Change: after the per-round exclusive scan of the run counts, each thread publishes its run's
end offset in LDS (`run_end[threadblock_size]`, 512 B) and the block votes with
`__syncthreads_or(count > ARCTO_CASCADED_RLE_BALANCE_THRESHOLD)` (16). If any run of the round
is longer, every thread fills a strided slice of the round's output (`o = tid, tid + TB, …`),
locating the owning run by a binary search over the round's ≤ TB run ends (log2(runs) ≤ 8 LDS
reads, wave-uniform addresses for neighbouring lanes → broadcast); otherwise the inherited
one-thread-per-run fill runs. Default on for AMD (`ARCTO_CASCADED_RLE_BALANCED`), off on the CUDA
backend (nvCOMP's schedule); threshold overridable.
Why (mechanism): with one thread per run, a run of length L costs L dependent LDS stores on one
lane while the other 255 idle — for the data Cascaded exists for (long runs) the RLE layer is a
serial loop: a 4096-element all-equal chunk is 4096 iterations on one lane per layer. The
balanced fill costs ≈ (aggregate / TB) × (≤ 8 reads + 1 read + 1 store) per lane, fully
parallel; the vote keeps short-run rounds (avg. ≤ 16) on the cheap path. Values and positions
unchanged → bytes identical.
Prediction: zeros / long-run inputs +20–50 % decompression (RLE layers dominate them), ints_mixed
+5–15 % (35 % of it is runs), TTI ≈ 0 (no runs → serial path), compression unchanged; bytes
identical. Try threshold 4 / 64 as flag variants.
Measured: (pending)
Result: (pending)
Verdict: (pending)

### Resource evidence — gfx942 (MI300A), ROCm 7.0.1, baseline kernels (before CAS-S2/S1/C1)
`-Rpass-analysis=kernel-resource-usage`, wave64, 128-thread blocks (inherited): `cascaded_compression_kernel<int,128,4096>` SGPR 106 / VGPR 71 / LDS 13456 B per block / 9 SGPR spills / est. 7 waves per SIMD; `cascaded_decompression_kernel<4B,128,4096>` SGPR 106 / VGPR 73 / LDS 13192 B / 12 SGPR spills / est. 6 waves per SIMD; `get_decompress_size_kernel` 22 / 10 / 0 LDS / 8. The SGPR spills (scalar state of the nested RLE/delta/bit-pack passes) and the 13 KB of LDS per 128-thread block (≤4 blocks per CU by LDS) are the two structural costs to attack (CAS-S2 launch bounds already committed; CAS-D5 LDS shrink queued).
