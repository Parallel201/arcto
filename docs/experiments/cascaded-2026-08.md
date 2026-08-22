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
Result (gfx1100, RX 7900 XT, wave32, 30 reps; ints_mixed / zeros / TTI; comp GB/s sat 75.0 / 175.7 / 57.8, decomp 76.6 / 46.6 / 287.3; x0 comp 50.6 / 8.97 / 3.26, decomp 67.8 / 2.55 / 21.7): ×1.000 / ×1.000 / ×1.001 comp, ×1.002 / ×1.005 / ×1.002 decomp — neutral, as predicted (bounds only). Bytes identical.
Result (gfx942, MI300A, wave64, 30 reps; ints_mixed / zeros / TTI; base comp GB/s sat 109.5 / 305.7 / 95.4, decomp 181.3 / 167.4 / 539.5; x0 comp 33.5 / 6.89 / 1.97, decomp 51.9 / 3.24 / 20.1): comp ×1.007 / ×0.947 / ×0.949 at saturation, ×1.000 / ×1.011 / ×1.003 at x0; decomp ×0.998 / ×0.933 / ×0.985 sat, ×0.937 / ×1.036 / ×1.112 x0 — mixed ±5 %, within this node's drift; bounds at the inherited 128 threads give the compiler nothing to use. Kept as the enabler for S1.
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
Result (gfx1100): wave32 keeps 128 threads — by construction a no-op here (×0.999–1.001). gfx942 pending.
Result (gfx942): compression ×1.474 ints (x32) / ×1.207 zeros / ×1.561 TTI at saturation, ×1.50 / ×1.19 / ×1.68 at x0; decompression ×1.246 / ×1.004 / ×1.248 at saturation, ×1.34 / ×1.04 / ×1.70 at x0 — the largest single decompression win on CDNA and second-largest for compression: same LDS per block, twice the resident waves, half the per-threadblock_size rounds per stage. Bytes identical.
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
Result (gfx1100): compression ×1.551 ints_mixed (75.0 → 116.3 GB/s at x32; x0 ×1.542), ×1.698 TTI (57.8 → 98.1 at x512; x0 ×1.707), ×1.050 zeros (bitwidth 0: one reduce pair either way); decompression unchanged; bytes identical. The largest single win of the branch: the per-128-element collective pair was the compressor's hot spot on bit-packed data.
Result (gfx942): compression ×1.44 ints (x32: 161 → 232 GB/s), ×1.57 TTI (149 → 234), ×1.14 zeros; x0 ×1.44 / ×1.69 / ×1.17; decompression unchanged. Same mechanism as on gfx1100.
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
Result (gfx1100): decompression ints_mixed ×1.031 (x32) / ×1.030 (x0), zeros ×1.000, TTI ×1.004; compression ×1.005 / ×1.004 / ×1.002 — small positive on the scan-heavy input, as predicted. Bytes identical.
Result (gfx942): compression ×1.19 ints (232 → 277), ×1.16 TTI (234 → 271), ×1.18 zeros (420 → 494) at saturation; decompression ×1.15 ints (242 → 278), ×1.10 TTI (656 → 722), zeros ≈; x0 ±3 %. Much larger than on RDNA3: hipCUB's default (rocPRIM `reduce_then_scan`) is expensive on wave64 CDNA and `using_warp_scan` (DPP) is the right block scan there.
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
Result (gfx1100): decompression ints_mixed ×1.134 at x32 (79.0 → 89.6 GB/s; x0 ≈, latency-bound), zeros ×1.173 at x512 (46.6 → 54.7), TTI ×0.976 at x512 (288.5 → 281.6: TTI is mostly the raw fall-back copy and gains nothing from residency, the −2 % is within its run-to-run band); compression unchanged; bytes identical. Occupancy lever confirmed on the int type even on RDNA3 (64 KB LDS per CU in CU mode).
Result (gfx942): decompression zeros ×1.12 at saturation (164 → 184 GB/s), ints ≈ (274 vs 278, within drift), TTI ≈; x0 ints ×0.90 (75.5 → 68.1 — the x0 ints rows on this node vary ±5 % between neighbouring commits; the layout change is the only code change); compression unchanged; bytes identical. Kept: the residency gain shows on the RLE-heavy input.
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
Result (gfx1100): decompression ints_mixed ×1.086 at x32 (89.6 → 97.3 GB/s), ×1.052 at x0; zeros ×0.984 (no delta layer work: the extra per-round register traffic), TTI ×1.005; compression unchanged; bytes identical.
Result (gfx942): decompression ints ×1.089 at x32 (274 → 299 GB/s), ×1.078 at x0; TTI ×1.02; zeros ≈; compression unchanged; bytes identical.
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
Result (gfx1100, threshold 16): decompression zeros ×7.90 at x512 (53.8 → 368 GB/s) and ×6.7 at x0 (2.55 → 17.0) — the serial one-lane expansion was the whole kernel on run data; TTI ≈ (no runs, serial path); ints_mixed ×0.935 at x32 (97.3 → 90.9) / ×1.003 at x0 — the balanced fill's log2 LDS reads per element cost more than the serial fill saves on rounds whose longest run is only moderately above 16. Bytes identical. Threshold variants (4 / 64 / 256) queued on gfx1100 (`-DARCTO_CASCADED_RLE_BALANCE_THRESHOLD=N`) to place the switch-over where the ints regression disappears while the zeros gain stays. Compression numbers at x0 on zeros become bimodal from this commit on (IQR 5–10 GB/s): the decompression phase got ~7× shorter and the 1 MB single-chunk compression timing now sees GPU clock ramping — use x512 for zeros comparisons.
Result (gfx942, threshold 16): decompression zeros ×3.0 at x512 (186 → 534 GB/s) and ×4.45 at x0 (3.34 → 14.4); TTI ≈; ints ×0.91 at x32 (299 → 272) / ×0.954 at x0 — the same ints regression as on gfx1100. Bytes identical. Verdict pending the threshold variants (gfx1100 cas4: 4 / 64 / 256).
Verdict: (pending)

### CAS-D1f — balance on "one run dominates the round" instead of an absolute run length        Category: C1   Status: PENDING (factor variants 4 / 8 / 16)
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_rle_decompress`)
Measured first (gfx1100, D1 absolute threshold 16 / 4 / 64 / 256 on the H2 head): ints_mixed
decompression ×1.00 / ×0.90 / ×1.00 / ×0.95–0.97 (x32 / x0), zeros ×1.00 / ×1.03 / ×1.03 / ×1.01, TTI ≈
— no cutoff recovers the −7 % on ints (97.3 GB/s before D1, 90.9 after, at every cutoff), because
ints_mixed rounds mix one or two 200–400-element runs with hundreds of 1–3-element runs: any
absolute cutoff sends those rounds down the balanced path, whose cost is `aggregate / threads`
elements per lane at ~10 LDS operations each, while the serial fill costs only the longest run.
Change: the vote becomes `count * threadblock_size > FACTOR * aggregate` (a run longer than
FACTOR × the round's average run), so a round dominated by one run (zeros: one 4096-run per round)
balances and a round of comparable runs stays serial; FACTOR default 8, knob
`ARCTO_CASCADED_RLE_BALANCE_FACTOR` (the absolute-threshold knob is gone).
Prediction: ints back to the pre-D1 level (≈ +7 %), zeros unchanged (×7.9 / ×3–4.5 kept), TTI ≈.
Measured (gfx1100, cas5, H3 head): factor 8 / 4 / 16 — ints_mixed decompression 89.0 / 85.4 / 88.8
GB/s at x32 (pre-D1 97.3; D1 abs-16 90.9), x0 72.5 / 69.1 / 72.9 (pre-D1 73.4); zeros x512 360 / 348
/ 355 (D1 abs-16 341); TTI ≈. Bytes identical; the full coverage matrix passes (72 684 assertions).
Result: the heuristic cannot recover ints either — the ~8 % at saturation is the per-round block
vote itself (`__syncthreads_or` = LDS vote + barrier on every round, taken or not), not the path
choice. Factor 8 keeps zeros at its best (+5.6 % over the absolute cutoff).
Verdict: superseded by CAS-D1w (next commit): wave-local balancing with a wave ballot — no block
vote, no extra barrier.

### CAS-D1w — wave-local balanced RLE expansion (ballot, no block vote)                   Category: C1   Status: COMMITTED (final sweep pending)
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_rle_decompress`)
Change: each wave owns the contiguous output range of its own `warpsize` runs of the round
(`wave_start`/`wave_end` from two shuffles of the scan's offsets); if `__ballot(count * warpsize >
FACTOR * wave_total)` is non-zero — wave-uniform, free — the wave's lanes fill its range cooperatively
(run ends in a wave-private LDS region, `__threadfence_block`, binary search over ≤ 64 entries);
otherwise the per-run fill. No block-wide vote or barrier is added to any round; the round's
existing `__syncthreads()` remains. Same values to the same positions: bytes identical.
Prediction: ints back to the pre-D1 level (≈ 97 GB/s at x32 on gfx1100), zeros keeps the D1 gain
(64-lane instead of 256-lane balancing of a single run: still ×6+), TTI ≈.
Measured (cas6, gfx1100, 30 reps; vs H3 = block vote, and vs base 777135f): decompression ints
x32 88.8 → 96.1 GB/s (×1.082 vs H3; base 75.3 → ×1.276 cumulative; pre-D1 was 97.3), x0 72.4 → 78.9
(×1.090; ×1.163 cumulative — above the pre-D1 73.4: the wave-local fill also speeds up rounds with
moderately long runs); zeros x512 348 → 296 (×0.85 vs H3; ×6.43 vs base; the 64-lane fill of a
single 4096-run is less parallel than the 256-lane one), x0 14.0 → 11.0 (bimodal band 8–18); TTI
≈ (×0.98 vs base at saturation). Compression ×1.00–1.01 vs H3 (H1b +0.6 % ints). Full coverage
matrix green (72 684 assertions); ladder identical.
Result: the ints cost of D1/D1f is gone (and a little more), zeros keeps most of its ×6–8, TTI flat.
Verdict: KEPT — the final form of CAS-D1. gfx942 (cas6, 30 reps, vs H3 / vs base): decompression ints
x32 233 → 256 GB/s (×1.10 / ×1.43), x0 68.9 → 72.7 (×1.056 / ×1.40); zeros x512 463 → 422 (×0.91 /
×2.49), x0 15.0 → 12.9 (×0.86 / ×3.86); TTI x0 ≈ (×1.54 vs base), x512 664 (the base row of this run
reads 737 with IQR 552–740, the earlier base 539 — noise band); compression ×1.01–1.02 vs H3 (H1b).
Full matrix green; ladder identical.

### CAS-H1b — header-gap zeroing folded into the header-writing thread-0 section          Category: C8   Status: COMMITTED
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Change: the CAS-H1 gap stores move from a separate post-barrier thread-0 section in `block_bitpack`
into `get_for_bitwidth`'s existing thread-0 header write (one divergent section per bit-pack call
instead of two). H1 measured −2…−3 % (gfx1100) / −4…−7 % (gfx942, drift band) compression.
Measured: (final sweep)

### CAS-C4 — incremental input-window arithmetic in `block_bitpack`                    Category: C1   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_bitpack`)
Change: the word-per-thread packing loop computed `input_idx_start = out_bit_start / bitwidth`
and `input_idx_end = roundUpDiv(out_bit_end, bitwidth)` for every output word — two integer
divisions by a runtime divisor. A thread's consecutive words are `threadblock_size` apart, so
both indices advance by a fixed quotient/remainder per iteration; they are now kept with an
exact incremental division (two divisions per thread before the loop). Inner loop unchanged.
Why (mechanism): 32-bit integer division by a runtime value is a ~30–40-instruction VALU
sequence on GCN/CDNA/RDNA (no hardware divider); at bitwidth 8 the inner loop is five 8-op
iterations, so the two divisions were roughly half the packing work per word. Packing runs on
every layer output with `use_bp` (RLE counts, final array); the words produced are identical.
Prediction: +5–15 % compression on bit-packed inputs (TTI/ints), ≈ 0 on zeros (bitwidth 0 →
no words); bytes identical.
Measured: (pending)
Result (gfx1100): compression ×0.999 / ×0.999 / ×1.010 at saturation (ints / zeros / TTI), x0 ×0.993 / ×1.00 / ×1.008 — neutral: the per-word divisions were not on the critical path (the inner packing loop and LDS reads dominate). Bytes identical. Kept as neutral (no measurable cost, less VALU work per word).
Result (gfx942): compression ×0.988 ints / ×1.001 zeros / ×1.007 TTI at saturation, x0 ×0.988 / ×1.003 / ×1.007 — neutral (−1 % on ints within drift); bytes identical. Neutral-kept.
Verdict: (pending)

### CAS-C2 — RLE compression: per-thread slab in registers                              Category: C4   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_rle_compress`, new template parameter `max_num_inputs`)
Change: `block_rle_compress` keeps its element→thread assignment (`num_inputs_per_thread`
consecutive elements per thread) but reads the slab from LDS once into a register array of
`max_num_inputs / threadblock_size` elements (4096-B chunk: int 8 @128 / 4 @256 threads, u8
32 / 16) plus the one element after it; the run-end count and the compaction pass then run on
registers. The inherited code re-read every element from LDS three times (count pass once,
compaction pass twice — `input_buffer[idx + 1]` and `input_buffer[idx]`). On by default on AMD
(`ARCTO_CASCADED_RLE_SLAB`), off on the CUDA backend.
Why (mechanism): LDS reads at a lane stride of `num_inputs_per_thread × sizeof(T)` (16 B for
int) are 4–8-way bank-conflicted on wave64 (`(addr/4) mod 32` repeats every 8 lanes), so each
of the three passes over the chunk cost several conflicted `ds_read_b32` per element; the slab
load pays that once (and is vectorisable to `ds_read_b128` when alignment is visible — a follow-up),
the rest is VALU. Same run ends in the same order: bytes identical.
Prediction: +5–15 % compression on RLE-using configurations (default: 2 RLE layers), more for
small types; ≈ 0 on decompression; bytes identical. Register cost: the u8 instantiation at 128
threads holds 32 slab registers — watch `-Rpass-analysis` for spills.
Measured: (pending)
Result (gfx1100): compression ints_mixed ×1.187 at x0 (78.2 → 92.8 GB/s) / ×1.145 at x32 (116.7 → 133.6), TTI ×1.147 at x512 (99.3 → 113.9), zeros ×1.097 at x512 (185 → 203); decompression unchanged; bytes identical. The three LDS passes over the chunk were the second hot spot after C1's collectives.
Result (gfx942): compression ints ×1.166 at x32 (273 → 318 GB/s) / ×1.17 at x0, TTI ×1.162 at x512 (274 → 318) / ×1.225 at x0, zeros ×1.08 at x512 / ×1.10 at x0; decompression unchanged; bytes identical.
Verdict: (pending)

### CAS-S6 — 32-bit block scan in RLE compression                                      Category: C2   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_rle_compress`)
Change: the exclusive sum of per-thread run-end counts ran on `size_type` = `size_t` (64-bit);
the values are in-chunk counts (< 65536 by construction), so the scan, the per-thread count and
the output index are now `uint32_t`. `*num_outputs` (shared, `size_type`) receives the 32-bit
total as before.
Why (mechanism): 64-bit adds/compares are VALU pairs (`v_add_co` + `v_addc`) and the scan's
LDS exchange moves 8 B instead of 4 B per lane; a pure-width change — same sums.
Prediction: +1–3 % compression (one scan per RLE layer); bytes identical.
Measured: (pending)
Result (gfx1100): compression ints_mixed ×1.012 (x32), zeros ×1.016 (x512), TTI ×1.006; decompression unchanged; bytes identical — as predicted (+1–2 %).
Result (gfx942): compression ×1.026 ints, ×1.042 zeros, ×1.018 TTI at saturation (x0 ×1.019 / ×1.044 / ×1.017); decompression code untouched (the ints x32 decomp row moved −7 % between C2 and S6 — drift on this node; x0 rows ≈); bytes identical.
Verdict: (pending)

### CAS-S5 — 16-byte cooperative copies for the chunk/layer streams                  Category: C3   Status: PENDING
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_copy_words`, `block_copy_elements`, `block_write`,
`block_read`, chunk load, final store), `src/lowlevel/CascadedBatch.hip`,
`src/highlevel/CascadedHlifKernels.hip` (`alignas(16)` on the LDS arenas)
Change: the four block-cooperative copies — chunk load (global → LDS), layer-output write
(LDS → global, `block_write`), layer-input read (global → LDS, `block_read`) and the final
store (LDS → global) — go through one helper that, on AMD, moves 16 B per lane
(`uint4`) when both pointers are 16-B aligned (runtime check) and words/elements otherwise; the
LDS arenas get `alignas(16)` so the LDS side always qualifies. CUDA keeps the word loops.
Why (mechanism): `global_load_dwordx4` / `ds_read_b128` move the same bytes with 4× fewer
instructions and `s_waitcnt` round trips; for the default `int` type the chunk load and the final
store are 4 KB each per chunk, the layer arrays 1–4 KB. Same bytes in the same order.
Prediction: +1–4 % on both directions for the default type (the layers dominate), more for
1-byte types whose element loops were byte-wide; bytes identical.
Measured (vs S6 c375cab, 30 reps): gfx1100 compression ints ×1.16 (x0) / ×1.21 (x32), zeros ×1.41
(x512), TTI ×1.18 (x512); decompression ×0.99–1.00 (zeros x512 ×0.976). gfx942 compression ints
×1.07 / ×1.11, zeros ×1.22, TTI ×1.15 (x512) / ×1.045 (x0); decompression ints ×0.875 (x32) /
×0.96 (x0), TTI ×0.93 (x512) / ×1.06 (x0), zeros ×1.04 — but the following commit (H1, compression
only) read TTI decompression ×0.72 on the same node minutes later, so the gfx942 decompression rows
of this window are not trustworthy. Bytes identical, tests green.
Result: the compression-side copies (chunk load, layer write) are a clear win on both parts; the
decompression final store gains nothing on gfx1100 and is at best neutral on gfx942.
Verdict: SPLIT (CAS-S5s, next commit): keep the 16-B helper for the chunk load, `block_write` and
`block_read`; the decompression final store returns to the element loop.

### CAS-S5s — keep the element loop for the decompression final store                    Category: C3   Status: COMMITTED
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`cascaded_decompression_fcn`)
Change: only the final LDS → global store reverts to the inherited element loop; everything else of
S5 stays. Bytes identical.
Measured: (final sweep)

### CAS-H1 (measured) — the zero-fill costs compression ints ×0.985 / zeros ×0.98 / TTI ≈ on gfx1100
and ×0.97 / ×0.95 / ×0.97 on gfx942 (x32/x512; thread-0 serial byte stores per bit-pack call); kept
for byte-determinism.

### CAS-H1 — deterministic compressed bytes: zero the never-written padding            Category: C8 (hygiene)   Status: COMMITTED
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`block_bitpack`, `block_write`, `do_cascaded_compression_kernel`)
Found by: `test_cascaded_coverage` ("every layer configuration…" — two compressions of the same
input compared byte for byte) fails on the *baseline* for `char, rles 0, deltas 0, bp 1` on
gfx1100 and gfx942. The two outputs decode correctly; they differ in bytes the kernel never
writes: (1) the bit-pack header gap between the frame of reference (`sizeof(data_type)` B) and
the 4-B-aligned bitwidth word (1-/2-byte types: 3/2 bytes), and between that word and the
`data_type`-aligned data (8-byte types: 4 bytes); (2) the padding of the last word of an unpacked
array whose byte size is not a multiple of 4 (`block_write` copied `roundUpTo(out_bytes, 4)`
bytes straight from the LDS buffer); (3) the gap word between RLE offsets and the delta header in
the 64-B chunk-metadata staging area for 8-byte types. All three came from LDS scratch (previous
chunk / layer data), so the stream was format-correct but not reproducible — inherited from
nvCOMP 2.2, and the reason no byte-identical gate could exist for Cascaded.
Change: thread 0 zeroes the header gaps after `get_for_bitwidth`; `block_write` copies full
words and writes the tail word masked; the chunk-metadata area is zeroed once per block.
Cost: a few scalar stores per layer; compressed sizes unchanged; decompressors (ARCTO and nvCOMP)
ignore the bytes. Effect: Cascaded output is now byte-deterministic, so the sweeps' gates and the
coverage test can compare bytes across commits and architectures.
Measured: (part of the sweep; expected ≈ 0)
### Cumulative FINAL on gfx942 (MI300A, wave64), base 777135f → head ddf1fd0 (cas6): compression ints ×2.60 (x0) / ×3.17 (x32), TTI ×3.58 (x0) / ×3.82 (x512), zeros ×1.60 (x0) / ×2.10 (x512); decompression ints ×1.40 (x0) / ×1.43 (x32), zeros ×3.86 (x0) / ×2.49 (x512), TTI ×1.54 (x0) / saturation inside the node's noise band (×0.90 vs a bimodal base row, ×1.23 vs the earlier base). Bytes identical; full coverage matrix green (72 684 assertions).

### Cumulative FINAL on gfx1100 (wave32), base 777135f → head ddf1fd0 (cas6): compression ints ×2.07 (x0) / ×2.16 (x32), TTI ×2.26 (x0) / ×2.32 (x512), zeros ×1.61 (x512); decompression ints ×1.16 (x0) / ×1.28 (x32), zeros ×4.3 (x0) / ×6.4 (x512), TTI ×0.98 (x512). Bytes identical on the ladder; full coverage matrix green.

### Cumulative on gfx942 (MI300A, wave64), base 777135f → S6 c375cab (S5/H1 rebuilt, measured separately): compression ints ×2.55 (x0) / ×2.98 (x32), TTI ×3.58 (x0) / ×3.39 (x512), zeros ×1.62 (x0) / ×1.82 (x512); decompression ints ×1.33 (x0) / ×1.38–1.48 (x32, drift band), zeros ×4.45 (x0) / ×3.0 (x512), TTI ×1.67 (x0) / ×1.34 (x512). Bytes identical on the ladder throughout. Order of contribution: S1 (256-thread blocks) > C1 > S4 > C2 > D2/D5 > S6; D1 is the zeros lever with an ints cost to tune.

### Cumulative on gfx1100 (wave32), base 777135f → S6 c375cab (S5/H1 rebuilt, measured separately): compression ints_mixed ×1.80 (x32), TTI ×1.98 (x512), zeros ×1.17 (x512); decompression ints_mixed ×1.21 (x32), zeros ×7.8 (x512), TTI ×0.985 (x512). Bytes identical on the ladder throughout; gfx942 pending.

### CAS-H2 — compression hang: a delta layer on a chunk with no elements left               Category: C8 (correctness)   Status: COMMITTED
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`do_cascaded_compression_kernel`)
Found by: `test_cascaded_coverage` ("every layer configuration…") — once CAS-H1 let the determinism
check pass, the test case ran on past it and the gate never returned: the GPU sat at 100 % for
20+ minutes (first seen in the diagnostic run, then in the cas3 gate on gfx1100). Per-configuration
timing shows every configuration round-trips in ~2.5 ms until `char, rles 3, deltas 2, bp 1`.
Mechanism: with `num_deltas` ≥ 2, a 1-element partition (or any chunk an RLE layer collapses to
one run — e.g. all-zero data with `{1 RLE, 2 deltas}`) reaches the second delta layer with zero
elements; `block_delta_compress` loops `element_idx < input_size - 1` on an unsigned `size_t`, i.e.
to SIZE_MAX — an effectively infinite loop on the GPU. Inherited from nvCOMP 2.2 (same code); the
default `{2 RLEs, 1 delta}` never reaches it, which is why it went unnoticed.
Change: before a delta layer, a chunk with `num_elements_current_chunk == 0` takes the existing
raw-copy fallback of the whole partition (`use_compression = false`), the only format-consistent
outcome since the decoder applies every configured layer unconditionally.
Effect: no change to any stream that did not hang; degenerate partitions now compress (raw) instead
of hanging. The coverage test now runs its full matrix (~1 s of GPU time).
### CAS-H3 — decoder applied the inverse layers in the wrong order for 0 < num_RLEs < num_deltas   Category: C8 (correctness)   Status: COMMITTED
Commit: (this commit)  (branch `opt/cascaded-2026-08`)
Files: `src/CascadedKernels.hiph` (`cascaded_decompression_fcn`)
Found by: `test_cascaded_coverage` after CAS-H1/H2 let the matrix run to `char, rles 1, deltas 2,
bp 1` (partition of 1024 elements): `REQUIRE(back == inputs[i])` — decompressed data differs from
the input, status success. The compressor applies, per iteration k, `[RLE if k < num_RLEs]` then
`[delta if k < num_deltas]` (so {1, 2} is RLE, Δ, Δ); the decoder decided the order with the
remaining counters — `delta_remaining >= rle_remaining` for a delta, then `rle_remaining >=
delta_remaining` (after the decrement) for an RLE — which undoes RLE before the first delta for
{1, 2} (Δ⁻¹, RLE⁻¹, Δ⁻¹). The rule is equivalent to the correct order exactly when num_RLEs ≥
num_deltas or num_RLEs == 0, i.e. for nvCOMP's default {2, 1} and the pure-delta configurations —
which is how it survived; inherited from nvCOMP 2.2.
Change: the decoder mirrors the encoder's layer indices: for k = max−1 … 0, undo delta if
k < num_deltas, then RLE if k < num_RLEs; the remaining counters still select the headers/offsets.
Effect: configurations with more delta than RLE layers now round-trip; every other configuration
executes the identical sequence (bytes and order unchanged). Perf-neutral.
Measured: (gate in cas5 / cas3b)
### Resource evidence — gfx942 (MI300A), ROCm 7.0.1, baseline kernels (before CAS-S2/S1/C1)
`-Rpass-analysis=kernel-resource-usage`, wave64, 128-thread blocks (inherited): `cascaded_compression_kernel<int,128,4096>` SGPR 106 / VGPR 71 / LDS 13456 B per block / 9 SGPR spills / est. 7 waves per SIMD; `cascaded_decompression_kernel<4B,128,4096>` SGPR 106 / VGPR 73 / LDS 13192 B / 12 SGPR spills / est. 6 waves per SIMD; `get_decompress_size_kernel` 22 / 10 / 0 LDS / 8. The SGPR spills (scalar state of the nested RLE/delta/bit-pack passes) and the 13 KB of LDS per 128-thread block (≤4 blocks per CU by LDS) are the two structural costs to attack (CAS-S2 launch bounds already committed; CAS-D5 LDS shrink queued).
