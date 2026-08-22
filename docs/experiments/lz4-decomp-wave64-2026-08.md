# LZ4 decompression on wave64 (CDNA) — experiment log (branch `opt/lz4-decomp-wave64-2026-08`)

Base: `origin/opt/curated` (E17 curated LZ4 lineage: LDS hash table, wave32-only copy
vectorisation / hybrid repeat / warp LSIC gated behind `USE_WARPSIZE_32`) + the benchmark per-rep
CSV and the `-x` duplication fix cherry-picked from the chore/test branches. One commit per change;
IDs are the LZ4-D* items of `docs/AMD_OPTIMIZATION_MAP.md`; categories C1..C8 as in the Snappy logs
(C1 sync/scheduling, C2 wave-level primitives, C3 memory access width/cache policy, C4 LDS layout,
C5 occupancy/registers/launch bounds, C6 launch geometry/compile-time tunables, C7 compiler flags,
C8 host side). Gates: `test_lz4`, `test_random_lz4`, `test_lz4batch_c_api`; exact-bytes ladder;
`benchmark_lz4_chunked` on tti / synth_binary / words at saturation and small batch (both nodes).
Baseline resource usage (gfx942, ROCm 7.0.1, from the Snappy branch's report build):
`lz4DecompressBatchKernel` SGPR 74 / VGPR 65 / LDS 1024 B / est. 7 waves per SIMD, no spills.

### LZ4-D7 — `__launch_bounds__` on the batched decompression kernel                    Category: C5   Status: PENDING
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/lowlevel/LZ4CompressionKernels.hip`
Change: `lz4DecompressBatchKernel` carries `__launch_bounds__(LZ4_DECOMP_THREADS_PER_CHUNK *
LZ4_DECOMP_CHUNKS_PER_BLOCK)` (128 threads on wave64, 64 on wave32), and
`-DARCTO_LZ4_DECOMP_MIN_WAVES_PER_EU=N` adds the occupancy target as the second argument (HIP:
min waves per EU; CUDA: min blocks per SM). Default: bounds only, no occupancy target.
Why (mechanism): without bounds hip-clang budgets registers for a 1024-thread workgroup; at 65
VGPRs the kernel sits at 7 waves/SIMD on gfx942 — one VGPR over the 64-register step for 8 waves.
The bound alone lets the compiler schedule for the real block; the flag variant pins the budget
so the re-test of the wave32-only vectorisation (LZ4-D1/D2) on wave64 is not confounded by a
VGPR step (the earlier wave64 losses were suspected to be exactly that).
Prediction: bounds only 0–3 %; `MIN_WAVES_PER_EU=8` 0–8 % if the 65th VGPR was the limiter, a
loss if the compiler spills to reach it (check `-Rpass-analysis`); bytes identical.
Measured: (pending) gfx942 (+ gfx1100 regression check).
Result: (pending)
Verdict: (pending)

### LZ4-D2 — incremental source index in the repeat (overlapped-match) copy             Category: C1   Status: PENDING
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/LZ4Kernels.hiph` (`coopCopyRepeat`)
Change: the strided loop `dest[i] = source[i % dist]` (the wave64/CUDA path for every overlapped
match, and the wave32 path for matches ≤ 2 × blockDim) keeps the lane's source index
incrementally: `r = lane % dist`, `step = blockDim.x % dist` once, then `r += step; if (r >= dist)
r -= dist` per iteration. Exact (r, step < dist, so one conditional subtract suffices).
Why (mechanism): a 32-bit remainder by a runtime divisor is ~30 VALU instructions on GCN/CDNA/
RDNA; the inherited loop paid it per byte. Repetitive inputs (zero runs, TTI's sparse regions) are
decoded almost entirely through this loop on wave64 (the doubling fast path is wave32-only in the
curated lineage). Bytes identical.
Prediction: gfx942 +10–30 % decompression on zeros / repetitive data, ≈ 0 on incompressible; gfx1100
small gain on short matches; bytes identical.
Measured: (pending)
Result: (pending)
Verdict: (pending)

### LZ4-D1 — vectorised copies and doubling repeat on wave64 (knob, default = curated)      Category: C3   Status: KNOB — variants pending
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/LZ4Kernels.hiph`
Change: the E17 wave32-only pieces — `coopCopyVec` (dword body fed by `unaligned_load32`, byte
head/tail) in `coopCopyNoOverlap`/`copyLiterals`, and the doubling expansion of long overlapped
matches in `coopCopyRepeat` — are selected by `ARCTO_LZ4_VEC_COPY` instead of `USE_WARPSIZE_32`:
default 1 on wave32, 0 on wave64 and CUDA (identical code to the curated lineage). Variants to
measure on gfx942: `-DARCTO_LZ4_VEC_COPY=1`, with and without `-DARCTO_LZ4_DECOMP_MIN_WAVES_PER_EU=8`.
Why (mechanism): on wave64 a byte copy issues one 64-B store per instruction; the dword path
issues 256 B per instruction with 4× fewer `s_waitcnt` round trips; the doubling turns a
period-`dist` match of length L into log2(L/dist) straight copies. The earlier wave64 loss (E17)
was measured without launch bounds; under LZ4-D7 the VGPR budget is explicit.
Prediction: with bounds, VEC_COPY=1 on wave64 +5–20 % on literal-/run-heavy inputs; if it still
loses, the cause is the extra registers (check `-Rpass-analysis` VGPRs vs 64) — then keep off.
Measured: (pending)
Result: (pending)
Verdict: (pending)
