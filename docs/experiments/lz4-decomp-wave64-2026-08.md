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
