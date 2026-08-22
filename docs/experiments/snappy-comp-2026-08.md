# Snappy compression — experiment log (branch `opt/snappy-comp-2026-08`)

Baseline: `main` @ `e3e1a1f` plus `chore/build-and-cleanup-2026-08` and `test/coverage-2026-08`
(build hygiene and the Snappy coverage gate — no kernel changes). The compressor (`snap_kernel`,
`src/snappy/compression.hiph`) is the hipified nvCOMP 2.2 design: one 2-wave block per chunk,
WARP0 emits the previous literal run and copy token while WARP1 searches the next 4-byte match
(12-bit hash map in LDS) and extends it; two `__syncthreads` per literal/copy pair. gfx1100
resource report (ARCTO_DEVICE_REPORTS, commit 798b0c4): 35 VGPRs, 8.2 KB LDS/block, occupancy
8 waves/SIMD (LDS-bound), 0 spills.

Plan (from `docs/AMD_OPTIMIZATION_MAP.md` §5.3): SNP-C5 → C2 → C1 → C3 (→ C4 if the resource
report on gfx942 shows a register constraint). Measured with `benchmark_snappy_chunked`
(compression column), the exact-bytes ladder and `test_snappy_coverage`, same protocol as the
decompression branch (`docs/experiments/README.md`). Branch is independent of
`opt/snappy-decomp-2026-08` (disjoint files except `snappy/config.h`); the two stack in an
integration branch.

---
