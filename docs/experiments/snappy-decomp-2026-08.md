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
