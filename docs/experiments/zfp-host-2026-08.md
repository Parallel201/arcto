# ZFP (host path and HIP backend glue) — experiment log (branch `opt/zfp-host-2026-08`)

Base: `test/coverage-2026-08` (chore + coverage tests + benchmark fixes). The vendored zfp
(`third_party/zfp`, LLNL develop `cccbb9d` = the fork's `amd-hip` base) provides the HIP kernels;
this branch covers the ARCTO wrapper (`src/lowlevel/ZFPBatch.cpp`) and the fork's host-side glue
(`src/hip/device.h`, `src/hip/interface.cpp`, `src/share/device.h`) — kernel work goes to the
fork branch `opt/zfp-kernels-2026-08`. IDs are the ZFP-* items of `docs/AMD_OPTIMIZATION_MAP.md`;
categories C1..C8 as in the other logs (C8 = host side). Gates: `test_zfp_payload_exact`
(ZFP-T1, below), `test_zfp_bitcompat`, `test_zfp_canonical`; `benchmark_zfp_single`.

### ZFP-T1 — byte-exact gate: HIP payload and decoded field vs canonical serial            Category: gate   Status: COMMITTED (to run on both nodes)
Commit: (this commit)  (branch `opt/zfp-host-2026-08`)
Files: `tests/test_zfp_payload_exact.cpp`
Change: for float/double/int32/int64 × 3-D (61×33×18, 64×64×32) / 2-D (70×41) / 1-D (1000) ×
smooth / mostly-zero-blocks / random × fixed-rate {8, 12.5, 16, 24}, fixed-precision {12, 20},
fixed-accuracy {1e-3, 1e-6} (floats), the test requires: (1) two HIP compressions are identical;
(2) the payload region of the ARCTO stream (header and the ARCTO index trailer stripped) equals
the `zfp_exec_serial` payload of the same field byte for byte (sizes within one word of flush
padding); (3) the HIP-decoded field equals the serial-decoded field bit for bit.
Why: the existing zfp tests are tolerance-based; every kernel/host change below must be shown to
leave the bits unchanged, and the test also pins the partial-block, all-zero-block and word-sharing
(rate 12.5) paths.
Measured: gfx942 (MI300A, ROCm 7.0.1): **308 cases byte-exact vs canonical serial** on the
baseline (LLNL cccbb9d HIP backend) and on every commit of this branch — the HIP encoder/decoder
are bit-exact with the serial backend for every type/shape/mode combination tested, including
rate 12.5 (word sharing), partial blocks and all-zero blocks. gfx1100: (zfp1 run queued).

### ZFP-H1a — no HIP execution policy on throwaway streams (ARCTO side)                 Category: C8   Status: COMMITTED (measure with the batch)
Commit: (this commit)  (branch `opt/zfp-host-2026-08`)
Files: `src/lowlevel/ZFPBatch.cpp`
Change: `configure_stream(zfp, opts, set_exec)`; `arctoZFPCompressGetMaxOutputSize` and
`serialize_header` (the header-only stream inside `arctoZFPCompress`) no longer call
`zfp_stream_set_execution(zfp_exec_hip)` — they only need the mode parameters. The compress and
decompress streams that run the kernels keep it.
Why (mechanism): `zfp_stream_set_execution(hip)` calls `zfp_internal_hip_init`, which runs
`device_init()` every time: `hipMalloc` + a 1-thread kernel + `hipHostMalloc` (pinned, the
expensive one) + synchronous `hipMemcpy` + `hipHostFree` + `hipFree`. `arctoZFPCompress` paid it
twice (header stream + payload stream) plus once per `GetMaxOutputSize`; for a 64³ float field the
kernels themselves take ~0.1 ms, so this is a large share of the per-call latency on small and
medium fields. Zero effect on the bytes (no stream parameters change).
Prediction: compress call latency −1…−3 ms on small fields; none at 512³; bytes identical (ZFP-T1).
Measured (gfx942, `benchmark_zfp_single`, 3 runs × 30 iterations, medians of the per-run means):
64³ fixed_rate 16 compress 0.456 → 0.435 ms (×0.95), fixed_precision 16 0.555 → 0.437 ms (×0.79);
256³ ≈ (×0.96 / ×1.26 — noise). This benchmark's run-to-run band is ±20 % (the T1 commit, identical
code, reads ×0.74 and ×1.22 on the two 256³/64³ fixed-rate rows), so H1a alone is below the
resolution of this protocol; it is kept as part of the host-path set measured next.
Result: within noise alone; bytes identical.

### ZFP-H1b / ZFP-H2a / ZFP-D4 — fork-side host glue (vendored zfp branch `opt/zfp-host-2026-08`)   Category: C8   Status: COMMITTED (measure with the batch)
Commit: (this commit: gitlink `third_party/zfp` → 6332c5c)  (fork commits ef9ee46 H1b, 3246564 H2a, 6332c5c D4 on top of LLNL cccbb9d)
Files (fork): `src/hip/interface.cpp`, `src/share/device.h`, `src/hip/decode{1,2,3}.h`
Changes:
- H1b: `zfp_internal_hip_init` runs `device_init()` once per process (static flag); it used to
  run on every `zfp_stream_set_execution(zfp_exec_hip)`: hipMalloc + 1-thread kernel + hipHostMalloc
  (pinned) + synchronous hipMemcpy + hipHostFree + hipFree.
- H2a: `malloc_async` / `free_async` on HIP use `hipMallocAsync` / `hipFreeAsync` on the null stream
  (HIP ≥ 6), mirroring the CUDA path's `cudaMallocAsync`; the per-call staging buffers (compressed
  stream, index, field copies) come from the stream-ordered pool and `hipFree`'s implicit device
  synchronisation disappears from every call.
- D4: in fixed-rate mode (minbits == maxbits, granularity 1) the decode launchers compute the final
  bit position as `blocks * maxbits` on the host instead of hipMalloc + hipMemset + synchronous D2H +
  hipFree of a device counter per call; the kernels skip the counter store when it is NULL; a
  `hipDeviceSynchronize` keeps the "decode is complete on return" contract for device-resident
  fields (the counter copy used to provide it). Variable-rate modes keep the counter.
Why: the ZFP wrapper's per-call latency on small/medium fields is dominated by HIP API calls, not by
the kernels (`benchmark_zfp_single` 64³ float ≈ 1 MB). None of these touch the bit streams (ZFP-T1).
Prediction: per-call compress/decompress latency −2…−5 ms for 64³; ≈ 0 at 512³; bytes identical.
Measured (gfx942, vs the branch base, medians of 3 runs × 30 iterations; T1 gate: 308 cases
byte-exact at this commit): 256³ fixed_rate 16: compress 3.80 → 0.82 ms (×0.22 time, 17.7 → 81.5
GB/s), decompress 2.98 → 0.82 ms (×0.27, 22.5 → 82.2 GB/s); 256³ fixed_precision 16: compress
3.01 → 2.49 ms (×0.83), decompress 0.91 → 0.40 ms (×0.44, 73.8 → 167 GB/s); 64³ fixed_rate:
compress 0.456 → 0.270 ms (×0.59), decompress 0.346 → 0.254 ms (×0.73); 64³ fixed_precision:
compress ×0.66, decompress ×0.75. The per-call hipMalloc/hipFree of the staging buffers (tens of MB
at 256³: `hipFree` synchronises and returns memory to the OS) and the device counter round trip
were most of the wall time of a call on the MI300A (unified memory makes the H2D/D2H copies cheap,
so the API overhead dominated even more than predicted).
Result: LARGE — the ZFP host path was API-bound; bytes identical (ZFP-T1 308/308).
Verdict: KEPT (all three); gfx1100 pending (zfp1 run queued).
