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
Measured: (pending — first run on gfx1100 and gfx942 establishes whether the baseline already
passes; a baseline failure is itself a finding).

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
Measured: (pending)
