# ARCTO on AMD — optimization results by category (August 2026)

Companion to `AMD_OPTIMIZATION_MAP.md` (the plan) and to the per-branch logs
`docs/experiments/*.md` (one entry per commit: change, mechanism, prediction, measurement, verdict).
This document aggregates the measured outcomes by the *kind* of optimization, so that each
category can be written up with all its evidence in one place. Every number below is a median of 30
repetitions from the per-repetition CSV of the chunked benchmarks, measured under the protocol of
§0; "bytes identical" means the exact-bytes ladder over `tests/data/*.bin` did not move and the
round-trip tests stayed green. Items still being measured are marked *pending*.

## 0. Method (shared by every item)

* Nodes: gfx1100 (RX 7900 XT, RDNA3, wave32, `-D USE_WARPSIZE_32=ON`) and gfx942 (MI300A, CDNA3,
  wave64), both ROCm 7.0.1 in the project's Singularity image, HIP language build with clang.
* One commit per change; each commit carries a category tag (C1..C8) and a log entry. A sweep
  builds every commit of a lineage incrementally, runs the codec's tests (gate), the exact-bytes
  ladder (gate), then `benchmark_*_chunked -p 65536` on three inputs at saturation (`-x 512`, or
  `-x 32` for the 16 MB text-like file) and at small batch (`-x 0`), 30 measured repetitions each,
  per-repetition CSV. Verdicts use medians and interquartile ranges; on the MI300A, saturation
  medians of the *same binary* drift ±5–8 % between runs (shared node, clock behaviour), so
  decisions there lean on the small-batch rows and on interleaved A/B runs.
* Categories: **C1** synchronisation/scheduling (barriers, serial sections, work balance);
  **C2** wave-level primitives (scans, broadcasts, readfirstlane, ballots); **C3** memory-access
  width and cache policy (dword/16-B paths, nontemporal); **C4** LDS layout and traffic;
  **C5** occupancy/registers/launch bounds; **C6** launch geometry and compile-time tunables;
  **C7** compiler flags; **C8** host side and infrastructure.

## 1. Results by category

### C1 — synchronisation, scheduling, serial sections

| Codec | ID | Change | gfx1100 (wave32) | gfx942 (wave64) | Verdict |
|---|---|---|---|---|---|
| Snappy decomp | SNP-D1 | `s_sleep` in the three spin-waits instead of busy polling | neutral | +2–4 % at head (60-rep A/B) | kept |
| Snappy decomp | SNP-D4 | wave-uniform decoder state computed on all lanes (no lane-0 + broadcast) | neutral (enabler for D3) | neutral | kept |
| Snappy comp | SNP-C2 | one barrier per literal/copy pair via double-buffered match state | −1…−4 % | −2…−6 % | reverted |
| Snappy comp | SNP-C1 | one-round-ahead prefetch of the match-search word | −4…−7 % | −4…−7 % | reverted |
| Cascaded decomp | CAS-D2 | multi-item (4/thread) block scan in delta decompression: 4× fewer scans+barriers per chunk | ints +8.6 % (sat), +5 % (x0); zeros −1.6 %; TTI ≈ | ints +9 % (sat), +8 % (x0); TTI +2 % | kept |
| Cascaded decomp | CAS-D1 → D1f → D1w | load-balanced expansion of long RLE runs: block-wide vote on an absolute run length (D1), on run > F × average (D1f), then **wave-local** (each wave balances its own runs' output; wave ballot, no block barrier) | D1: zeros ×7.9 (sat), ints −6.5 % at any cutoff/factor (the block vote itself); D1w: ints +8–9 % vs D1 (×1.28 sat / ×1.16 x0 vs base), zeros ×6.4 (sat) / ×4.3 (x0) | D1: zeros ×3.0 / ×4.45, ints −9 % / −4.6 %; D1w: ints +6–10 % vs D1 (×1.43 / ×1.40 vs base), zeros ×2.49 (sat) / ×3.86 (x0) | D1w kept |
| Cascaded comp | CAS-C1 | single-pass min/max in `get_for_bitwidth` (2 block reduces instead of 2 per 128 elements) | ints ×1.54, TTI ×1.70, zeros ×1.05 | ints ×1.44, TTI ×1.57, zeros ×1.14 | kept (largest Cascaded compression win) |
| Cascaded comp | CAS-C4 | incremental input-window indices in `block_bitpack` (no per-word integer division) | ≈ 0 (±1 %) | ≈ 0 (−1 %, in drift) | neutral-kept |
| LZ4 decomp | LZ4-D2 | incremental `i % dist` in the overlapped-match repeat copy | ≈ 0 (wave32 uses the doubling path) | ≈ 0 on non-repetitive inputs; zeros *pending* | kept (exact, cheaper per byte) |

### C2 — wave-level primitives

| Codec | ID | Change | gfx1100 | gfx942 | Verdict |
|---|---|---|---|---|---|
| Snappy decomp | SNP-D3 | `readfirstlane`/`readlane` for wave-uniform broadcasts (instead of `__shfl`) | +3–5 % | +2–11 % | kept |
| Snappy decomp | SNP-D2 | rocPRIM DPP warp scan / all-lanes reduce for the symbol-batch prefix sums | +1–5 % | +1–5 % | kept |
| Snappy decomp | SNP-D12 | 32-lane decode groups inside a wave64 (knob) | n/a | TTI +8 % (x0), words −9 % | input-dependent, knob off by default |
| Cascaded | CAS-S4 | `BLOCK_SCAN_WARP_SCANS` (rocPRIM `using_warp_scan`) for the three block scans | decomp ints +3 %, others ≈ | comp +16–19 %, decomp +10–15 % | kept (big on CDNA) |
| Cascaded comp | CAS-S6 | 32-bit scan type for in-chunk run counts (was `size_t`) | +1–1.6 % | +2–4 % | kept |
| LZ4 decomp | LZ4-D5 | token / LSIC / offset through `readfirstlane` (scalar sequence parsing) | ±1–3 % mixed | words −5…−8 %, others ≈; re-tested on the vectorised build: −2.5…−6 % everywhere | reverted |

### C3 — memory-access width and cache policy

| Codec | ID | Change | gfx1100 | gfx942 | Verdict |
|---|---|---|---|---|---|
| Snappy decomp | SNP-D6 | dword-granular prefetch of the compressed stream into the LDS ring | +6 % TTI (sat) | +2.5 % | kept |
| Snappy decomp | SNP-D8 literals | 4 bytes per lane for literal copies | +14 % TTI (x0) | +9 % TTI (x0) | kept on all AMD |
| Snappy decomp | SNP-D8 copies | 4 bytes per lane for non-overlapping copies | +8 % binary | −23 % binary | **split**: wave32 only (`USE_WARPSIZE_32`) |
| Snappy decomp | SNP-D7a/b | dword window reads in the serial decoder / per-lane symbol window | neutral | neutral | neutral-kept |
| Snappy comp | SNP-C3 | dword literal emission in `StoreLiterals` | +0.4…+0.7 % | −0.4…−4 % | reverted |
| Cascaded | CAS-S5 / S5s | 16-byte cooperative copies (chunk load, layer read/write) when 16-B aligned; the decompression final store measured neutral/negative and returned to the element loop | comp ints +16–21 %, zeros +41 %, TTI +18 %; decomp ≈ | comp ints +7–11 %, zeros +22 %, TTI +15 %; decomp store −7…−12 % at saturation (noisy window) → split off | kept (compression side) |
| LZ4 decomp | LZ4-D1 / D1s / D1w | E17 wave32-only vectorised copies and doubling repeat on wave64 (knobs: decompression side, compression side, short-length cutoff) | n/a (curated config kept; cutoff 128 ≈ neutral) | decompression side on: binary/TTI ×2.15–2.18 (x0), +30/+45 % (sat), zeros −5 % (sat), words −17…−24 % at every cutoff — the kernel grows from 66 VGPRs/7 waves to 84 VGPRs/5 waves per SIMD (report), the latency-bound short-sequence input loses two waves of latency hiding; compression side on: words compression −54 % | **wave64 default: decompression side on, compression side off**, knobs kept; occupancy-target variants (7/6 waves) *pending* |

### C4 — LDS layout and traffic

| Codec | ID | Change | gfx1100 | gfx942 | Verdict |
|---|---|---|---|---|---|
| Snappy decomp | SNP-D5 | 8-byte-aligned symbol queue, one 64-bit LDS access per symbol (was two 32-bit volatile) | +1–2 % | +2.5–4 % TTI | kept |
| Snappy comp | SNP-C5 | clear the whole hash map with 16-B LDS stores (fixes a wave64 half-clear bug) | neutral | +1–3 % | kept (correctness) |
| Cascaded decomp | CAS-D5 | RLE count bit-unpack scratch aliases the dead element buffer: LDS/block int 12.4 → 10.3 KB, u16 16.5 → 12.4 KB | ints +13 % (sat), zeros +17 % (sat), TTI −2 % | zeros +12 % (sat); ints/TTI ≈ | kept |
| Cascaded comp | CAS-C2 | RLE compression reads each thread's slab into registers once (was three bank-conflicted LDS passes) | ints +15–19 %, TTI +15 %, zeros +10 % | ints +17 %, TTI +16–22 %, zeros +8–10 % | kept |

### C5 — occupancy, registers, launch bounds

| Codec | ID | Change | gfx1100 | gfx942 | Verdict |
|---|---|---|---|---|---|
| (all) | BLD | `ARCTO_LAUNCH_BOUNDS(threads, min_waves_per_eu, min_blocks_per_sm)` macro with explicit per-platform semantics; `ARCTO_DEVICE_REPORTS` option (`-Rpass-analysis=kernel-resource-usage`, `--save-temps`) | — | — | infrastructure |
| Snappy decomp | (report) | `unsnap_kernel` 50 VGPRs, 8 waves/SIMD, no spills on gfx942; 44 VGPRs @16 waves on gfx1100 — no headroom for a launch-bounds item (SNP-D9 not pursued) | — | — | evidence |
| Cascaded | CAS-S2 | `__launch_bounds__(threadblock_size)` on the batched kernels | neutral | ±5 % (drift) | kept (enabler) |
| LZ4 decomp | LZ4-D7 | `__launch_bounds__(threads×chunks)`; optional waves-per-EU target | exactly neutral (interleaved A/B; identical resources: 91 VGPRs) | bounds neutral; `MIN_WAVES_PER_EU=8` −8…−13 % at saturation | bounds kept, no occupancy target |

### C6 — launch geometry and compile-time tunables

| Codec | ID | Change | gfx1100 | gfx942 | Verdict |
|---|---|---|---|---|---|
| Snappy decomp | SNP-D11 | `LITERAL_SECTORS=8` (two dwords per lane per literal step) | TTI +4 % (sat), +10 % (x0) | −4 % | **default 8 on wave32 only** |
| Snappy decomp | SNP-D11 | 8 KB prefetch ring | −23…−28 % | −21 % (sat), +17 % TTI (x0) | rejected (residency) |
| Snappy decomp | SNP-D11 | 4-sector prefetch granule | −25 % | ≈ 0 | rejected |
| Snappy decomp | SNP-D10 | one chunk per thread in the decompressed-size query kernel | neutral | neutral | neutral-kept |
| Cascaded | CAS-S1 | 256-thread blocks on wave64 targets (128 on wave32 and CUDA) | no-op | comp ×1.47–1.68, decomp ×1.25–1.70 | kept (largest Cascaded decompression win on CDNA) |

### C7 — compiler flags

| Item | gfx1100 | gfx942 | Verdict |
|---|---|---|---|
| explicit `-O3 -DNDEBUG` for device code (chore branch) | exactly neutral (0263261 ≡ main) | exactly neutral | kept for visibility (`Device flags` message at configure) |

### C8 — host side and infrastructure

| Item | Effect | Status |
|---|---|---|
| Benchmark `-x` duplication bug: `multi_file()` inserted a range of the vector into itself; with libstdc++ every reallocating round appended *empty* chunks — 160 of 8208 chunks at `-x 512` (503 MB instead of 513 MB). Snappy/LZ4 tolerate empty chunks (all relative comparisons stay valid, 2 % short of nominal); Cascaded reported `arctoErrorCannotDecompress` and the benchmark aborted on every `-x` run. | fixed (`reserve` + element-wise copy) | test/coverage branch |
| Cascaded compressor non-determinism (CAS-H1/H1b): bit-pack header gaps (1-/2-/8-byte types), tail word of odd unpacked arrays and a chunk-metadata gap word were never written — LDS scratch went into the stream (format-correct, not reproducible; inherited from nvCOMP 2.2) | zero-filled (folded into the header write); output now byte-deterministic; cost −1…−3 % compression | cascaded branch |
| Cascaded **decoder layer order** (CAS-H3): for `0 < num_RLEs < num_deltas` the inverse layers ran in the wrong order (RLE⁻¹ before the first Δ⁻¹) and the API returned wrong data with `arctoSuccess`; equivalent to the correct order only for `num_RLEs ≥ num_deltas` or `num_RLEs == 0` (nvCOMP's default and the pure-delta cases) — inherited from nvCOMP 2.2, found by the coverage test once H1/H2 let it reach `{1 RLE, 2 deltas}` | decoder mirrors the encoder's layer indices; the full 96-configuration × 8-type coverage matrix now passes on both nodes (72 684 assertions) | cascaded branch |
| Cascaded compressor **hang** (CAS-H2): with `num_deltas ≥ 2`, a chunk that runs out of elements before its last delta layer (1-element partition; all-equal chunk collapsed by an RLE layer) made `block_delta_compress` loop to `size_t(0) − 1` — an infinite GPU loop, inherited from nvCOMP 2.2 and invisible with the default `{2 RLEs, 1 delta}`; found by the coverage test once CAS-H1 let it run its full matrix | degenerate partitions take the raw-copy fallback | cascaded branch |
| Per-repetition CSV in the chunked benchmarks (`ARCTO_PER_REP_CSV`, `ARCTO_PER_REP_TAG`); `-F` short flag; README | enables medians/IQR per commit | chore branch |
| Coverage tests `test_snappy_coverage`, `test_cascaded_coverage` (every layer configuration × type × size profile, misalignment, determinism), `test_zfp_payload_exact` (ZFP-T1: HIP payload and decoded field byte-exact vs canonical serial) | gates for the optimization loops | test/coverage + zfp branches |
| `CONFIGURE_DEPENDS` on the test source glob | a new test file is picked up by an existing build dir | test/coverage branch |
| hipCUB include path from the imported target; deprecated generic `arctoDecompress*` API removed; MIT test CMakeLists; HLIF chunk-size table estimate from its actual type | hygiene | chore branch |
| ZFP host path: no HIP execution policy on throwaway streams (H1a); device warm-up once per process (H1b); `hipMallocAsync`/`hipFreeAsync` for the per-call staging buffers (H2a); fixed-rate decode bit count computed on host (D4) | call latency: 64³ fixed-rate compress ×0.30 (gfx1100) / ×0.59 (gfx942), decompress ×0.48 / ×0.73; 256³ fixed-rate compress ×0.82 / **×0.22**, decompress ×0.91 / **×0.27** (APU: no PCIe copies, so the API calls were the whole cost); ZFP-T1 308/308 byte-exact on both nodes at every commit | kept |

## 2. Outcome per codec (vs `main`, bytes identical, tests green)

* **Snappy** (`opt/snappy-2026-08`): decompression gfx1100 ×1.24 (binary) / ×1.39 (TTI) / ×1.075
  (words) at saturation and the same at small batch; gfx942 ×1.12 / ×1.23 / ×1.21 at small batch,
  ×1.08 / ≈1.00 / ×1.15 at saturation. Compression: neutral on gfx1100, +1–2.5 % on gfx942 (SNP-C5,
  which also fixes the wave64 half-cleared hash map). Three compression attempts (C1, C2, C3) were
  measured negative on both parts and reverted; the log keeps them.
* **Cascaded** (`opt/cascaded-2026-08`), final head vs base (`777135f`), full coverage matrix
  green, bytes identical: gfx1100 compression ints ×2.07 (x0) / ×2.16 (sat), TTI ×2.26 / ×2.32,
  zeros ×1.61 (sat); decompression ints ×1.16 / ×1.28, zeros ×4.3 / ×6.4, TTI ×0.98 (sat). gfx942
  compression ints ×2.60 / ×3.17, TTI ×3.58 / ×3.82, zeros ×1.60 / ×2.10; decompression ints ×1.40
  / ×1.43, zeros ×3.86 / ×2.49, TTI ×1.54 (x0; saturation in the node's noise band). Order of
  contribution: S1 256-thread blocks (CDNA) and C1 single-pass min/max, then S4 warp scans, C2
  register slab, S5 16-B compression-side copies, D2 multi-item delta, D5 LDS shrink, D1w wave-local
  RLE expansion (the zeros lever), S6; H1/H2/H3 fix three inherited defects.
* **LZ4 decompression on wave64** (`opt/lz4-decomp-wave64-2026-08`, base `origin/opt/curated`):
  final head vs base on gfx942 — decompression binary ×2.28 (small batch) / ×1.35 (saturation), TTI
  ×2.20 / ×1.12, zeros ×1.17 / ×1.37, words ×0.80 / ×0.75; compression unchanged; gfx1100
  configuration-identical (×0.995–1.005). Bounds neutral, modulo removal exact, D5 reverted; the
  decompression-side vectorised copies are the lever, and their cost on short-sequence data is the
  84-VGPR / 5-wave build; occupancy targets (7/6 waves) leave the per-chunk work unchanged and only move the unstable saturation rows, D5-on-top is negative — bounds-only stays.
* **ZFP** (`opt/zfp-host-2026-08` + vendored-zfp branch): the HIP path is bit-exact with canonical serial (T1, 308 cases, both nodes); the host path was API-bound — 64³ calls ×2–3.3 faster on both parts, 256³ fixed-rate ×3.6–4.6 on the MI300A APU, ×1.1–1.2 on the discrete RX 7900 XT. Kernel items (fork) not started.

## 3. Cross-cutting findings worth writing up

1. The wave32 ↔ wave64 split is real and per-mechanism: the same dword copy path is +8 % on RDNA3
   and −23 % on MI300A (Snappy copies); `LITERAL_SECTORS=8` +4…+10 % vs −4 %; the LZ4 vectorised
   copies ×2 on long runs and −25…−44 % on short sequences on wave64. Gating by `USE_WARPSIZE_32`
   (or by measured knobs) rather than by "AMD" is the portable answer; the CUDA backend keeps the
   nvCOMP paths untouched in every item.
2. Occupancy levers only pay where the kernel is occupancy-bound: Cascaded's 13 KB-per-block LDS
   (CAS-D5/S1) yes; Snappy's decoder (8 waves/SIMD, 10 blocks/CU by LDS) no — launch-bounds items
   were evidence-checked with the resource report before being pursued.
3. Serial sections dominate when present: the one-lane RLE expansion (×7.9) and the per-128-element
   collective pair in `get_for_bitwidth` (×1.5–1.7) were the two biggest wins of the campaign, both
   pure scheduling changes with identical output.
4. Measurement hygiene mattered as much as the kernels: the benchmark's duplicate-chunk bug, the
   Cascaded padding non-determinism, the MI300A run-to-run drift and one baseline outlier (LZ4-D7
   on gfx1100, −21 % that an interleaved A/B showed to be exactly 0) all shaped verdicts.
5. The coverage tests paid for themselves beyond the gates: three inherited Cascaded defects
   (non-deterministic bytes, a GPU hang for `num_deltas ≥ 2` on exhausted chunks, and a silent
   wrong-order decode for `0 < num_RLEs < num_deltas`) surfaced only because the new test walks every
   layer configuration for every type — none is reachable from nvCOMP's default configuration.
6. Register growth is the hidden price of vectorised copy paths on wave64: LZ4's decoder moved from
   66 to 84 VGPRs (7 → 5 waves per SIMD) with the dword paths compiled in — ×2.2 on long copies,
   −24 % on short-sequence data from the lost latency hiding — so such paths need either an
   occupancy target or a per-workload switch, not a blanket default.

## 4. Open / next

* Cascaded on gfx942 (S1 256-thread blocks, D5/D1 on CDNA), the D1 threshold, S5/H1 rebuilt.
* LZ4 wave64 defaults from lz4b; LZ4-D5 readfirstlane; zeros input for LZ4-D2.
* ZFP: T1 gate on both nodes, host-path latency (H1a/H1b/H2a/D4), then the kernel items in the fork
  (C1/D1 vectorised gather/scatter, C5 64-bit div/mod, C3 aligned slots).
* MI300A Snappy at saturation: counter-level profile (rocprof) — the per-chunk gains (+12–23 %) do
  not all reach the saturated regime.
