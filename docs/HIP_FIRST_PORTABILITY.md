# HIP-first, NVIDIA-portable — rules for ARCTO

ARCTO is written once, in HIP, and is expected to (a) run at AMD-native speed on CDNA and RDNA and
(b) still build and run on NVIDIA GPUs without a CUDA-specific source tree. This document states the
rules that make both true at the same time, each with the evidence from the August-2026 optimization
campaign (numbers in `OPTIMIZATION_GAINS_2026-08.md`, per-commit entries in
`docs/experiments/*.md`). It is meant to be read before adding an optimization, and to be cited in
the write-up as the portability contract.

## 0. The shape of the port

* **One source tree, two backends.** The device sources are HIP. On AMD they compile with hip-clang
  through CMake's HIP language. On NVIDIA, `-D CUDA_BACKEND=ON` compiles *the same files* with
  `nvcc` after `include/cuda_shim/hip/hip_runtime.h` maps the HIP runtime names to CUDA ones
  (`hipMalloc` → `cudaMalloc`, …) and `src/arcto_hipcub.hiph` maps `hipcub` → `cub`. There is no
  second implementation to keep in step, and nothing in the AMD path depends on the CUDA framework.
* **The public API is ARCTO's own** (`arctoBatched*`), not nvCOMP's. Benchmarks and tests therefore
  build unmodified on both platforms; comparisons against nvCOMP are a separate, explicitly
  compiled backend of the comparison harness (`benchmarks/benchmark_compare_chunked.cu`), never an
  `#ifdef` inside the library.
* **The portable build must not regress by construction.** Every optimization is *added* behind a
  platform or wave-size condition and the inherited nvCOMP code path stays as the fallback, so the
  NVIDIA build keeps the behaviour it had before the change unless the change was measured there.

## 1. Gate by capability, not by vendor

Three different conditions get confused if they are all spelled `#ifdef __HIP_PLATFORM_AMD__`:

| Condition | Spelling | Use it for |
|---|---|---|
| "AMD-specific instruction or intrinsic" | `#if defined(__HIP_PLATFORM_AMD__) \|\| defined(__HIP_PLATFORM_HCC__)` | `__builtin_amdgcn_*`, DPP-based primitives, `s_sleep` |
| "wave64 vs wave32 behaviour differs" | `#if defined(USE_WARPSIZE_32)` / `defined(USE_WARPSIZE_64)` | anything whose cost scales with lanes per wave |
| "measured on AMD, unmeasured on NVIDIA" | AMD gate **plus** a per-build knob | new access-width or scheduling paths |

Evidence that the middle row is not hypothetical — the same change, opposite signs:

* Snappy dword copies (SNP-D8): **+8 %** on gfx1100 (wave32), **−23 %** on gfx942 (wave64) →
  `ARCTO_SNAPPY_DWORD_COPIES` defaults on only under `USE_WARPSIZE_32`.
* `LITERAL_SECTORS = 8` (SNP-D11): **+4…+10 %** wave32, **−4 %** wave64 → wave32 default only.
* LZ4 vectorised copies (LZ4-D1): the E17 lineage measured them **positive on wave32 and negative
  on wave64**; this campaign found them **×2.2 on wave64** for literal/long-match data once the
  compression side was excluded — i.e. the right gate was not "wave size" alone but
  "wave size × which side of the codec", hence three knobs (`ARCTO_LZ4_VEC_COPY_DECOMP`,
  `_COMP`, `_MIN`).

**Rule.** A new fast path gets a named macro with a default per condition, never an unconditional
`#ifdef AMD`. The knob is documented in the header next to the default and stays after the
measurement, because the best default is workload-dependent (LZ4 text loses 20 % where seismic
data gains 120 %).

## 2. One source of truth for the wave size

`src/arcto_device_types.h` defines `arcto::warpsize`, `arcto::uwarpsize` and the `warp_mask_t`
types from `USE_WARPSIZE_32` (AMD default: wave64; CUDA backend: 32). Everything else must derive
from it.

* A second, independent derivation is a silent-corruption trap: `LZ4Kernels.hiph` used to require
  `ENABLE_HIP_OPT_WARPSIZE64 && USE_WARPSIZE_64`, so a build passing only `-DUSE_WARPSIZE_64`
  produced 32-bit lane masks in LZ4 while the rest of the library ran wave64. Fixed, and now
  `static_assert(warpsize == arcto::warpsize, …)`.
* The build's wave size must equal the wavefront size the compiler targets:
  `static_assert(warpsize == __AMDGCN_WAVEFRONT_SIZE__)` in device passes catches a wave64 build of
  an RDNA target that forgot `-mwavefrontsize64`, and a wave32 build of gfx9.
* At run time `HipUtils::check_wave_size()` refuses only what cannot work: a **wave32 build on a
  device whose native wavefront is 64**. A wave64 build is valid everywhere, because RDNA executes
  wave64 code objects — which is what makes "gfx1100 in either mode" a real choice rather than a
  rebuild-and-hope.

**Rule.** Never write `32`, `64`, `0xffffffff` or `warpSize` in device code; use `warpsize`,
`warp_mask_t`, `LANE_MASK_FULL`. `warpSize` in particular is a non-constexpr built-in on HIP and
cannot size a type or a template parameter.

## 3. Wave-level primitives: prefer the library, then measure the intrinsic

| Item | Change | Measured |
|---|---|---|
| CAS-S4 | `hipcub::BlockScan<…, BLOCK_SCAN_WARP_SCANS>` instead of hipCUB's default | +16–19 % compression, +10–15 % decompression on gfx942; +3 % on gfx1100 |
| SNP-D2 | rocPRIM `warp_scan`/`warp_reduce` (DPP) instead of shuffle trees | +1–5 % on both |
| SNP-D3 | `__builtin_amdgcn_readfirstlane` for wave-uniform broadcasts | +3–5 % gfx1100, +2–11 % gfx942 |
| LZ4-D5 | the same `readfirstlane` idea in LZ4's serial sequence parser | **−5…−8 %** on short-sequence data; reverted |

Reading: hipCUB/rocPRIM map to DPP row operations and `ds_swizzle` on AMD and to CUB on NVIDIA, so
one line of code is fast on both — that is the first thing to try. A hand-written AMD intrinsic is
worth it when the operation is *uniform* (a broadcast) and on the critical path, but the same
intrinsic in a latency-bound serial chain costs more than it saves (LZ4-D5): the SALU/VALU
hand-offs add latency the wave cannot hide. **Rule.** Library collective first; intrinsic second,
behind a knob, and only if the per-kernel measurement says so.

## 4. `__launch_bounds__` does not mean the same thing on the two platforms

```
CUDA: __launch_bounds__(maxThreadsPerBlock, minBlocksPerMultiprocessor)
HIP : __launch_bounds__(maxThreadsPerBlock, minWavesPerExecutionUnit)   // amdgpu_waves_per_eu
```

The second argument caps the VGPR budget on AMD (CDNA: ≤ 64 VGPRs for 8 waves/SIMD, ≤ 72 for 7,
≤ 80 for 6, ≤ 96 for 5). `ARCTO_LAUNCH_BOUNDS(threads, min_waves_per_eu, min_blocks_per_sm)`
(in `src/common.h`) states both intents and expands to the right form per platform.

Measured: stating the *real* block size is free and always right (CAS-S2, LZ4-D7: neutral, and it
documents the launch); *forcing* an occupancy target is not — `MIN_WAVES_PER_EU=8` on the LZ4
decoder cost 8–13 % at saturation on gfx942, because the compiler bought the eighth wave with
spills. **Rule.** Always bind the first argument; touch the second only with a resource report and
a measurement.

## 5. Access width is a register decision, not only a bandwidth decision

Wider per-lane accesses (dword literals/copies, `uint4` cooperative copies) were the biggest
memory-side wins of the campaign (Snappy literals +9…+14 %, Cascaded 16-byte copies +7…+22 %
compression), and their cost is registers:

* LZ4's decoder went from **66 VGPRs / 7 waves per SIMD** to **84 VGPRs / 5 waves** with the dword
  copy paths compiled in (`-Rpass-analysis=kernel-resource-usage`, `-D ARCTO_DEVICE_REPORTS=ON`).
  That is why the same build is ×2.2 on long copies and −24 % on short-sequence text: the latter is
  latency-bound and loses two waves of latency hiding.
* Cascaded's decompression *final store* gained nothing from 16-byte copies while its compression
  side gained 7–22 % → the width change was kept only where it paid (CAS-S5s).

**Rule.** Every access-width change is accompanied by the resource report before/after, and by a
latency-bound input in the benchmark set (short-sequence text, small batch) — not just the
throughput-friendly one.

## 6. Occupancy levers only pay where the kernel is occupancy-bound

`unsnap_kernel` sits at 50 VGPRs / 8 waves per SIMD with no spills on gfx942 and is bounded by LDS
residency (6.1 KB/block → 10 blocks/CU) and by its serial decode chain: launch-bound tuning there
was dropped after the report (SNP-D9, not pursued). Cascaded's 13 KB/block was the opposite case:
the LDS shrink (CAS-D5) and 256-thread blocks on wave64 (CAS-S1, ×1.25–1.7 decompression, ×1.5
compression) were the two largest CDNA wins. **Rule.** Read the report before proposing an
occupancy item; state which of (registers, LDS, waves per block) is the limiter.

## 7. Runtime-API differences are their own class of bug

The hipified code inherited `#if CUDART_VERSION >= 11020` around `cudaMallocAsync`. On AMD
`CUDART_VERSION` is never defined, so **every** HLIF scratch allocation silently took the
synchronous `hipMalloc`/`hipFree` path — and `hipFree` synchronises the device. Replaced by
`ARCTO_ASYNC_SCRATCH` (HIP ≥ 6 or CUDA ≥ 11.2). The same class of bug in the vendored zfp cost
2–4× per-call latency on small fields (ZFP-H2a).

**Rule.** After a hipify-style port, grep for `CUDART_VERSION`, `__CUDA_ARCH__`, `__CUDACC_VER_*`
and every `#if` that mixes vendor macros with feature tests; a macro that is *undefined* silently
evaluates to 0 and disables the fast path instead of failing the build. Guard comparisons with
`defined(...)` so `-Wundef` can find them.

## 8. Numerics and bytes are not allowed to move

Every kept commit of this campaign produced **byte-identical compressed output** (exact-bytes ladder
over the fixture set + round-trip tests), and ZFP is additionally gated by `test_zfp_payload_exact`
(308 cases byte-exact against `zfp_exec_serial`). That is what allows an optimization to be judged
on speed alone, and what makes an AMD-only fast path safe: a stream written on AMD is readable by
nvCOMP/zfp on NVIDIA and vice versa.

**Rule.** No `-ffast-math`, no reassociation of floating-point reductions, no "equivalent" bit
layouts. When a change alters the output at all (e.g. zeroing padding, CAS-H1), it is a separate
commit with its own justification.

## 9. Build system

* HIP language target with hip-clang (not `hipcc` as a compiler variable), `CMAKE_HIP_ARCHITECTURES`
  for the target list, `-D CMAKE_POLICY_VERSION_MINIMUM=3.5` for the vendored zfp's old CMake.
* `USE_WARPSIZE_32=ON` is refused for gfx9 targets; with the default wave64, RDNA targets are
  compiled with `-mwavefrontsize64` so one library serves every listed target at one compile-time
  wave size.
* CUDA backend: CUB lives in `<toolkit>/include` (CUDA 12) or `<toolkit>/include/cccl` (CUDA 13);
  both are searched.
* `-D ARCTO_DEVICE_REPORTS=ON` turns on `-Rpass-analysis=kernel-resource-usage` and `--save-temps`
  for the register/LDS/occupancy evidence the rules above require.

## 10. Checklist for a new optimization

1. Which condition is it? (AMD-only intrinsic / wave size / access width / occupancy / host side)
2. Is there a library primitive that is already fast on both platforms?
3. Does the inherited path remain the default where the change is unmeasured?
4. Resource report before/after (VGPR, LDS, waves, spills).
5. Byte-exact gate + round-trip tests green.
6. Measured on both wave sizes, at saturation **and** at small batch, with at least one
   latency-bound input.
7. Knob left in place, default set from the measurement, entry in `docs/experiments/<branch>.md`
   (hypothesis → mechanism → prediction → measurement → verdict).
