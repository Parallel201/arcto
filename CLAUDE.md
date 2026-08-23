# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What ARCTO is

Batched GPU compression for AMD (HIP/ROCm ≥ 6.1): NVIDIA nvCOMP **branch-2.2** (v2.2.0, `a6e4e64`)
hipified and hand-tuned for gfx90a (CDNA2, MI210/MI250), gfx942 (CDNA3, MI300X) and gfx1100
(RDNA3, wave32). Codecs: LZ4, Snappy, Cascaded (RLE+Delta+BitPack) and ZFP (vendored LLNL zfp,
HIP backend). ANS/GDeflate/Bitcomp wrappers exist but need proprietary libs (`ARCTO_EXTS_ROOT`).
An experimental CUDA backend (`-D CUDA_BACKEND=ON`) compiles the *same* HIP sources on NVIDIA so
ARCTO can be compared against nvCOMP itself — the portable build must never regress.

**This workstation has no ROCm/CUDA toolchain or GPU.** Everything here is static work; builds,
tests and benchmarks run on remote AMD/NVIDIA nodes. Do not try to build locally.

## Build, test, benchmark

```bash
git submodule update --init --recursive         # third_party/zfp is required (CMake errors otherwise)

# AMD (wave64: gfx906/gfx90a/gfx942)
cmake -S . -B build -D CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake \
  -D CMAKE_HIP_ARCHITECTURES="gfx90a;gfx942" -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_TESTS=ON -D BUILD_BENCHMARKS=ON
# gfx1100 (RDNA3) MUST add:  -D USE_WARPSIZE_32=ON   (wave size is a compile-time constant)
# NVIDIA backend:            -D CUDA_BACKEND=ON -D CMAKE_CUDA_ARCHITECTURES="80;90"
# Warnings-as-errors dev build: -D DEVEL=1 ; static lib: -D BUILD_STATIC=ON
cmake --build build -j

cd build && ctest                                # all tests
ctest -R '^test_lz4$' --output-on-failure        # one test by name (regex)
./build/bin/test_lz4 "[small]"                   # tests/ binaries land in build/bin; most are Catch2 (tag filters, -s)
./build/src/test/BitPackGPU_test                 # src/test and src/highlevel/test unit tests land next to their CMakeLists

HIP_VISIBLE_DEVICES=0 ./build/bin/benchmark_lz4_chunked -f data.bin -p 65536 -w 2 -i 10 -c true
./build/bin/benchmark_zfp_single -f field.bin    # ZFP modes + fidelity metrics
```

Benchmark flags: `-p` chunk size, `-P` pinned coalesced input, `-A` adaptive tiled staging,
`-R` per-phase host cost report, `-c` CSV, `-x` duplicate data (use it to saturate MI300X: 1600
chunks ≈ 1.3 waves/SIMD), `-g` GPU id, `-i`/`-w` iterations/warmup. Timing brackets only the
`*Async` call with events; the last iteration verifies the round trip. On the CUDA backend the
chunked benchmarks compile against **nvCOMP** (`#else` of `__HIP_PLATFORM_AMD__`) — that is the
cross-vendor baseline, not ARCTO.
Test fixtures: `tests/data/*.bin` (1 MB each, compiled in as `ARCTO_TEST_DATA_DIR`; env
`ARCTO_ZFP_TTI_DATA`/`ARCTO_ZFP_TTI_RSF` point at larger off-tree datasets). ZFP tests are
tolerance-based, not byte-exact.

## Code architecture

Two public API layers share one set of kernels:

- **Low-level batched C API** — `include/arcto/<codec>.h` → `src/lowlevel/<Codec>Batch.cpp`
  (arg checks, temp-space sizing, launch) → device code in `src/<Codec>Kernels.hiph`,
  `src/lowlevel/<Codec>*Kernels.hip`, `src/snappy/*.hiph`, `src/CascadedKernels.hiph`.
  Caller owns device pointers/sizes arrays and the temp buffer (`TempSpaceBroker` carves it).
- **High-level manager C++ API (HLIF)** — `include/arcto/<codec>.hpp` + `arctoManager.hpp` →
  `src/highlevel/<Codec>Manager.*` built on `ManagerBase`/`BatchManager`, with the chunking/header
  pipeline in `src/arcto_common_deps/hlif_shared.hiph` and per-codec `<Codec>HlifKernels.hip`.
  `arctoManagerFactory` reads the header so a buffer can be decompressed without knowing its codec.
- **ZFP** is not hipified nvCOMP: `src/lowlevel/ZFPBatch.cpp` wraps the vendored
  `third_party/zfp` (built static; `ZFP_WITH_HIP` on AMD, `ZFP_WITH_CUDA` on the CUDA backend) and adds
  a self-identifying index trailer so FIXED_PRECISION/ACCURACY streams decompress on GPU. The
  submodule pin is LLNL `cccbb9d`; AMD-side zfp work goes in the fork
  `github.com/cristianokunas/zfp` branch `amd-hip` (see `opt/curated`).
- **Host staging** (`include/arcto/host_batch*.h`, `src/lowlevel/HostBatch*.cpp`): pinned
  coalesced upload and the adaptive tiled-window cost model with per-arch windows.
- `src/BitPackGPU.*`, `DeltaGPU.*`, `RunLengthEncodeGPU.*` are the older per-stage Cascaded
  kernels (unit-tested in `src/test/`); the batched path is `src/lowlevel/CascadedBatch.hip`.

Cross-cutting conventions:

- `.hip` = device translation unit, `.hiph` = device header. Benchmarks are `.cu` but compiled as
  HIP (`set_source_files_properties(... LANGUAGE HIP)`); on the CUDA backend all `.hip` sources are
  relabelled `LANGUAGE CUDA` and `include/cuda_shim/hip/hip_runtime.h` maps `hip*` → `cuda*`.
- Wave size is compile-time: `USE_WARPSIZE_32` vs `ENABLE_HIP_OPT_WARPSIZE64` → `arcto::warpsize`
  / `warp_mask_t` (`src/arcto_device_types.h`) and the helpers/launch-bound macros in
  `src/amd_optimizations.h` (`WAVEFRONT_SIZE`, `wave_ballot`, `wave_scan_*`, `AMD_LAUNCH_BOUNDS_*`).
  Code that is wave-width dependent must handle both; a wave64 build running on gfx1100 is wrong.
- Element-type dispatch is via `src/type_macros.h` (`ARCTO_TYPE_ONE_SWITCH` etc.).
- AMD-only behaviour is gated with `#if defined(__HIP_PLATFORM_AMD__)` (and `USE_WARPSIZE_32` for
  RDNA3) so the CUDA backend keeps nvCOMP's inherited defaults — e.g. `ARCTO_KRESTRICT`,
  `ARCTO_LZ4_LDS_TABLE`, `ARCTO_LZ4_MAX_HASH_TABLE_SIZE` on the `opt/*` branches. Shared device
  code may only use HIP runtime symbols that the shim maps (`include/cuda_shim/hip/hip_runtime.h`);
  `__builtin_amdgcn_*`, `__builtin_nontemporal_*`, `amdgpu_waves_per_eu` need the AMD guard.
- HIP gotchas that differ from CUDA: the second argument of `__launch_bounds__` is **min waves
  per EU (SIMD)**, not min blocks per SM (the comment in `amd_optimizations.h` is wrong about
  this); Release HIP flags (`-O3 -DNDEBUG`) come from CMake defaults, the project adds only
  `-fPIC` and the platform/wave defines; `CUDART_VERSION` gates (e.g. `hipMallocAsync` in
  `ManagerBase.hpp`) are never true on HIP.
- Licensing: nvCOMP-derived files keep their NVIDIA BSD-3/Apache header verbatim and add the
  "MIT License / Modifications Copyright (C) 2025-2026 Cristiano Künas" block; new files are MIT
  (long form or one-line `Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.`).
  Per-file provenance is tracked in `NOTICES.md` — update it when adding/relicensing files.
  `third_party/zfp` is a submodule: never edit it in place; changes go to the fork and the pin moves.

## Optimization work — rules from the maintainer

- **Never change the base algorithm** (LZ4 stays LZ77 hash-table + sliding window, Snappy stays
  Snappy, Cascaded stays RLE/Delta/BitPack, ZFP stays bitstream-compatible with canonical zfp).
  Only kernel execution/memory aspects change: launch geometry, wave64/wave32 paths, LDS layout,
  registers/occupancy (`__launch_bounds__`, waves-per-EU), coalesced/vectorized loads, L1/L2/
  Infinity-Cache behaviour, warp primitives, atomics, compile flags.
- Ground proposals in AMD's own documentation and in the literature; cite sources.
- **One commit per change** so each can be benchmarked in isolation. Commit bodies state
  hypothesis → prediction → measured verdict (see `git log origin/opt/lz4-compress-2026-08`).
  Failed attempts are reverted with an explicit "Revert attempt X" commit and stay in history.
- **One branch per optimization aggregation**, off `main`, named `opt/<topic>-<yyyy-mm>`
  (e.g. `opt/lz4-compress-2026-08`); stacked combinations get their own branch
  (user shorthand: opt1, opt2, opt1+2). Curated integrations live on `opt/curated`.
- **Never add `Co-Authored-By` trailers. Never `git push`.** Committing is fine.
- Acceptance gates used so far: ctest green on wave64 and wave32, compressed bytes identical
  (or ratio deviation reported in exact bytes), round-trip bit-exact, throughput measured on
  gfx906/gfx90a/gfx942/gfx1100 with the chunked benchmarks.
- Before touching LZ4 compression, read the `origin/opt/*` and `origin/feature/kernel-opt`
  branches: LDS-resident hash table, small-table defaults, wave64 LDS claim-table twin search,
  64-VGPR `__launch_bounds__` pin, wave32 vectorized copies are already measured and kept there.
