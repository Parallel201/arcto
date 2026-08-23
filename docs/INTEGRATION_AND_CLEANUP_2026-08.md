# ARCTO — integration, cleanup and portability pass (August 2026)

Phase 2 of the campaign: put every optimization branch into one lineage, re-measure it against
`main` as it came out of hipify, remove what is not implemented or not used, make the tree behave
correctly in every supported AMD configuration (including gfx1100 in **either** wave mode), and
verify that "HIP-first, NVIDIA-portable" is a fact rather than an intention — by building and
testing the same sources on a GH200 and measuring them against nvCOMP.

Companion documents: `OPTIMIZATION_GAINS_2026-08.md` (the numbers),
`HIP_FIRST_PORTABILITY.md` (the rules), `NVIDIA_COMPARISON_2026-08.md` (ARCTO vs nvCOMP),
`AMD_OPTIMIZATION_MAP.md` (the plan), `docs/experiments/*.md` (per-commit logs).

## 1. Branches

| Branch | Contents |
|---|---|
| `bench/baseline-2026-08` | `main` + **only** the benchmark instrumentation (per-repetition CSV, the `-x` duplication fix). This is the "post-hipify, no optimization" measurement baseline: no kernel or library change of any kind. |
| `opt/claude/integration-2026-08` | `main` merged with all six work branches — `chore/build-and-cleanup`, `test/coverage`, `opt/snappy`, `opt/cascaded`, `opt/zfp-host`, `opt/lz4-decomp-wave64` (which itself carries the earlier `opt/curated` LZ4 lineage). All six merged without conflicts. |
| `chore/cleanup-2026-08` | the integration lineage + the 14 cleanup/portability commits below (**+865 / −8581 lines**). This is the head that was measured and that the numbers refer to. |

The merge branch is namespaced `opt/claude/...` to keep it clear of the pre-existing
`origin/opt/integration-2026-08`, which is part of the earlier curated LZ4 lineage and unrelated
to this merge.

## 2. What was removed

| Removed | Why | Size |
|---|---|---|
| ANS, GDeflate and Bitcomp wrappers (`include/arcto/{ans,bitcomp,gdeflate}.*`, their low/high-level stubs, benchmarks and tests, and the `ARCTO_EXTS_ROOT` / `find_package(ans\|bitcomp\|gdeflate)` discovery) | nvCOMP 2.2 implements these on top of **proprietary NVIDIA libraries**; ARCTO never carried anything but stubs that threw `arctoErrorNotSupported`. The HLIF format ids stay reserved so a foreign stream is still rejected with a clear error. | ~4 300 lines |
| Legacy per-stage Cascaded kernels: `BitPackGPU`, `DeltaGPU`, `RunLengthEncodeGPU`, `unpack.h` and their three unit tests | leftovers of nvCOMP's pre-2.0 Cascaded API; the batched path (`CascadedKernels.hiph`) is the implementation. Nothing but their own tests referenced them — and `BitPackGPU_test` was the one test that had been failing on both AMD GPUs since before this campaign. | ~3 300 lines |
| `src/amd_optimizations.h` | 372 lines that **no translation unit included**: wavefront constants (its NVIDIA branch even defined a 32-bit "wavefront mask"), block-size presets and hand-written ballot/shuffle helpers, all duplicating — with different answers — what `arcto_device_types.h` and `device_functions.hiph` define for real. Its one useful part, the `ARCTO_LAUNCH_BOUNDS` macro that documents the differing meaning of the second `__launch_bounds__` argument, moved to `src/common.h` where it is reachable. | 372 lines |
| The nvCOMP copy of the chunked benchmark template | the template carried a second, drifting copy of itself written against the nvCOMP API under `#else` of `__HIP_PLATFORM_AMD__` — unreachable in ARCTO's own CUDA backend, which has no nvCOMP headers. The benchmarks are now one source that uses ARCTO's API on both platforms. Comparisons against nvCOMP are a separate, explicit build of `benchmark_compare_chunked.cu`. | ~520 lines |

## 3. AMD adherence: one wave size per build, and gfx1100 in either mode

* **`arcto_device_types.h` is the single source of truth.** `LZ4Kernels.hiph` used to derive its own
  lane-mask width from `(AMD && ENABLE_HIP_OPT_WARPSIZE64 && USE_WARPSIZE_64)`, so a build that
  passed only `-D USE_WARPSIZE_64` got 32-bit lane masks in LZ4 while the rest of the library ran
  wave64. It now follows the shared header and `static_assert`s that the two agree.
* **Compile-time check**: `static_assert(warpsize == __AMDGCN_WAVEFRONT_SIZE__)` in device passes.
* **Build system**: `USE_WARPSIZE_32=ON` is refused for gfx9 targets (wave64-only hardware); with the
  default wave64, RDNA targets are compiled with `-mwavefrontsize64`, so one library serves every
  listed target at one compile-time wave size.
* **Run time**: `HipUtils::check_wave_size()` now refuses only what cannot work — a wave32 build on a
  device whose native wavefront is 64. A wave64 build is valid everywhere, since RDNA executes
  wave64 code objects. (Previously *any* mismatch threw, which made wave64 builds unusable on
  gfx1100 even though the hardware supports them.)

**Measured caveat — wave64 on RDNA builds and computes, but it is performance-pathological; wave32
stays the RDNA default.**

* ROCm's own libraries do not support it: rocPRIM derives its hardware wavefront size from the
  *architecture family* (ROCm 7.0.1 reports 32 for every gfx10/11/12 target, `ROCPRIM_NAVI`) and
  ignores `-mwavefrontsize64`, so asking it for 64-lane warp primitives fails a `static_assert`
  inside `rocprim/intrinsics/arch.hpp`. ARCTO detects that case and uses the portable shuffle trees
  the CUDA backend already used — identical results, a few percent slower — which is what makes the
  build possible at all.
* The runtime cost is far larger than that fallback, and it is **not** confined to the code that
  uses the ROCm collectives. On a wave64 gfx1100 build, two independent test binaries that take
  seconds in wave32 did not finish:

  | test | wave32 | wave64 (same GPU, same build tree) |
  |---|---|---|
  | `test_cascaded` | 13.8 s | still running at 12.6 min (**> 55×**), 100 % GPU, stopped |
  | `test_snappy_batch_c_api` | 1.66 s | still running at > 30 min (**> 1000×**), stopped |

  Cascaded goes through hipCUB block collectives (which still compute with rocPRIM's
  architecture-derived wave size); Snappy does not — it uses ARCTO's own wave primitives and a
  three-wave producer/consumer design with spin-waits. The common factor is the RDNA wave64
  execution model itself (a 64-lane wave issued over the 32-wide SIMD), which changes both the
  spin-wait dynamics and the resident-group count. **The exact mechanism was not isolated** — the
  runs were stopped rather than profiled — so this is recorded as a measurement, not an explanation.
* Conclusion: the configuration is *available* (it compiles, and the wave32-independent parts of the
  test suite that did finish passed), but wave32 remains the default and the recommendation for
  RDNA. On CDNA nothing changes: gfx9 is wave64-only and is unaffected by any of this.

## 4. HIP-first with a CUDA backend: what it took to make it true

The CUDA backend had not been built against a current toolkit. Building it on the GH200 (CUDA 13.2,
aarch64) exposed one ARCTO bug and three defects **inherited verbatim from nvCOMP 2.2** — the same
three broke nvCOMP 2.2 itself when we built it there for the comparison:

| Defect | Fix |
|---|---|
| `set(GPU_ARCHS "60")` hard-coded Pascal; CUDA 13 removed `sm_60`/`sm_70` | honour `CMAKE_CUDA_ARCHITECTURES` (as the AMD side honours `CMAKE_HIP_ARCHITECTURES`); bound the default list by toolkit version |
| CUDA language compiled as C++14; CCCL (Thrust/CUB/libcu++) requires C++17 | `CMAKE_CUDA_STANDARD 17` next to the existing CXX/HIP standards |
| `cub::Sum` / `cub::Min` / `cub::Max` removed in CCCL 3 | ARCTO passes its own functors (`arcto::ops::*`), valid with any generation of CUB or hipCUB |
| nvcc device flags attached to `COMPILE_LANGUAGE:HIP`, so they never applied on the CUDA backend | attach to `COMPILE_LANGUAGE:CUDA`, and add `--expt-relaxed-constexpr` (hip-clang accepts `std::numeric_limits<T>::max()` in device code; nvcc needs the flag) |

Two smaller portability items: `benchmark_common.h` used `std::istringstream` without
`#include <sstream>` (libstdc++ pulled it in for hip-clang but not for nvcc), and
`HipUtils_test` required `device_pointer()` to throw for an unregistered host pointer — on
Grace-Hopper (and on coherent-memory APUs generally) the runtime legitimately reports a device
address for ordinary host memory, so the test now accepts either.

Also fixed while in the area: the HLIF scratch allocator kept the hipified
`#if CUDART_VERSION >= 11020` gate around `hipMallocAsync`, a macro that is **never defined on
AMD** — so every HLIF scratch allocation had silently been taking the synchronous
`hipMalloc`/`hipFree` path (and `hipFree` synchronises the device). It now uses stream-ordered
allocation on HIP ≥ 6 and CUDA ≥ 11.2.

## 5. Verification matrix

| Configuration | Build | Tests |
|---|---|---|
| gfx1100 (RX 7900 XT), wave32 — default for RDNA | ✅ | **17/17** |
| gfx1100, wave64 (`-mwavefrontsize64`) | ✅ | builds; the tests that complete pass, but two of them are > 55× / > 1000× slower than in wave32 and were stopped (§3) |
| gfx942 (MI300A), wave64 — default for CDNA | ✅ | **17/17** |
| GH200 (`sm_90`), CUDA 13.2, `-D CUDA_BACKEND=ON` | ✅ | **17/17** |

The pre-existing `BitPackGPU_test` failure disappeared with the legacy kernels it tested; the suite
went from 22 tests with one failing to 17 tests all passing.
