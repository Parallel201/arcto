# ARCTO — AMD Optimization Map

Static review of the whole tree (no GPU on the review machine; nothing was built or run).
Baseline read: `main` @ `e3e1a1f`. State of the art for LZ4 compression: `origin/opt/curated`
(13 commits ahead of the merge base, 2026-08-20). Everything marked *expected* below is a
hypothesis to be measured on gfx90a / gfx942 / gfx1100 with the protocol in §9 — nothing here is a
measured result unless it quotes one of the `opt/*` commit messages.

**Measured results (August 2026).** The status chips below were updated from the optimization
campaign of 2026-08-22; the numbers, per category, are aggregated in
`docs/OPTIMIZATION_RESULTS_2026-08.md`, and every commit has an entry in `docs/experiments/*.md`
on its lineage (`opt/snappy-2026-08`, `opt/cascaded-2026-08`, `opt/lz4-decomp-wave64-2026-08`,
`opt/zfp-host-2026-08`) — all of which are merged into the single branch `opt/aggregate`.

<div class="callout rule"><p><strong>Ground rule.</strong> No proposal in this document changes a
codec's algorithm or on-the-wire format: LZ4 stays LZ77 with a hash table and a 64 KiB sliding
window; Snappy stays Snappy; Cascaded stays RLE → Delta → BitPack with the same headers; ZFP stays
bit-compatible with canonical zfp. Every item only changes <em>how</em> the kernel executes — launch
geometry, wave64/wave32 paths, LDS layout, registers/occupancy, load/store width and cache policy,
wave-level primitives, synchronization, compile flags, and host-side staging around the kernels.</p></div>

## 0. How to read this document

**ID scheme.** `BLD-n` build/flags · `HLIF-n` / `HOST-n` / `BENCH-n` shared infrastructure ·
`LZ4-Cn` / `LZ4-Dn` compression / decompression · `SNP-*` Snappy · `CAS-*` Cascaded (`S` shared,
`C`, `D`) · `ZFP-*` (`T` gate, `C`, `D`, `H` host side). File:line references are against `main`
@ `e3e1a1f` unless the text says `opt/curated`.

**Columns.** *Impact* is a qualitative expectation (H/M/L) for the affected kernel on the default
configuration; *Effort* S (< 1 day), M (1–3 days), L (> 3 days); *CUDA risk* is the risk of
breaking or slowing the experimental CUDA backend (the portable build must not regress by
construction — AMD-only behaviour is gated with `#if defined(__HIP_PLATFORM_AMD__)`);
*Preserves* states why the compressed bytes / algorithm are unaffected.

**Status chips.** <span class="chip kept">kept</span> measured and kept on `opt/curated` ·
<span class="chip revert">reverted</span> tried on `opt/lz4-compress-2026-08` and reverted ·
<span class="chip todo">open</span> not yet tried · <span class="chip caution">wave32 only</span> /
<span class="chip caution">wave64 only</span> landed for one wave size only.

**Acceptance gates** (the ones already used in the experiment log, keep them): `ctest` green on
wave64 and wave32 builds; compressed bytes identical (or the ratio deviation reported in exact
bytes when a change legitimately alters match choice); round trip bit-exact; throughput from the
chunked benchmarks at saturation (`-x` duplication so MI300X holds ≥ ~10 k chunks) and in the
undersubscribed regime; for ZFP additionally the byte-exact payload gate `ZFP-T1`.

**Branches.** One `opt/<topic>-<yyyy-mm>` branch per aggregation off `main`, one commit per
change (hypothesis → prediction → measured verdict in the body), explicit "Revert attempt X"
commits, combinations on their own branch. Suggested branch plan in §9.

---

## 1. Repository map

### 1.1 Layers and call chains

| Layer | Entry | Host wrapper | Device code |
|---|---|---|---|
| Low-level batched C API, LZ4 | `include/arcto/lz4.h` → `src/lowlevel/LZ4Batch.cpp:71-224` | `lz4BatchCompress/Decompress` `src/lowlevel/LZ4CompressionKernels.hip:142-318` | `lz4CompressBatchKernel<T>` / `lz4DecompressBatchKernel` (`.hip:73-136`) → `compressStream<T>` / `decompressStream` in `src/LZ4Kernels.hiph` |
| Low-level, Snappy | `include/arcto/snappy.h` → `src/lowlevel/SnappyBatch.cpp` | `SnappyBatchKernels.hip` | `src/snappy/*.hiph` (compression, decode strategies, warp scans, prefetch) |
| Low-level, Cascaded | `include/arcto/cascaded.h` → `src/lowlevel/CascadedBatch.hip:329-462` | type dispatch `ARCTO_TYPE_ONE_SWITCH` (`src/type_macros.h:219`) | `do_cascaded_compression_kernel` / `cascaded_decompression_fcn` in `src/CascadedKernels.hiph:761-1435` |
| ZFP | `include/arcto/zfp.h` → `src/lowlevel/ZFPBatch.cpp` | zfp C API (`zfp_compress/decompress`) | vendored `third_party/zfp/src/hip/*` (`encode3_kernel`, `decode3_kernel`, `compact_stream_kernel`) |
| High-level manager (HLIF) | `include/arcto/<codec>.hpp`, `arctoManager.hpp`, `arctoManagerFactory.hpp` | `src/highlevel/ManagerBase.hpp` → `BatchManager.hpp` → `<Codec>BatchManager` (`<Codec>Manager.hpp`) | `HlifCompressBatchKernel` / `HlifDecompressBatchKernel` (`src/arcto_common_deps/hlif_shared.hiph:229-447`) wrapping the same per-chunk device functions via `<Codec>HlifKernels.hip` |
| Host staging | `include/arcto/host_batch.h`, `host_batch_adaptive.h` | `src/lowlevel/HostBatch.cpp`, `HostBatchAdaptive.cpp` | — |

Legacy per-stage Cascaded kernels (`src/BitPackGPU.hip`, `DeltaGPU.hip`, `RunLengthEncodeGPU.hip`)
are **only** referenced by `src/test/*_test.cpp`; the batched and HLIF paths use the fused kernels
in `CascadedKernels.hiph`. They are still compiled into `libarcto` (glob in
`src/CMakeLists.txt:76-79`) and are the only place `src/amd_optimizations.h` is used.

### 1.2 Kernel inventory (one line per kernel family)

| Kernel | Where | Block / grid | LDS | `__launch_bounds__` | Notes |
|---|---|---|---|---|---|
| `lz4CompressBatchKernel<T>` | `LZ4CompressionKernels.hip:73` | `warpsize` × 1 wave per chunk, grid = batch | 256 B (main) / +128–256 B hash table (curated) | none (main) / `(warpsize, 8)` on wave64 (curated) | one wave = one chunk; per-round dependent chain |
| `lz4DecompressBatchKernel` | `.hip:96` | `(warpsize, 2)`, grid = ⌈batch/2⌉ | 2 × 512 B window | none | the two waves never synchronize |
| `HlifCompressBatchKernel<lz4…>` / `HlifDecompressBatchKernel<lz4…,2>` | `hlif_shared.hiph:229/419` | `warpsize` / `(warpsize,2)`, grid = CUs × occupancy | as above | none | persistent, atomic work stealing |
| Snappy `snap_kernel` / `unsnap_kernel` / `get_uncompressed_sizes_kernel` (+ HLIF) | `SnappyBatchKernels.hip:65-82, 145-160, 84-134` | `2·warpsize` (2 waves: emit + match) / `3·warpsize` (decode, prefetch, process waves) / `warpsize`, grid = batch | ≈ 8.2 KB (`uint16 hash_map[4096]`) / ≈ 6.3 KB (symbol queue + 4 KB ring) | single-arg `__launch_bounds__` (`:65,84,145`); none on HLIF | zero temp space; three specialised waves spin-wait on each other through `volatile` LDS |
| `cascaded_compression_kernel<T,size_t,128>` | `CascadedBatch.hip:97` | 128, grid = batch | 10.4–24.7 KB static | none | one block per partition, serial over 4 KB chunks |
| `cascaded_decompression_kernel_type_check<1/2/4/8>` | `CascadedBatch.hip:151` | 128, grid = batch, **4 launches per call** | 10.4–24.7 KB | none | 3 of 4 launches early-exit |
| `get_decompress_size_kernel` | `CascadedBatch.hip:262` | 128, grid = ⌈batch/128⌉ | — | — | trivial |
| zfp `encode{1,2,3}_kernel<S>` / `decode{1,2,3}_kernel<S>` | `third_party/zfp/src/hip/encode3.h:45`, `decode3.h:39` | 128, grid = ⌈blocks/128⌉ | 0 / 256 B | `(256,1)` on 3-D only | **one thread = one zfp block**; `fblock/iblock/ublock[64]` private arrays |
| zfp `compact_stream_kernel` (+ `copy_length_kernel`, `align_stream_kernel`) | `hip/variable.h:157` | 512 (tile × tiles), cooperative launch | ≤ 48 KB dynamic | `(512)` | variable-rate only |
| legacy `bitPack*`, `deltaKernel`, `rle*Kernel` | `src/BitPackGPU.hip`, `DeltaGPU.hip`, `RunLengthEncodeGPU.hip` | 256 / 1024 / 128 | small | `AMD_LAUNCH_BOUNDS_*` (main) / `(BLOCK_SIZE)` (curated) | tests only |

### 1.3 Wave-size model

`USE_WARPSIZE_32` vs `ENABLE_HIP_OPT_WARPSIZE64` is a **CMake-time global** (`CMakeLists.txt:128-132`),
consumed by `src/arcto_device_types.h:28-44` (`warp_mask_t`, `arcto::warpsize`),
`src/amd_optimizations.h:46-65`, `src/device_functions.hiph:199-294` (ballot / warp reductions),
`src/LZ4Kernels.hiph:87-103` (`lane_mask_t`, `LZ4_*_THREADS_PER_CHUNK = warpsize`) and
`src/snappy/config.h:45-53` (`LOG2_BATCH_SIZE`, block sizes). Consequences: one binary cannot serve
gfx90a/gfx942 and gfx1100 correctly; nothing in `main` checks `hipDeviceProp_t::warpSize` at runtime
(`opt/curated` adds `HipUtils::check_wave_size()` at the batched entry points). The vendored zfp
kernels do not depend on wave size at all. The CUDA backend compiles everything with
`warpsize == 32` and native CUDA intrinsics (the shim `include/cuda_shim/hip/hip_runtime.h` only
aliases runtime API names).

---

## 2. Build system and compile flags

### 2.1 What reaches HIP device code today (verified in the CMake files)

Project-added: `-fPIC`, `-D__HIP_PLATFORM_HCC__ -D__HIP_PLATFORM_AMD__`, one of
`-DENABLE_HIP_OPT_WARPSIZE64` / `-DUSE_WARPSIZE_32`; Debug adds `-g -ggdb`; the zfp target gets
`-Wno-unused-result`. **Release `-O3 -DNDEBUG` comes from CMake's own HIP defaults** — the explicit
`-O3` at `CMakeLists.txt:152` is for CXX only; nothing in the tree says `-O3` for HIP. Not present
anywhere: `-munsafe-fp-atomics`, `-ffast-math`, `-fgpu-rdc`, `-flto`, `-mcumode`,
`-mwavefrontsize64`, `-Rpass-analysis`, `--save-temps`, `-mllvm …`, `hipcc`. CMake drives
`clang++` directly; hipcc's habitual `-mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false`
are therefore **not** applied. `DEVEL=1` warnings never reach HIP TUs (`CMakeLists.txt:196`).
`hip::hipcub` is found but never linked (headers resolve via `/opt/rocm/include` by accident).
On the CUDA backend `src/CMakeLists.txt:87` relabels every `.hip` as `LANGUAGE CUDA`; the chunked
benchmarks then compile against **nvCOMP** (`benchmark_template_chunked.cuh:34-52`), which is the
intended cross-vendor baseline.

### 2.2 Opportunities

| ID | What | Why / mechanism | Risk | Effort |
|---|---|---|---|---|
| BLD-1 <span class="chip kept">done: explicit, exactly neutral</span> | Make HIP optimisation explicit: `set(CMAKE_HIP_FLAGS_RELEASE "${CMAKE_HIP_FLAGS_RELEASE} -O3")` next to `CMakeLists.txt:152`, and `message(STATUS)` the HIP release flags | Today correctness of every benchmark depends on a CMake default; a toolchain file or a future CMake can change it silently. Also confirm `-DNDEBUG` reaches device code — device `assert()` (`LZ4Kernels.hiph:599,802,914`; `CascadedKernels.hiph:783,826,844`) would otherwise distort every measurement | none | S |
| BLD-2 <span class="chip kept">done: ARCTO_DEVICE_REPORTS</span> | Opt-in `ARCTO_DEVICE_REPORTS=ON` → `$<$<COMPILE_LANGUAGE:HIP>:-Rpass-analysis=kernel-resource-usage>` (+ `--save-temps`) on `arcto` | Prints VGPR/SGPR/AGPR/LDS/occupancy/spill per kernel per `--offload-arch` at compile time — the cheapest way to validate every launch-bound and register-pressure item in this document; `--save-temps` gives the `.s` ISA to confirm `global_load_dwordx4`, `ds_bpermute`, `v_readlane` emission | none | S |
| BLD-3 <span class="chip todo">open</span> | Try hipcc's inlining flags on `arcto`: `-mllvm -amdgpu-early-inline-all=true -mllvm -amdgpu-function-calls=false` | The HLIF wrappers call the per-chunk device functions through template objects; the Snappy decoder is a state machine of many small device functions. If clang left any real call, it costs a stack frame and kills register allocation across the call. Check `-Rpass-analysis` first for non-inlined calls | compile time ↑ | S |
| BLD-4 <span class="chip todo">open</span> | gfx1100 only: A/B `-mcumode` (CU mode vs default WGP mode) | In WGP mode two CUs share the 128 KB LDS and the workgroup can be split across them; CU mode sometimes helps kernels with small-LDS single-wave workgroups (LZ4) and hurts large-LDS ones (Cascaded). Apply per-arch build dir | none | S |
| BLD-5 <span class="chip todo">open</span> | `HIP_ENABLE_WARP_SYNC_BUILTINS` (ROCm ≥ 6.2) to re-point `SYNCWARP/__shfl_sync/__match_any_sync` branches on AMD | Lets LZ4's `__CUDACC_VER_MAJOR__` / `__CUDA_ARCH__ >= 700` branches (`LZ4Kernels.hiph:71-85,221`) use HIP's software `__match_any_sync` and real `__syncwarp` instead of no-ops; compare codegen with the hand-written claim table | must verify availability | S |
| BLD-6 <span class="chip caution">design</span> | Per-target wave size instead of the CMake global: derive lanes from `__AMDGCN_WAVEFRONT_SIZE__` / `__builtin_amdgcn_wavefrontsize()` in `arcto_device_types.h`, `amd_optimizations.h`, `LZ4Kernels.hiph`, `snappy/config.h`; host launch dims from `hipDeviceProp_t::warpSize` | Enables a correct `gfx90a;gfx942;gfx1100` fat binary. Interim: the runtime guard (`opt/curated`) plus a configure-time warning when `CMAKE_HIP_ARCHITECTURES` mixes gfx9 and gfx10+/11 | medium | L |
| BLD-7 <span class="chip kept">done: include path (link injects offload flags)</span> | Link `hip::hipcub` (or `roc::rocprim`) PRIVATE to `arcto`; add `-Wall -Wextra -Wno-unused-parameter` for `COMPILE_LANGUAGE:HIP` under `DEVEL` | Hygiene: makes the include path explicit and lets clang flag the unknown `likely_uniform` attribute (`amd_optimizations.h:350`) and similar | none | S |
| BLD-8 | Flags **not** worth adding (checked) | `-munsafe-fp-atomics`: no FP atomics anywhere (only integer `atomicAdd/atomicXor` in `hlif_shared.hiph` and zfp). `-ffast-math`/`-Ofast`: integer codecs gain nothing and zfp's `frexp/ldexp` float path could change the bitstream — never on `zfp`. `-fgpu-rdc`: every kernel is fully instantiated in its TU. LTO: device gain ≈ 0 | — | — |

---

## 3. Shared infrastructure

### 3.1 HLIF pipeline (how a manager compresses)

`ManagerBase::compress` (`src/highlevel/ManagerBase.hpp:203-228`) → `BatchManager::do_compress`
(`BatchManager.hpp:226-257`): `hipMemcpyAsync` of the codec's `FormatSpecHeader` (from pinned host),
`hipMemsetAsync(comp_data_size)`, `hipMemsetAsync(ix_chunk)`, then **one persistent kernel**
`HlifCompressBatchKernel` with `grid = multiProcessorCount × hipOccupancyMaxActiveBlocksPerMultiprocessor`
and the codec's block size. Inside (`hlif_shared.hiph:160-227`): block 0 writes the `CommonHeader`
(`hlif_shared_types.hpp:68-83`: magic, version 2.2, format id, atomic `comp_data_size` cursor,
sizes, `uncomp_chunk_size`, `comp_data_offset`); each CTA compresses chunk `ix` into its private
scratch slot, thread 0 reserves output space with a 64-bit `atomicAdd` on `comp_data_size`, the
group copies scratch → final (`copyScratchBuffer`, `:127-158`: `char4` loads + **four byte
stores** per lane, or a byte loop when scratch is not 4-B aligned), then takes the next chunk via
`atomicAdd(ix_chunk, 1)`. Layout after the headers: `size_t comp_chunk_offsets[n]`,
`size_t comp_chunk_sizes[n]`, two `uint32` checksum arrays (never filled), gap-less payload.
Scratch = `max_ctas × max_comp_chunk_size` (+ LZ4 hash tables); with 64-thread blocks the AMD
occupancy answer can reach 32 blocks/CU, i.e. ~9.7 k CTAs and ~0.9 GB of scratch on a 304-CU
MI300X for 64 KB chunks (`LZ4Manager.hpp:147-151`; `tests/test_lz4.cpp:253-254` warns about it).
`create_manager` (`arctoManagerFactory.cpp:65-148`) copies the `CommonHeader` to host,
**synchronises**, switches on `format`, copies the format header (second sync) and constructs the
codec manager. Scratch allocation uses `hipMallocAsync` only behind `CUDART_VERSION >= 11020`
(`ManagerBase.hpp:190-243`), which is never defined on HIP → blocking `hipMalloc`/`hipFree` on AMD.

### 3.2 Host staging

`arctoHostBatch` (`HostBatch.cpp`): prefix sum of chunk sizes, one `hipHostMalloc`, one
`hipMemcpyAsync` (measured ~4.3× over scattered pageable copies per `host_batch.h:1-27`).
`arctoHostBatchAdaptive` (`HostBatchAdaptive.cpp`): `t_total(W) = t_alloc + ⌈input/W⌉·(t_h2h + t_h2d + t_kernel)`,
`W_opt = max(W_kernel_sat, W_pcie_amort, W_launch_amort)` with per-arch `W_kernel_sat` = wave slots ×
chunk size (gfx906 15 MiB, gfx90a 52 MiB, gfx942 76 MiB @ 8 KB chunks, gfx1100 48 MiB,
`:43-75`), detected by substring on `gcnArchName` (`:63-73`) — the only arch detection in the tree.
The online EMA of measured rates is recorded but **never fed back** into `W_opt` (`:143-152, 201-229`).

### 3.3 Opportunities

| ID | Where | What | Mechanism / expected effect | Risk | Effort | Measure |
|---|---|---|---|---|---|---|
| HLIF-1 <span class="chip todo">open</span> | `hlif_shared.hiph:127-158, 197-203` | Reserve `roundUpTo(comp_chunk_size, 16)` in the output atomic so every chunk start is 16-B aligned, then copy scratch → final with `uint4` loads/stores + byte tail | Today four `global_store_byte` per lane (64 B per wave-instruction); compressed bytes are written twice, so for ratio ≈ 1 data the copy is a second full pass. Readers use the per-chunk offset table, so unmodified readers still decode; `comp_data_size` grows ≤ 15 B/chunk | low–medium (gap-less packing assumption) | S | `benchmark_lz4_chunked` HLIF on `synth_random_1mb`-like data; rocprof `WRITE_SIZE` |
| HLIF-2 <span class="chip todo">open</span> | `ManagerBase.hpp:222-224`, `BatchManager.hpp:254,114` | Fold the format-header copy and the two memsets into one init kernel (or one 16-B struct + one memset); pass the tiny format header by value | Three extra stream packets per call (~5–20 µs each on ROCm); visible for buffers < ~8 MB or many calls | low | S | rocprof timeline; 1000 × 1 MB `compress()` wall time |
| HLIF-3 <span class="chip todo">open</span> | `hlif_shared.hiph:197-225, 334-338` | Grab chunks in batches of *k* (`atomicAdd(ix_chunk, k)`) or stride statically when `num_chunks` is small | Two global atomics + two group syncs per chunk; with 4–8 KB chunks (gfx942 sweet spot per the adaptive model) one address serialises ~300 CUs × occupancy | low | S | `-p 4096…65536` sweep on gfx942 |
| HLIF-4 <span class="chip todo">open</span> | `LZ4Manager.hpp:147-151`, `LZ4HlifKernels.hip:222-262` (Snappy/Cascaded equivalents) | Cap `max_ctas` at e.g. `CUs × 8` (or a byte budget) | Avoids ~1 GB scratch and thousands of idle persistent CTAs for small inputs; 8 single-wave CTAs/CU already equals max VGPR-bound occupancy × 2 | none | S | scratch size log; tail latency on small inputs |
| HLIF-5 <span class="chip todo">open</span> | `ManagerBase.hpp:190-243` | `#if CUDART_VERSION>=11020 || HIP_VERSION>=50300000` → `hipMallocAsync/hipFreeAsync` | Removes a device-wide sync on first use / `set_scratch_buffer` | low | S | API trace |
| HLIF-6 <span class="chip todo">open</span> | `ManagerBase.hpp:164-168`, `BatchManager.hpp:133-137` | Copy the whole `CommonHeader` into the already-pinned `common_header_cpu` with one async copy + event; fix the 8-B copy into a `uint32_t num_chunks` | D2H into pageable memory is effectively synchronous; the 8-into-4 copy is UB (benign only by tail padding) | none | S | — |
| HLIF-7 <span class="chip todo">open</span> | `HipUtils.hip:91-170`, `LZ4Batch.cpp:107-117` | Cache `DeviceInfo` (CU count, warpSize, gcnArchName, LDS/block, L2) once per process; make the 6–7 `hipPointerGetAttributes` per low-level call debug-only or memoised | `hipGetDeviceProperties` is heavy on ROCm; `hipPointerGetAttributes` costs µs each → tens of µs host overhead per API call, comparable to a launch | loses an early host-pointer error | S | host timer around `arctoBatched*Async` on an empty stream |
| HOST-1 <span class="chip todo">open</span> | `HostBatchAdaptive.cpp:143-152` | Re-choose `W_opt` from the measured rates after a first probe window (or document the model as priors-only) | The API claims online adaptation that never affects the window | none | S | `-A -R` report |
| HOST-2 <span class="chip todo">open</span> | `HostBatch.cpp:45`, `HostBatchAdaptive.cpp:149` | A/B `hipHostMalloc` flags (write-combined / non-coherent / `hipExtHostAlloc` coarse-grained) for pure H2D staging | Fine-grained coherent pinned memory is the default and slower for streaming uploads; the benchmark documents a gfx1100 anomaly after `hipHostMalloc` (`benchmark_template_chunked.cuh:451-456`) | none | S | `-P -R` H2D GB/s |
| HOST-3 <span class="chip todo">open</span> | `host_batch_adaptive.h:51` | Double-buffer the adaptive window (two pinned windows, H2H of N+1 overlapping H2D/kernel of N) | The cost model is explicitly single-buffered; overlap hides `t_h2h` | API gains a second buffer | M | `-A -R` total time |
| BENCH-1 <span class="chip todo">open</span> | `benchmark_template_chunked.cuh:695-698,776-778` | One decompression output allocation sliced by prefix sum instead of one `hipMalloc` per chunk per iteration | 3200 allocator calls per iteration for 1600 chunks; outside the timed region but fragments the heap and makes `-i 100` sweeps slow | none | S | wall time |
| TMP-1 <span class="chip todo">open</span> | `TempSpaceBroker.h:78-88` | Align sub-buffers to ≥ 16 B (ideally 256 B) and add the slack to the `GetTempSize` formulas | Enables `dwordx4` access to temp regions; today `alignof(T)` ≤ 8 | temp-size formulas must include padding | S | — |

---

## 4. LZ4

### 4.1 Code map

Compression: `arctoBatchedLZ4CompressAsync` → `lz4BatchCompress` (type dispatch on
`arctoType_t`) → `lz4CompressBatchKernel<T><<<batch, warpsize>>>` → `compressStream<T>`
(`src/LZ4Kernels.hiph:793-969` on `main`). One wave owns one chunk; per round each lane loads one
`T` (`:849-851`), the 4-byte key is assembled with 2 (uint8) / 1 (uint16) `__shfl_down`
(`shuffleLiterals`, `:743-791`), lanes ≥ `numValidThreads` (`invalid_threads = 3/sizeof(T)`) are
don't-care, so a wave64 advances 61/63/64 positions per round; local (intra-window) match search,
ballot, hash-table gather (`:643`), `isValidHash` re-reads the candidate (`:656`), insert,
`lengthOfMatch` extends with 2 coalesced loads per lane per 64 positions and a ballot per step
(`:592-617`), `writeSequenceData` emits token/literals/offset/LSIC (`:665-715`). Hash table:
`min(roundUpPow2(chunk), MAX_HASH_TABLE_SIZE)` entries of `uint16_t` (`LZ4Types.h`,
`LZ4CompressionKernels.hip:142-156`), entries store `pos & MAX_OFFSET`, `NULL_OFFSET = 0xFFFF`,
hash `(brev(key) + (key ^ 0xc375)) & (size-1)` (`:557-561`). Output is plain LZ4 regardless of `T`
(positions/lengths rescaled to bytes, `:946-954`); the decompressor is type-agnostic.

Decompression: `lz4DecompressBatchKernel<<<⌈batch/2⌉, (warpsize,2)>>>` → `decompressStream`
(`:971-1097`) with a per-wave LDS window of `warpsize × 8` B refilled when half-consumed
(`BufferControl`, `:353-518`); token parsing is done redundantly by all lanes; literal and match
copies are byte-granular, overlapped matches use `source[i % dist]` (`coopCopyRepeat`, `:530-542`).
HLIF: `lz4_compress_wrapper<T>` (hash table = `tmp_buffer + blockIdx.x × hash_table_size`) and
`lz4_decompress_wrapper` (LDS slice per `threadIdx.y`) inside the persistent kernels.

### 4.2 What `opt/curated` already does (do not redo)

| Change | Status | Measured (from commit bodies) |
|---|---|---|
| `MAX_HASH_TABLE_SIZE` default 128 entries on wave32, 64 on wave64 (CUDA keeps 1<<14); `-DARCTO_LZ4_MAX_HASH_TABLE_SIZE` override | <span class="chip kept">kept</span> O1/E14 | 29.2 / 64.4 / 8.2 GB/s dense 64 K on gfx1100 / gfx90a / gfx906; ratio deviation ≤ +0.0485 % in exact bytes |
| Hash table resident in LDS (`ARCTO_LZ4_LDS_TABLE=1` on AMD) | <span class="chip kept">kept</span> A8/E11 | +25 % gfx90a (80.5 GB/s), +9 % gfx1100 (31.9), +300 % gfx906 (32.6) over the packed global table, byte-identical |
| wave64 local-match search = 256-bucket LDS claim table `findEarliestTwin` (atomicMin + one shuffle); wave32 keeps the 32-shuffle rotation; CUDA `__match_any_sync` | <span class="chip kept">kept</span> + E17 gating | MI300X 4.85 → 11.7 GB/s @64 K (2.4×); gating restored −8 % on gfx1100 |
| `insertHashTableWarp`: race-tolerant plain store (reader re-validates) | <span class="chip kept">kept</span> | gfx1100 compress +16 % (with the rotation) |
| `__launch_bounds__(LZ4_COMP_THREADS_PER_CHUNK, 8)` on wave64 (64-VGPR pin, 20 spills accepted); wave32 only the block cap | <span class="chip kept">kept</span> | MI300X +5.8 % @64 K vs −10 % without the hint |
| `ARCTO_KRESTRICT` (`__restrict__`) on AMD wave64 only | <span class="chip kept">kept</span> E17 | +18.5 % MI210 / +12.7 % MI50 in interaction with the pin; −1.8 % gfx1100 |
| `WARP_READ_LSIC` (warp-cooperative length decode) | <span class="chip caution">wave32 only</span> | gfx1100 zeros decomp +21 %; MI300X −21 % |
| `coopCopyVec` dword-vectorised literal / no-overlap copies; `coopCopyRepeat` prefix doubling for long overlapped matches | <span class="chip caution">wave32 only</span> | gfx1100 146 → 151–159 GB/s decomp @8 K, zeros +20 %; "every variant tried cost 5–20 % decomp on MI300X" |
| `HipUtils::check_wave_size()` runtime guard at all batched entry points | <span class="chip kept">kept</span> | — |
| Legacy BitPack/Delta/RLE `__launch_bounds__` tied to real `BLOCK_SIZE` | <span class="chip kept">kept</span> | neutral on gfx1100 |
| Drop / avoid the per-chunk hash-table clear loop (A1/A2) | <span class="chip revert">reverted</span> E10 | neutral — the clear is not the cost |
| Runtime table cap from device properties (O3) | <span class="chip revert">reverted</span> E15 | locates the knee, not the optimum; constants derived instead |
| wave32 min-waves knob (O4) | <span class="chip revert">reverted</span> E16 | 0.94× / 0.79× — spill signature |

Side effect worth recording: `main` has a latent wave64 defect — `numValidThreadsToMask` returns
`int` (`LZ4Kernels.hiph:717-720`) and `insertHashTableWarp` truncates the 64-bit match mask to
`int` (`:732`), so on wave64 the "last matching lane inserts" rule could pick no lane and
`warpMatchAny` compared against never-written LDS slots (61..63 for uint8). On `opt/curated` the
wave64 path no longer calls `warpMatchAny` (claim table + race-tolerant insert), so the defect is
moot there; it only matters if the rotation is ever re-enabled on wave64.

### 4.3 NVIDIA-isms still present on `opt/curated`

- `SYNCWARP1()` / `SYNCWARP(m)` / `syncCTA()` expand to **nothing** on AMD (`LZ4Kernels.hiph:71-85,
  204-211`): the claim table (`claim[k] = …; SYNCWARP1(); atomicMin; SYNCWARP1(); claim[h]`), the
  decompression window refill and the LDS-sourced overlapped copy rely on wave lock-step and on the
  compiler not reordering may-alias LDS accesses. Correct in practice; `__builtin_amdgcn_wave_barrier()`
  (already provided as `arcto::amd::wave_barrier()`) is a zero-cost scheduling barrier that makes it
  robust → LZ4-C12.
- `shuffleLiterals` assembles the key with `ds_bpermute` shuffles when an unaligned dword load would do → LZ4-C2.
- `readWord/writeWord` are byte-assembled (`:247-265`), used on the hottest dependent load (`isValidHash`) and for the decode offset → LZ4-C3 / LZ4-D4.
- `lengthOfMatch` makes one dependent L2/MALL round trip per 64 positions → LZ4-C5.
- Wave-uniform broadcasts via `__shfl` (`:891,921`) instead of `v_readlane`; `CLZ(BREV(m))` instead of `ffs` → LZ4-C9.
- Decompression on wave64: byte-granular copies, per-byte `urem`, 512-B window, no launch bounds → LZ4-D*.
- HLIF grid amplification of scratch (§3.1) → HLIF-4.
- Measurement pitfall: one wave per chunk means a 100 MB / 64 KB batch is 1600 waves ≈ 1.3 waves
  per SIMD on MI300X — no latency hiding; use `-x` duplication so the batch reaches ~10 k chunks
  before comparing kernel micro-optimisations.

### 4.4 Remaining opportunities — compression

| ID | Where (`main` lines) | What | Mechanism on CDNA2 / CDNA3 / RDNA3 | Impact | CUDA risk | Preserves | Effort | Measure |
|---|---|---|---|---|---|---|---|---|
| LZ4-C2 <span class="chip todo">open</span> | `LZ4Kernels.hiph:848-854, 754-781` | For `T = uint8/uint16`, load `next` with one unaligned dword (`__builtin_memcpy(&next, p + idx + lane, 4)`) when `idx + lane + 4 ≤ length`; keep the shuffle path for the last lanes only | One `global_load_dword` (unaligned OK on gfx9+) replaces a byte load + 2 `ds_bpermute_b32` + `s_waitcnt lgkmcnt` per round and removes a cross-lane dependency from the per-round chain; 64 lanes read 67 contiguous bytes (two lines) | M | low (memcpy → 4 byte loads on NVIDIA) | identical `next` for all valid lanes → identical hashes/matches | S | compress GB/s; `--save-temps`: no `ds_bpermute` in the loop |
| LZ4-C3 <span class="chip todo">open</span> | `:247-265` (`readWord/writeWord`), used at `:656, 700/704, 1040-1042` | Implement with `__builtin_memcpy` into `T` | `isValidHash` issues 4 byte loads (uint8) at a random address on the dependent chain; one unaligned dword halves VMEM instruction count and frees VGPRs | L–M | none | same values | S | VGPRs via BLD-2; GB/s |
| LZ4-C5 <span class="chip todo">open</span> | `:592-617` (`lengthOfMatch`) | Compare 4 B (or 16 B when aligned) per lane per step via `memcpy` loads, ballot "any byte differs", resolve the first differing byte in the winning lane with `ctz(a ^ b)/8`, clamp the last partial lane exactly like the scalar loop; optionally software-pipeline one step ahead | Each 64-position step is a dependent memory round trip; a 64 KB zero run costs ~1024 serial latencies; 4 B/lane cuts dependent trips 4× (16× with dwordx4), same `num_matches` | H on compressible data, neutral on random | low (byte loads on NVIDIA, fewer ballots) | identical match length | M | GB/s on `-x`-duplicated zeros/RLE vs random; `SQ_WAIT_INST_ANY`, `SQ_INSTS_VMEM_RD` |
| LZ4-C6 <span class="chip caution">wave32 done</span> | `:575-585` (`copyLiterals`, compression side) | Dword copy via `memcpy` on wave64 too (wave32 already uses `coopCopyVec`) | 4× fewer load/store instructions for long literal runs; separate from the decompression-side losses on MI300X — measure on its own and watch VGPRs against the 64-VGPR pin | L–M (incompressible data) | none | pure copy | S | GB/s on random data |
| LZ4-C7 <span class="chip todo">open</span> | stores at `:677-680, 686, 691-692, 700-705, 711` | `__builtin_nontemporal_store` for `compData` on AMD (plain store / `__stcs` on CUDA) | Output is write-once; streaming it past L2 leaves L2 for tables and input windows — matters most on CDNA2 (8 MB L2 per GCD, no Infinity Cache), less on CDNA3 (256 MB MALL) | L–M CDNA2, L elsewhere | none (guarded) | same bytes | S | `TCC_HIT/MISS`, `TCC_EA_WRREQ` |
| LZ4-C9 <span class="chip todo">open</span> | `:891, 921` and `:878, 884, 887, 912, 611, 734` | `__builtin_amdgcn_readlane` for wave-uniform broadcasts; `__ffsll(m)-1` instead of `CLZ(BREV(m))` | Avoids `ds_bpermute` + `lgkmcnt` waits and two 64-bit brev/clz sequences per use, inside the per-round chain | L | none (guarded) | same | S | ISA diff |
| LZ4-C11 = HLIF-4 | | cap `max_comp_ctas` | see §3.3 | mem H | none | yes | S | |
| LZ4-C12 <span class="chip todo">open</span> | `:82-83` (`SYNCWARP1/SYNCWARP`), `:204-211` (`syncCTA`) | On AMD expand to `__builtin_amdgcn_wave_barrier()` | Zero runtime cost; prevents future reordering bugs around the claim table, window refill and LDS-sourced copies | none (hardening) | none | yes | S | ctest |
| LZ4-C13 = BLD-2/4/5 | | resource reports, `-mcumode`, warp-sync builtins | | | | | | |

Not recommended (changes bytes, hence out of scope): altering the hash function, hash-size policy
beyond the measured defaults, or `ARCTO_TYPE_*` defaults.

### 4.5 Remaining opportunities — decompression (wave64 is the open front)

The E04 note "every variant tried cost 5–20 % decomp on MI300X" concerns copy vectorisation. A
plausible reason is register pressure: the decompress kernel has **no** launch bounds on `opt/curated`
(only the compress kernel got the 64-VGPR pin), so any widening that adds live registers can drop a
wave64 occupancy step. Do LZ4-D7 first and re-measure the wave32-only pieces on wave64 under it.

| ID | Where | What | Mechanism | Impact | CUDA risk | Preserves | Effort | Measure |
|---|---|---|---|---|---|---|---|---|
| LZ4-D7 <span class="chip kept">kept: exactly neutral; no occupancy target</span> | `LZ4CompressionKernels.hip:96`, `LZ4HlifKernels.hip:123` | `__launch_bounds__(2·warpsize, N)` after reading BLD-2 numbers; pin at the VGPR step that keeps spills ≈ 0 | Same mechanism as the compress pin (CDNA: ≤ 64 VGPRs → 8 waves/SIMD; 65–96 → 5); then re-run D1/D2 variants on wave64 | L–M, enabler | none | yes | S | `-Rpass-analysis`; GB/s |
| LZ4-D2 <span class="chip kept">kept: exact; zeros +37 % with D1w</span> | `:530-542` (`coopCopyRepeat`) | Keep `r = lane % dist`, `step = blockDim.x % dist` once; per iteration `r += step; if (r >= dist) r -= dist` (ALU-only, no widening); fast paths `dist == 1` (broadcast) and `dist ≥ 4` (4 consecutive bytes per lane) | 32-bit `urem` is a ~20–40-VALU software sequence on AMD and it runs per byte per lane for every overlapped match — the cheapest data for LZ4 (zeros, RLE) becomes the most VALU-expensive for the AMD decoder; wave32 has prefix doubling only for `length > 2·blockDim` | H on repetitive data | none | identical index sequence | S–M | zeros decomp GB/s (`synth_zeros_1mb`); `SQ_INSTS_VALU` |
| LZ4-D1 <span class="chip kept">wave64: decomp side on (binary ×2.3/×1.35, TTI ×2.2/×1.1, zeros ×1.2/×1.4, words ×0.8); knobs</span> | `:520-528` (literals / no-overlap) | Dword (16 B when both aligned) per lane via `memcpy`, byte head/tail — on wave64 under LZ4-D7 | Store-issue bound: 64 B per store instruction today; 4–16× fewer instructions and more bytes in flight per `s_waitcnt` | H for low-ratio data | none | pure copy | S–M | decomp GB/s vs ratio; `SQ_INSTS_VMEM_WR` |
| LZ4-D3 <span class="chip todo">open</span> | `:137-145, 458-500, 988-991`; LDS decls `LZ4CompressionKernels.hip:108`, `LZ4HlifKernels.hip:131,208` | Window 1–2 KB per wave, 16-B aligned, refilled with one `uint4` per lane, optional register prefetch of the next window; keep `batchedLZ4DecompMaxBlockOccupancy`'s `shmem_size` in sync | A refill is a dependent global round trip every 256 B of compressed input; 4× fewer with 2 KB windows and more literal copies served from LDS | M low-ratio data | none | cache of the same bytes | M | decomp GB/s on random data |
| LZ4-D4 = LZ4-C3 | `:1037-1043` | unaligned ushort offset read | one load instead of two on the per-sequence chain | L | none | yes | S | |
| LZ4-D5 <span class="chip revert">reverted: words −5…−8 % wave64; −2.5…−6 % on the vec build too</span> | `:994-1050` | Pass token/LSIC/offset through `__builtin_amdgcn_readfirstlane` so `comp_idx`, `decomp_idx`, `num_literals`, `offset`, `match` live in SGPRs | Lower VGPR pressure (→ occupancy), SALU/VALU overlap, scalar branches instead of `exec` manipulation for OOB checks and loop bounds | L–M | none (identity on NVIDIA) | yes | S–M | VGPR count; GB/s |
| LZ4-D6 <span class="chip todo">open</span> | `:128`, `LZ4CompressionKernels.hip:236-237` | `LZ4_DECOMP_CHUNKS_PER_BLOCK` 1 / 2 / 4 | Waves never synchronise, so this is dispatch granularity only; fewer workgroups to launch for large batches | L | none | yes | S | `-x 8` |
| LZ4-D8 = LZ4-C12 | `:204-211` | wave barrier in `syncCTA` | | | | | | |

Not recommended: non-temporal stores for the *decompressed* output — it is re-read by later match copies.

### 4.6 Verify on hardware

1. `-Rpass-analysis=kernel-resource-usage` for `lz4CompressBatchKernel<uint8_t/16/32>`,
   `lz4DecompressBatchKernel`, both HLIF instantiations on gfx90a / gfx942 / gfx1100: VGPR vs the 64 /
   96 / 128 steps (CDNA), SGPRs, scratch, LDS. Decides LZ4-D7 and whether D5 is worth it.
2. rocprof on the compressor: `SQ_WAIT_INST_ANY`, `SQ_INSTS_VALU/SALU/LDS/VMEM_RD`, `SQ_LDS_BANK_CONFLICT`,
   `TCC_HIT/MISS`, `TCC_EA_RDREQ`, `MeanOccupancyPerCU` — expect a dependent-chain (wait) profile;
   ranks LZ4-C2/C3/C5 vs C7.
3. rocprof on the decompressor: `SQ_INSTS_VMEM_WR` (low-ratio data) vs `SQ_INSTS_VALU` (zeros) — ranks LZ4-D1 vs D2.
4. Confirm with `--save-temps` that `__builtin_memcpy(&w, p, 4)` lowers to one `global_load_dword` /
   `ds_read_b32` per target (else C2/C3/C5/D1/D4 are void) and that nvcc accepts the same forms.
5. `__match_any_sync` availability under `HIP_ENABLE_WARP_SYNC_BUILTINS` on the deployed ROCm (BLD-5).
6. `-x 0,1,3,7` at `-p 65536` and `-p 32768` on MI300X to find saturation; report results at saturation.
7. gfx1100: `-mcumode` A/B (BLD-4).

---

## 5. Snappy

On `opt/curated` the only Snappy change is the `check_wave_size()` guard; the kernels themselves are
still the hipified nvCOMP 2.2 design — this is the first AMD pass for the codec.

### 5.1 Code map

**Compressor** `snap_kernel` (`SnappyBatchKernels.hip:65-82`) → `snappy::do_snap`
(`src/snappy/compression.hiph:281-385`): one 2-wave block per chunk, zero temp space
(`SnappyBatch.cpp:179`), state in static LDS (`compression_state.hiph:49-59`: pointers, three
`volatile` length/distance fields, `uint16_t hash_map[4096]` = 8 KB; `HASH_BITS = 12`,
`MAX_LITERAL_LENGTH = 256`, `MAX_COPY_LENGTH = 64`, `MAX_COPY_DISTANCE = 32768`, `config.h:83-93`).
Per iteration (`:338-378`): all lanes read the three fields, `__syncthreads`; **WARP0** emits the
previous literal run (`StoreLiterals`, `:73-117`: lane 0 writes the tag, lanes copy `dst[i] = src[i]`
byte-wise, stride 64) and the copy token (`StoreCopy`, lane 0, copy-1/copy-2 only); **WARP1** runs
`FindFourByteMatch` (`:190-246`): up to 4 rounds of `unaligned_load32(src+pos+t)` (two aligned
dword loads + funnel shift, `device_functions.hiph:330-341`), 12-bit hash, `HashMatchAny` (12
ballots, `:157-172`), LDS `hash_map` lookup, verifying load, ballot, `ffs`, hash-map update, then
`Match60` extends by ≤ 60 bytes with one ballot on wave64 (`:251-269`); `__syncthreads` (`:377`).

**Decompressor** `unsnap_kernel` (`:145-160`) → `do_unsnap` (`decompression.hiph:195-211`) →
`_do_unsnap<DecodeSymbols<TryDecodeStringOf2To3ByteSymbols, TryDecodeStringOf2To5ByteSymbols>, PrefetchByteStream, ProcessSymbols>`
(`:106-190`): three specialised waves per chunk communicating only through a `volatile unsnap_queue_s`
in LDS (`decompression_state.hiph:56-124`: `LZ77Symbol batch[4 × 64]` at struct offset 76 —
**4-B aligned only** — `batch_len[4]`, `batch_prefetch_rdpos[4]`, byte ring `buf[4096]`, cursors).
**WARP1 prefetcher** (`decompression_prefetch.hiph:71-129`): aligns `base+pos` to 64 B, then per 512-B
granule each lane loads 8 bytes and stores them **byte-wise** into the ring; polls `prefetch_rdpos`.
**WARP0 decoder** (`decompression_decode.hiph:207-289`): lane 0 waits for the prefetcher
(`WAIT_FOR_PREFETCHER`, `:163-172`); `TryDecodeStringOf2To3ByteSymbols` (`decompression_decode_strategies.hiph:118-185`):
three ballots over tag candidates, **lane 0** folds the 192 bits through `get_len3_mask_64` (16
dependent `k_len3lut` lookups, `warp_scans.hiph:221-300`) and broadcasts the 64-bit mask, each lane
computes its symbol start, decodes `(len, offset)`, wave prefix sum of lengths (`WarpReduce<64,64>`,
shuffle tree `device_functions.hiph:227-268`), range-check trims `batch_len`; optionally
`TryDecodeStringOf2To5ByteSymbols` (`:227-324`); then **lane 0 serially** decodes the rest
(`decode_and_fill_batch_using_single_thread`, `:72-150`: 1–5 dependent `volatile` LDS byte reads per
symbol); submit, wait for the next free slot (`WAIT_FOR_SYMBOL_PROCESSOR`, `NANOSLEEP(100)` hard-coded,
`:174-181`). **WARP2 processor** (`decompression_process.hiph:79-263`): polls `batch_len`, fast path
when the first two distances > 8 (prefix sums, `stop_mask`/`start_mask` via `WarpReduce::sum` of
`1<<bofs`, one output byte per lane), per-symbol loop with copies up to 2 × 64 B (`pos % dist` for
overlaps, `:188,194`) and literals `LITERAL_SECTORS` × 64 B from the ring or global; all output
stores are **byte-granular**. `NANOSLEEP` is `__nanosleep` only on `__CUDA_ARCH__ ≥ 700`; on AMD it is
`clock()` (`device_functions.hiph:176-181`) — i.e. **no sleep at all**. HLIF wraps both with
`chunks_per_block = 1` and no launch bounds; `get_uncompressed_sizes_kernel` (`:84-134`) uses one active
lane per 64-lane workgroup. The `snappy/` directory holds developer tools (Python reference parser,
len3/len5 mask models, host copy of `get_len3_mask_64` with its LUT) that validate the wave64
generalisation of nvCOMP's 32-lane warp scans — nothing is built from it.

### 5.2 NVIDIA-isms and AMD pitfalls

1. **Spin-waits that never yield** (`device_functions.hiph:176-181` + four polling loops): tight `volatile` LDS polls burn issue slots and LDS bandwidth of the SIMD shared with other chunks' productive waves; the AMD constants in `config.h:101-104` equal the NVIDIA ones and are ignored. AMD's primitive is `__builtin_amdgcn_s_sleep(n)` (~64·n clocks, yields the SIMD).
2. **`__shfl` broadcasts of wave-uniform values** (`SHFL10/SHFL1`, `device_functions.hiph:150-162`; ~20 call sites listed in the review) lower to `ds_bpermute_b32` (LDS crossbar + `lgkmcnt`) where `v_readfirstlane/v_readlane` (SGPR result) would do; divergent-index shuffles (`compression.hiph:213`, `decompression_process.hiph:134-135,142-143`) must stay.
3. **Shuffle-tree scans/reductions** (`WarpReducePos64/WarpReduceSum64`, `device_functions.hiph:228-240`): 6 dependent `bpermute` steps (12 for the 64-bit `start_mask` sum) on the decoder and processor hot paths; rocPRIM/hipCUB use DPP row/bank ops on AMD.
4. **"Lane 0 computes, then broadcast"** for values that are SALU-able from ballot results (`decode_strategies.hiph:139,145,149,285`): forces a divergent region + 64-bit `bpermute` where the whole `get_len3_mask_64` chain could be scalar (`s_lshr_b64`, `s_load_dword` of the `__constant__` LUT).
5. **Symbol queue alignment/layout** (`decompression_state.hiph:56-63`, `symbol.hiph:88-107`): `batch[]` at offset 76 → `set/get` are two `volatile` 32-bit LDS ops at 8-B lane stride (2-way conflicts) instead of one aligned `ds_write/read_b64`.
6. **Byte-granular prefetch** (`decompression_prefetch.hiph:111-118`): 8 `global_load_ubyte` + 8 `ds_write_b8` per lane per 512 B although `base+pos` is 64-B aligned by construction.
7. **Byte-granular output** in the processor (`decompression_process.hiph:139,174,199,203,228,250`): 64 B per wave store instruction; AMD permits unaligned dword global access (ROCm default), the CUDA backend does not.
8. **Serial decoder reads one byte per LDS op** (`decompression_decode.hiph:84-134`) with 63 idle lanes — entered on every batch with a literal > 4 chars or a copy-4.
9. **Integer modulo on the copy hot path** (`decompression_process.hiph:188,194`) — software `urem` on AMD.
10. **Launch bounds**: single-argument form on the batched kernels; none on HLIF (`hlif_shared.hiph:254,419`); HIP's second argument = min waves per EU.
11. **Size kernel geometry**: one active lane per 64-lane workgroup, `count` workgroups (`:84-134,207`).
12. **Wave-size**: gfx1100 compiles wave32 by default; without `USE_WARPSIZE_32` the 3-wave block becomes 6 hardware waves and every 64-bit ballot/shuffle is wrong (guarded on `opt/curated` for the batched path; HLIF needs the same guard).
13. Strict-aliasing punning (`device_functions.hiph:206-209`, `warp_scans.hiph:335-348`) — only instantiated for the unused `GROUPMASK_T = uint32_t` on wave64 configuration.
14. Prefetch ring depth: granule is 512 B on wave64 (256 B on CUDA) but the ring stayed 4 KB → 8 granules instead of 16 (`config.h:59-67,116-117`, own TODO).
15. `volatile` everywhere (`decompression_state.hiph:122`, `compression_state.hiph:55-57`) prevents the compiler from merging byte accesses — any widening must be explicit.

Already AMD-aware: wave64 typing (`warp_mask_t`, `BATCH_SIZE = 64`, block sizes `2·/3·warpsize`),
64-bit ballots and `WarpReduce<64,64>`/`<32,64>` specialisations, wave64 `get_len3_mask_64`/`get_len5_mask_64`,
ballot-loop `HashMatchAny`, single-ballot `Match60`, 64-B-aligned prefetch granules, sync-less
`__shfl/__ballot`. LDS per block is < 9 KB static (≈ 7 compress / ≈ 9 decompress blocks per CU on
CDNA by LDS alone), so registers are the probable occupancy binder for the decoder.

### 5.3 Opportunities — compression (`snap_kernel`, HLIF)

| ID | Where | What | Mechanism | Impact | CUDA risk | Preserves | Effort | Measure |
|---|---|---|---|---|---|---|---|---|
| SNP-C1 <span class="chip revert">reverted: −4…−7 % both</span> | `compression.hiph:207-244` | Software-prefetch the per-round `data32` for all ≤ 4 `FindFourByteMatch` rounds up front (8 aligned dword loads per lane), then run the hash/ballot/LDS/verify chain on registers | Removes one global-load latency from the start of rounds 2–4 and overlaps memory with round 1's ballots/LDS; CDNA2 (16 KB L1, high L2 latency) gains most | single digits to ~15 % on literal-heavy input | none | identical match sequence and hash updates | M | random-bytes and text inputs; `SnappyLargeTokens_test` byte-exactness (`:381-447`) |
| SNP-C2 <span class="chip revert">reverted: −1…−6 %</span> | `:338-378` (barriers `:342,377`; fields `compression_state.hiph:55-57`) | Ping-pong the three fields on iteration parity so the first `__syncthreads` per iteration disappears (WARP1 writes slot `(i+1)&1` while slot `i&1` is read) | Thousands of iterations per chunk; halving the barrier count shortens the per-iteration critical path on all targets | ~5–10 % | none | same symbol sequence; only the sync schedule changes | M | benchmark + byte-exact tests |
| SNP-C3 <span class="chip revert">reverted: ≤ +0.7 % / −4 %</span> | `:111-115` (`StoreLiterals`) | AMD only: 4 B per lane (`unaligned_load32` source, `aligned(1)` dword store), byte tail; byte loop on CUDA | 4× fewer VMEM instructions for literal bodies (unaligned dword stores legal under ROCm's unaligned mode — verify `global_store_dword` in ISA) | small; visible on incompressible data | must be guarded | identical bytes | M | incompressible input |
| SNP-C4 <span class="chip todo">open</span> | `SnappyBatchKernels.hip:65`; `hlif_shared.hiph:254` | Platform macro `__launch_bounds__(COMP_THREADS_PER_BLOCK, min_waves_per_eu)` (e.g. 4 on HIP) after reading resource usage; same on HLIF | LDS caps `snap` at ≈ 7 blocks/CU (3.5 waves/SIMD); > 128 VGPRs would drop below that | 0 to moderate | none (separate values) | hints only | S | BLD-2 report; benchmark |
| SNP-C5 <span class="chip kept">kept: +1–3 % gfx942, fixes wave64 half-clear</span> | `:330-334` | Clear `hash_map` with `uint4` stores (16 → 4 `ds_write` per thread) | free | negligible | none | yes | S | — |
| SNP-C6 = HLIF-1 | `hlif_shared.hiph:139-148` | dword scratch copy | | | | | | |

Not proposed (would change output): shrinking `HASH_BITS`, `MAX_LITERAL_LENGTH`, giving WARP0 search
work, multi-chunk blocks sharing barriers. Grid-filling note: MI300X needs > ~2000 chunks in flight
(304 CUs × ~7 blocks); separate per-block latency from occupancy with `-p 32768` vs `65536` and `-x`.

### 5.4 Opportunities — decompression (`unsnap_kernel`, HLIF, size kernel)

| ID | Where | What | Mechanism | Impact | CUDA risk | Preserves | Effort | Measure |
|---|---|---|---|---|---|---|---|---|
| SNP-D1 <span class="chip kept">kept: ≈ / +2–4 % gfx942</span> | `device_functions.hiph:176-181`; `decompression_decode.hiph:168-171,178-180`; `decompression_prefetch.hiph:107`; `decompression_process.hiph:94`; `config.h:101-109` | AMD `NANOSLEEP(ns)` → `__builtin_amdgcn_s_sleep(k)` (start k = 1 decode/process, 2–4 prefetcher); replace the literal `NANOSLEEP(100)` with named constants so the AMD branch of `config.h` means something; poll once before sleeping | With ~6–9 blocks per CU a large share of resident waves are pollers at any moment; `s_sleep` yields the SIMD to productive waves of other chunks. Applies to CDNA2/3 and RDNA3 alike | potentially the largest single decompression win at realistic occupancy (tens of %) | none | pure scheduling | S | decomp GB/s vs k; `SQ_WAIT_INST_LDS`, `SQ_INSTS_LDS`, `SQ_BUSY_CYCLES` |
| SNP-D2 <span class="chip kept">kept: +1–5 %</span> | `device_functions.hiph:227-268`; users `decode_strategies.hiph:160,306`, `decompression_process.hiph:123,127` | Implement `WarpReduce<GROUPSIZE,WARPSIZE>::prefix_sum/sum` with `hipcub::WarpScan<T,GROUPSIZE>::InclusiveSum` / `hipcub::WarpReduce<T,GROUPSIZE>::Sum` (via `arcto_hipcub.hiph`; cub on CUDA); tiny `TempStorage` in the existing LDS state; keep the `<32,64>` logical-warp specialisation | rocPRIM selects DPP row/bank ops on AMD vs six dependent `bpermute` round trips (twelve for 64-bit); the processor's combine loop runs scan + 64-bit reduce + ~5 shuffles per 64 output bytes | 10–25 % on short-symbol data | low | identical numeric results | M | short-symbol input; `SQ_INSTS_LDS` |
| SNP-D3 <span class="chip kept">kept: +3–5 % / +2–11 %</span> | `device_functions.hiph:150-162` and the uniform-broadcast sites | AMD `SHFL10(v)` → `__builtin_amdgcn_readfirstlane` (two halves for 64-bit/pointers); new `SHFL1_UNIFORM(v,i)` → `__builtin_amdgcn_readlane` only where the source lane is wave-uniform (`batch_len-1`, `batch_add-1`, loop index, `n-1`; not `it`) | Moves broadcasts off the LDS crossbar; results land in SGPRs and shrink VGPR pressure | 5–15 % on the decoder/processor critical path | none | same values | S–M (site audit) | ISA `ds_bpermute_b32` count; benchmark |
| SNP-D4 <span class="chip kept">kept: neutral enabler</span> | `decode_strategies.hiph:139,145,149,285`; `decompression_decode.hiph:267-275` | Compute `len3_mask`, `len5_mask`, `batch_len`, `b` on **all** lanes from the ballot results (move the `batch` update after the `batch_len` broadcast so every lane tracks it) | Inputs are SGPR-uniform: the whole `get_len3_mask_64` chain becomes SALU + scalar-cache LUT loads, no divergent region, no 64-bit/pointer shuffles | small–medium per batch; pairs with D3 | none | identical masks | S | ISA: `s_load_dword` + `s_lshr_b64` in the chain |
| SNP-D5 <span class="chip kept">kept: +1–4 %</span> | `decompression_state.hiph:56-63`; `symbol.hiph:88-107` | Pad/reorder so `batch[]` is 8-B aligned; `set/get` through one `volatile uint64_t` | Conflict-free full-rate `ds_write/read_b64` instead of 2 × 32-bit with 2-way conflicts; halves volatile LDS instructions in `set()`/`get()` | small but free | none | storage only | S | `SQ_LDS_BANK_CONFLICT` |
| SNP-D6 <span class="chip kept">kept: +6 % / +2.5 %</span> | `decompression_prefetch.hiph:111-118` | Lane `t` loads `uint32_t` at `base+pos+4t+256i` (aligned by construction), writes 4 bytes to the ring (one `ds_write_b32` when `(pos & 3) == 0`, else 4 byte writes on consecutive banks) — ring contents identical | 8 `global_load_ubyte` → 2 `global_load_dword` per lane per 512 B; same coalescing, 4× fewer instructions | small–medium when the prefetcher is on the critical path (large, poorly compressible chunks) | none | yes | M | incompressible input; VMEM instruction counts |
| SNP-D7 <span class="chip kept">kept: neutral</span> | `decompression_decode.hiph:84-134`; `decompression_state.hiph:70-75` | Lane 0 reads two aligned ring dwords covering `cur..cur+7` (wrap-safe, `4096 % 4 == 0`) and extracts `b0..b4` with shifts/funnel shift; parsing unchanged | 1–5 dependent `ds_read_u8` + waits per symbol → 2 independent `ds_read_b32`; the serial path runs with 63 idle lanes on every batch containing a literal > 4 chars or a copy-4 | medium on text-like / well-compressed data | none | yes | S–M | text/log input; `decompress_large_literal`, `decompress_long_2B/4B_match_case`, `test_snappy_app_1/2` |
| SNP-D8 <span class="chip kept">kept: literals +9…+14 % all AMD; copies wave32 only (−23 % wave64)</span> | `decompression_process.hiph:206-252` (literals), `181-205` (copies), `131-180` | Stage 1: literals from `literal_base` or the ring as 4 B per lane (AMD unaligned-tolerant; byte path on CUDA and tails); stage 2: copies with `dist ≥ 4` and no intra-lane overlap | 4× fewer VMEM store instructions; gfx942/gfx1100 are store-issue bound on literal-heavy streams | medium on incompressible data | must be guarded | identical bytes | M–L | incompressible input; add a test with deliberately misaligned `device_out_ptr` |
| SNP-D9 <span class="chip caution">not pursued: no register headroom (report)</span> | `SnappyBatchKernels.hip:145`; `hlif_shared.hiph:419` | `__launch_bounds__(DECOMP_THREADS_PER_BLOCK, min_waves)` with platform-specific second argument after reading VGPR counts (HIP: 6–7 if ≤ 72 VGPRs, else 4); same on HLIF | LDS allows ≈ 9 blocks/CU (27 waves); registers are the probable binder; D2–D4 reduce VGPRs on their own | unknown until measured, possibly large | none | hints | S | BLD-2 report |
| SNP-D10 <span class="chip kept">kept: neutral</span> | `SnappyBatchKernels.hip:84-134,200-213` | One chunk per thread, 256 threads per block, `grid = ⌈count/256⌉` | Removes 63/64 idle lanes and ~256× fewer workgroups | µs-level | none | yes | S | — |
| SNP-D11 <span class="chip kept">LITERAL_SECTORS=8 wave32 only; ring/granule rejected</span> | `config.h:59-72,116-117` | Evaluate `-DLOG2_PREFETCH_SIZE=13` (8 KB ring restores the 16-granule depth; LDS ≈ 10.3 KB/block, still ≥ 6 blocks/CU), `-DPREFETCH_SECTORS=4`, `-DLITERAL_SECTORS=2/8` | Ring depth governs how far the prefetcher runs ahead; on wave64 it is half as deep as designed | small–medium, input dependent | none | yes | S | matrix |
| SNP-D12 <span class="chip caution">knob: input-dependent (TTI +8 %, words −9 %)</span> | `decompression.hiph:203-210` | 32-lane decode groups inside a 64-wide wave (`DecodeSymbols<…<uint32_t, warp_mask_t>>`, supported by `ballot1<uint32_t,64>`, `WarpReduce<32,64>`, `get_len3_mask<uint32_t,uint64_t>`); fix the punning casts first | Halves the serial `get_len3_mask` chain (8 lookups) and per-batch LDS traffic at the cost of half the lanes; nvCOMP's strategies were tuned for 32 symbols per batch | unknown (either way) | none | yes | S to try | benchmark |
| SNP-D13 <span class="chip caution">speculative</span> | `decompression_process.hiph:138,173,188-194` | LDS mirror of the last N KB of output to serve short-distance copies (N = 4 KB → ≈ 10.3 KB/block) | Only if `TCC`-level latency shows `out - dist` loads as the limiter on CDNA2 (write-through L1) | ? | none | yes | L | gated by profiling |

Hygiene: `static_cast` instead of pointer punning; extend the wave-size guard to the HLIF manager path.

### 5.5 Verify on hardware

VGPR/SGPR/spills for `snap_kernel`, `unsnap_kernel` and both HLIF kernels per arch (decides C4/D9);
polling share (`SQ_WAIT_INST_LDS`, `SQ_INSTS_LDS`, `SQ_BUSY_CYCLES`) before/after D1 and the best
`s_sleep` per role and arch; whether the compiler already scalarises the `get_len3_mask_64` chain
(`s_load_dword`/`s_lshr_b64` vs `v_*`) and ever turns uniform `ds_bpermute` into `v_readlane`;
copy-source latency (`out - dist` hitting TCP or L2: `TCP_TOTAL_CACHE_ACCESSES`, `TCC_HIT/MISS`);
`SQ_LDS_BANK_CONFLICT` for the symbol queue; unaligned-access lowering (`global_store_dword` with
`align 1`) on all targets incl. gfx1100; wave64 vs wave32 on gfx1100 for this codec and D12 on CDNA;
ring tuning (D11); throughput vs `-p` (16–256 KB) and batch size per GPU; rocPRIM DPP selection for
`WarpScan<int,64>` / `WarpReduce<uint64_t,64>` and for 32-lane logical warps on gfx1100. Test
coverage to add before C3/D6/D7/D8: odd chunk sizes, odd input/output alignments, `BATCH_COUNT`
wrap-around, a randomized compressibility sweep cross-checked with python-snappy.

---

## 6. Cascaded (RLE → Delta → BitPack)

### 6.1 Code map and the one structural fact

`arctoBatchedCascadedCompressAsync` (`CascadedBatch.hip:329-357`) → `ARCTO_TYPE_ONE_SWITCH` →
`cascaded_compression_kernel<T,size_t,128><<<batch, 128>>>` → `do_cascaded_compression_kernel`
(`CascadedKernels.hiph:761-1058`): one block per partition, serial over 4096-B chunks; per chunk:
load chunk → LDS (`:896-902`, `sizeof(T)` per lane), layer loop `for layer < max(num_RLEs, num_deltas)`
(`:910-980`): `block_rle_compress` (values → ping-pong buffer, counts → `shared_count_buffer`,
`BlockScan<size_t,128>` for output positions `:157-190`), `block_write<uint16_t>` of the counts
(bit-packed via `block_bitpack` when `use_bp`), swap; `block_delta_compress` + delta header; final
`block_write<data_type>`; thread 0 flushes the ≤ 64-B chunk metadata and the 8-B partition header;
out-of-bound → raw fallback. `get_for_bitwidth` (`:394-471`) computes min/max with **two
`BlockReduce` + two `__syncthreads` per 128 elements** (u8: 64 collectives per array).
Decompression: **four launches** `cascaded_decompression_kernel_type_check<1/2/4/8>`
(`CascadedBatch.hip:387-429`), each reads byte 3 of partition 0 and only the matching width proceeds
(`:166-168`) into `cascaded_decompression_fcn` (`CascadedKernels.hiph:1106-1435`): per chunk,
thread 0 computes `rle_offsets` serially (`:1291-1306`), `block_read` + `block_bitunpack`, reverse
layer loop with `block_delta_decompress` (one `BlockScan` + barrier per 128 elements, `:343-377`) and
`block_rle_decompress` (scan per 128 runs then **each thread serially expands its own run**,
`:290-296`), copy LDS → global (`:1404-1407`, `sizeof(T)` per lane). Temp space is 0 for both.
HLIF wraps the same functions with `chunks_per_block = 1`.

**LDS / occupancy.** Static LDS per 128-thread block (compression `:789-837`, decompression
`compute_smem_size` `:1069-1080`): u8 ≈ 24.7 KB, u16 ≈ 16.5 KB, u32 ≈ 12.4 KB, u64 ≈ 10.4 KB, plus
hipcub `TempStorage`. On CDNA (64 KB LDS per CU, a 128-thread block = 2 wave64) that is 2 / 3 / 5 /
6 blocks per CU → **1–3 waves per SIMD out of 8**; every `__syncthreads`/collective round trip is
fully exposed. On RDNA3 (wave32, 128 KB per WGP) 5–12 blocks per WGP → 5–12 waves per SIMD. The
kernels are barrier/latency bound, so this is the single most important fact for the codec. None of
`amd_optimizations.h` is used on this path; no kernel carries `__launch_bounds__`; hipcub
`BlockScan` is left on `BLOCK_SCAN_RAKING` (`:157, 273, 350`).

### 6.2 Pitfalls (file:line)

1. 128-thread blocks + 10–25 KB static LDS ⇒ LDS-bound occupancy (above).
2. No `__launch_bounds__` (`CascadedBatch.hip:102,156,262`; `hlif_shared.hiph:232-236,391-401`) ⇒
   compiler budgets for 1024-thread groups. The `AMD_LAUNCH_BOUNDS_*` macros (`amd_optimizations.h:104-113`)
   document the second argument as "min blocks per CU"; in HIP it is **min waves per EU** — `(256,8)`
   would cap VGPRs at 64 on CDNA.
3. `BLOCK_SCAN_RAKING` (→ rocprim `reduce_then_scan`) instead of `BLOCK_SCAN_WARP_SCANS` (→ `using_warp_scan`, DPP/`ds_swizzle`).
4. One collective + barrier per 128 elements everywhere (`get_for_bitwidth`, delta decompress, RLE decompress).
5. `block_rle_compress` gives each thread **32 consecutive bytes** (`:160-217`) ⇒ 8-dword lane stride ⇒ ~8-way LDS bank conflicts on every `input_buffer` read, two passes per layer (same stride in the legacy RLE kernels).
6. Serial, divergent RLE expansion (`:290-296`): one long run keeps 63 lanes idle.
7. Scalar per-element global traffic: `sizeof(T)` per lane for chunk load / final store / fallback; `uint32` per lane in `block_read/write`; no `dwordx4` anywhere.
8. HLIF scratch → output copy uses byte stores although Cascaded chunk sizes are multiples of 4 (`:1041-1052`).
9. Four decompression launches per call, three no-ops (~5–15 µs launch latency each on ROCm); also assumes all partitions share partition 0's type.
10. 64-bit arithmetic for in-chunk quantities (`BlockScan<size_t>`, `num_outputs`, `out_bytes`) — VALU pairs + 64-bit LDS scan traffic.
11. `blockDim.x` vs the template constant mixed (`:324, 585, 675, 898, 1026, 1244`) — blocks unrolling.
12. Two runtime integer divisions by `bitwidth` per packed output word (`:531-532`).
13. Thread-0 serial sections (metadata flush `:1004-1014`, `rle_offsets` `:1291-1306`).
14. Device `assert()` (`:783, 826, 844, 1209`) — compiled out only with `-DNDEBUG` (BLD-1).

### 6.3 Opportunities — shared by compression and decompression

| ID | Where | What | Mechanism on CDNA2 / CDNA3 / RDNA3 | Impact | CUDA risk | Preserves | Effort | Measure |
|---|---|---|---|---|---|---|---|---|
| CAS-S1 <span class="chip kept">kept: gfx942 comp ×1.5, decomp ×1.25–1.7</span> | `CascadedKernels.hiph:94-95`; launches `CascadedBatch.hip:294-295, 387-429`; HLIF `CascadedHlifKernels.hip:174, 214, 290, 344` | `threadblock_size = wave64 ? 256 : 128` (device code is already templated on it; audit the `blockDim.x` uses) | LDS per block is fixed by the 4 KB chunk, not by thread count: 256 threads doubles resident waves per CU at the same LDS (u32: 5 blocks × 4 waves = 5 waves/SIMD vs 2.5; u8: 2/SIMD vs 1), halves every "per-128" round, halves `num_inputs_per_thread` (16-B stride → 4-way instead of 8-way conflicts). RDNA3 also gains (8 waves per block) | comp 15–30 %, decomp 10–25 % | low | thread→element mapping only; RLE order and bitpack words are index-defined | S | `benchmark_cascaded_chunked -p 65536 -w 3 -i 20`; `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQ_WAIT_INST_LDS`; `test_cascaded_batch` (8 types × bp) |
| CAS-S2 <span class="chip kept">kept: enabler</span> | `CascadedBatch.hip:102,156,262`; `hlif_shared.hiph:232-236,391-401` | `__launch_bounds__(threadblock_size)` (+ on HIP a waves-per-EU target of 2–4); introduce `ARCTO_LAUNCH_BOUNDS(threads, min_waves_per_eu_hip, min_blocks_per_sm_cuda)` because the second argument differs in meaning; fix the comment in `amd_optimizations.h:98` | Tells the compiler the true group size (128/256) so it can use up to 256/128 VGPRs without spilling and balance against the LDS-derived occupancy ceiling; enabler for S1/S3 | 0–5 % | low if macro-wrapped | hints only | S | `.s` metadata (`.vgpr_count`, `.group_segment_fixed_size`, scratch) |
| CAS-S3 <span class="chip todo">open</span> | comp `:795-823`; decomp `:1069-1080, 1135-1179`; `CascadedBatch.hip:174-241`; `CascadedHlifKernels.hip:139-143` | Shrink static LDS: alias `temp_count_array` with the dead ping-pong buffer during count bit-unpack (= CAS-D5), put the hipcub `TempStorage`s in one explicit union (never simultaneously live), drop duplicated `+4+sizeof(T)` slack | Occupancy ceiling is `⌊64 KB / LDS⌋`: u8 decompression 24.7 → ≤ 21.3 KB moves from 2 to 3 blocks/CU (+50 % waves) | u8/u16 10–30 % on CDNA; u32/u64 0–10 % | none | LDS is scratch | M | `hipOccupancyMaxActiveBlocksPerMultiprocessor` printout (HLIF already computes it); out-of-bound tests (`test_cascaded_batch.cpp:992-1009`) |
| CAS-S4 <span class="chip kept">kept: +3 % gfx1100, +15–19 % gfx942</span> | `:157, 273, 350` (+ legacy `RunLengthEncodeGPU.hip:286`) | `hipcub::BlockScan<T, N, hipcub::BLOCK_SCAN_WARP_SCANS>` | rocprim `using_warp_scan`: per-wave DPP/`ds_swizzle` scan, one LDS exchange of wave prefixes, fewer `s_waitcnt lgkmcnt(0)` + barriers than `reduce_then_scan`; identifier exists in CUB too | 2–5 % | low | integer scan, same result | S | `SQ_INSTS_LDS`, `SQ_LDS_BANK_CONFLICT`, `SQ_WAIT_INST_LDS` |
| CAS-S5 <span class="chip kept">kept: compression-side 16-B copies +7–22 %; decomp store scalar</span> | comp chunk load `:896-902`, `block_write` `:673-677`, fallback `:1025-1028`; decomp `block_read` `:722-725`, final store `:1404-1407`, raw `:1242-1246`; HLIF `hlif_shared.hiph:138-157` | Runtime alignment check (`& 15`) then `uint4` main loop + scalar tail; at least the `uint32` path for u8/u16 | 16 B per lane per instruction (`global_load_dwordx4` / `global_load_b128`) ⇒ 4–16× fewer memory instructions and more bytes in flight — critical with 1–3 resident waves/SIMD; byte/short stores also cost partial-line write combining in TCP/L2. `hipMalloc` gives 256-B alignment and HLIF chunk offsets are multiples of 4096, so the fast path dominates; the API only guarantees `data_type` alignment, so the check is mandatory | u8/u16 10–25 % of copy phases; u32/u64 5–10 %; HLIF copy 2–4× | low (`uint4` is plain C++) | same bytes | M | `TCP_TOTAL_CACHE_ACCESSES`, `TCC_EA_WRREQ/RDREQ` per byte; `.s` for `dwordx4` |
| CAS-S6 <span class="chip kept">kept: +1–4 %</span> | `:157-190, 837-838, 890-892, 1314, 1371` | `uint32_t` for in-chunk quantities and the RLE scan type (values < 65536 by construction, `:783`) | 64-bit adds/compares are VALU pairs; halves the scan's LDS traffic; fewer VGPRs | 1–4 % | none | yes | S | VGPR count |
| CAS-S7 <span class="chip todo">open</span> | `:324, 585, 675, 898, 1026, 1244` | Use the template constant instead of `blockDim.x`; `#pragma unroll` short loops | Folds trip counts, removes the dispatch-packet SGPR load | 0–2 % | none | yes | S | `.s` |
| CAS-S8 <span class="chip todo">open</span> | `block_write` `:673-677`, final store `:1404-1407`, HLIF copy | `__builtin_nontemporal_store` (AMD) / `__stcs` (CUDA) on the vectorised path | Streams write-once output past L2 / Infinity Cache so it does not evict the input working set; more relevant on RDNA3 (small L2) and CDNA3 (shared MALL) | 0–5 % | medium (guard) | yes | S | `TCC_HIT/MISS` |
| CAS-S9 = BLD-4 / BLD-1 | | `-mcumode` on gfx1100; confirm `-O3 -DNDEBUG` | CU mode halves the LDS pool per workgroup group but may raise occupancy for 12–25 KB blocks | 0–10 % gfx1100 (sign unknown) | none | yes | S | benchmark matrix |

### 6.4 Opportunities — compression

| ID | Where | What | Mechanism | Impact | CUDA risk | Preserves | Effort | Measure |
|---|---|---|---|---|---|---|---|---|
| CAS-C1 <span class="chip kept">kept: comp ×1.5–1.7 / ×1.4–1.6</span> | `:394-471` (`get_for_bitwidth`) | Thread-local min/max over `i = tid; i < n; i += threads` in registers (no barriers), then **one** `BlockReduce(Min)` + **one** `BlockReduce(Max)` (or one reduce on a `{min,max}` pair); keep the signed interpretation and `__clz` math verbatim | Removes 2×(rounds−1) block reductions and barriers per bit-packed array (u8: 62 → 2), called for every RLE count array and the final array (3× per chunk with `{2 RLE, bp}`); with 1–3 waves/SIMD the barrier chain is the dominant latency; unit-stride LDS reads are conflict-free | **20–40 %** of compression time with `use_bp` | none | min/max are order-independent; FOR and bitwidth bit-identical | S | default opts vs bp-off hack; barrier count in `.s`; `test_predefined_cases<T>(1)` (`test_cascaded_batch.cpp:919-936`) |
| CAS-C2 <span class="chip kept">kept: comp +15–22 % both</span> | `:160-217` (`block_rle_compress`) | Keep the element→thread assignment; load each thread's 32-B slab with two `ds_read_b128` (`uint4` alias) into registers and iterate there; `val_buffer[output_idx] = val` instead of re-reading `input_buffer[idx]` (`:205, 213`). Alternative: pad the LDS staging stride by 1 dword per slab when writing at `:896-902` | 8 lanes × 16 B per LDS cycle cover contiguous-per-lane slabs and hit all 32 banks (near conflict-free vs 8-way today); halves LDS instructions for steps 1 and 3 | 5–12 % per RLE layer | none (`uint4` from `__shared__` is fine on CUDA) | identical per-thread comparison sequence and outputs | M | `SQ_LDS_BANK_CONFLICT`, `SQ_LDS_DATA_FIFO_FULL`; RLE-only config |
| CAS-C3 <span class="chip todo">open</span> | `:167-183, 196-217` | With C2 the slab is in registers: compute `next_val != val` flags once (bitmask), reuse in step 3 | Halves LDS reads per layer | 2–5 % | none | yes | S (with C2) | as C2 |
| CAS-C4 <span class="chip kept">neutral-kept</span> | `:523-552` (`block_bitpack` inner loop) | Precompute `input_idx_start/end` per output word with a reciprocal (`__umulhi` magic for `bitwidth ≤ 64`) or iterate incrementally; keep word-per-thread ownership (no cross-thread word sharing, no atomics) | Removes two `udiv` expansions (~60–80 VALU) per 32-bit output word | 2–6 % of BP phase | none | same word values | S–M | `.s` for `v_rcp`/udiv expansions |
| CAS-C5 <span class="chip todo">open</span> | `:925-936, 955-977` | Run the RLE-count bit-pack and the following delta of the values in the same barrier interval (independent buffers) | Cuts 2–3 barriers per RLE layer and lets VALU/LDS work of the two phases interleave | 2–5 % | none | no data dependence | S | barrier count |
| CAS-C6 <span class="chip todo">open</span> | `:1004-1014` | Threads `0..meta_words-1` store one metadata word each | Removes a serial 4–16-store loop between two barriers | < 1 % | none | yes | S | — |
| CAS-C7 <span class="chip caution">optional</span> | `CascadedBatch.hip:294-295`, `CascadedKernels.hiph:840-1057` | Chunk-parallel compression across blocks for few, large partitions (scratch + scan + compaction; API already allows non-zero temp) | Fills 304 CUs when `batch ≪ #CUs`; costs an extra write+read of compressed data | workload-dependent | low | format identical | L | large `-p` |

### 6.5 Opportunities — decompression

| ID | Where | What | Mechanism | Impact | CUDA risk | Preserves | Effort | Measure |
|---|---|---|---|---|---|---|---|---|
| CAS-D1 <span class="chip kept">kept (D1w wave-local): zeros ×6.4 / ×2.5–3.9, ints ×1.28 / ×1.43</span> | `:255-305` (`block_rle_decompress`) | Keep the per-round `ExclusiveSum`; replace the per-thread serial fill (`:290-296`) with a cooperative fill — scanned offsets of the round's ≤ 128 runs in LDS, all threads loop over output positions and find their run with a 7-step branchless upper bound; or two-level: own run if `count ≤ 8`, long runs expanded by the whole block | A 4096-element run currently costs 4096 serial LDS stores on one lane while 63 idle (2× worse on wave64 than warp32); becomes ~`aggregate/threads` unit-stride conflict-free stores per thread — exactly the data Cascaded is chosen for | 10–50 % on RLE-heavy data; ~0 % short runs | none | identical output array | M | short-run vs long-run datasets; `SQ_WAVE_CYCLES` vs `SQ_BUSY_CYCLES` |
| CAS-D2 <span class="chip kept">kept: ints +9 % both</span> | `:343-377, 278-304` | Multi-item `BlockScan<T,N>::ExclusiveScan(in[ITEMS], out[ITEMS], init, op, aggregate)` with `ITEMS = 4–8` (blocked arrangement, 16-B LDS reads), carry `initial_value` across rounds as now | Divides block scans and barriers by `ITEMS` (u8: 32 → 4–8 rounds); per-thread part in registers | 10–20 % of the delta phase | low (CUB has the overloads) | integer prefix sums identical; order identical | S–M | `num_deltas=1` config; `SQ_INSTS_LDS` |
| CAS-D3 <span class="chip caution">conditional</span> | `CascadedBatch.hip:366-436` | One launch that switches on the header byte and calls the four `cascaded_decompression_fcn<uintN_t>` with one `__shared__` buffer sized by the max — only if CAS-S3 makes the union affordable (else leave and document) | ROCm launch latency × 3 wasted launches per call; relevant for small batches | 0–10 % small batches | low | yes | S–M | `rocprofv3 --kernel-trace` gaps |
| CAS-D4 <span class="chip todo">open</span> | `:1291-1306` | Wave-level scan of the ≤ 8 `rle_offsets` (hipcub `WarpScan` for portability) or overlap with `block_read` | Removes a serial section + barrier per chunk | < 1–2 % | low | yes | S | — |
| CAS-D5 <span class="chip kept">kept: zeros +12–17 %, ints +13 % gfx1100</span> | `:1166-1176, 1356-1363, 1077-1079` | Alias `temp_count_array` with `shared_output_buffer` (dead while the count array is read); `tot_count_bytes = 1×…` | u8 24 688 → 16 496 B (2 → 3 blocks/CU), u16 16 496 → 12 400 (3 → 5), u32 12 400 → 10 352 (5 → 6) | 10–30 % u8/u16 | none | scratch only | S | occupancy printout; out-of-bound tests |
| CAS-D6 <span class="chip todo">open</span> | `:563-618` (`block_bitunpack`) | For `bitwidth ≤ 32` load the two `uint32` words once per element (aligned `ds_read_b32` + funnel shift, `v_alignbit_b32`) instead of `unsigned_type`-width sub-dword reads; optionally 2–4 outputs per thread from one 64-bit window | Fewer sub-dword LDS ops (LDS handles b32 natively) | 2–8 % of BP phase u8/u16 | low (`__funnelshift_r` exists in CUDA) | exact bit extraction semantics | M | u8 data |
| CAS-D7 <span class="chip caution">optional</span> | `:1268-1414`; `CascadedBatch.hip:366-436` | Chunk-parallel decompression within a partition (O(k) header walk or a two-kernel offset pass; status via atomicMin) | Fills CUs when `batch` is small | workload-dependent | low | yes | L | large `-p` |

### 6.6 Verify on hardware

VGPR/LDS/occupancy for `cascaded_compression_kernel<T>` and the four decompression instantiations
per arch (BLD-2), incl. `private_segment_fixed_size` (spills); per-phase time split (`get_for_bitwidth`
rounds vs RLE vs bitpack vs copies — the benchmark has no CLI for `num_RLEs/num_deltas/use_bp/type`,
edit `benchmark_cascaded_chunked.cu:38-45` or add flags); hipcub algorithm mapping on the installed
ROCm (S4); `SQ_LDS_BANK_CONFLICT / SQ_LDS_IDX_ACTIVE` in `block_rle_compress` (C2); launch-overhead
share of the three no-op launches (D3); `-mcumode` and wave64 on gfx1100; long-run datasets for D1;
whether CDNA3's XCD-partitioned L2 (4 MB per XCD) changes the value of the HLIF scratch copy.
Test-coverage gap: no cases for `num_RLEs = 0`, `num_deltas = 0`, `num_RLEs > 2`, or non-default
`chunk_size` — add them before touching LDS aliasing (S3/D5) or the RLE paths (C2/C3/D1).
Correctness side-notes (not perf): decompression derives the type from partition 0 only
(`CascadedBatch.hip:166-168`); `cascaded_decompress_wrapper::get_output_status()` never propagates
the status written by `cascaded_decompression_fcn` (`CascadedHlifKernels.hip:136,159-164`);
`CascadedManager.hpp:92-100` passes default opts to `compute_max_compressed_chunk_size` (harmless today).

---

## 7. ZFP (vendored LLNL zfp, HIP backend)

### 7.1 Code map

Submodule at LLNL `cccbb9d` (`develop` after 1.0.x, `0.5.0-1259-gcccbb9d`; HIP backend present,
generated from `src/cuda` by `hipify-perl` — `hip/Makefile:1-39`). `opt/curated` repoints
`.gitmodules` to `github.com/cristianokunas/zfp` branch `amd-hip` (same commit); kernel work goes
there and the pin moves deliberately. ARCTO forces `ZFP_WITH_HIP` (AMD) / `ZFP_WITH_CUDA` (CUDA),
static, tests/CLI/Python off (`CMakeLists.txt:263-310`); zfp inherits ARCTO's platform/wave defines
and `-fPIC`; no `-ffast-math` anywhere (must stay that way).

**Compress** `arctoZFPCompress` (`ZFPBatch.cpp:222-331`): `serialize_header` (`:195-220`) sets the
HIP execution policy on a throwaway stream → `device_init()` (`hip/device.h:19-54`: hipMalloc +
1×1 kernel + **hipHostMalloc** + sync memcpy + hipHostFree + hipFree); payload stream opened at
`h_output + header_bytes` (the HIP encoder writes from word 0 of `stream->begin`, `:186-194`);
`configure_stream` again → second `device_init()`; FIXED_RATE sets `minbits = maxbits = 64·rate` (3-D),
FIXED_PRECISION / FIXED_ACCURACY create an offset index with granularity 1 (`:271-282`); REVERSIBLE
rejected. `zfp_compress` (`zfp.c:1303-1405`; 4-D entries NULL → 4-D unsupported on GPU although
`arcto/zfp.h:73` allows it) → `zfp_internal_hip_compress` (`hip/interface.cpp:44-132`): field
zero-copy if it is a device pointer; **hipMalloc of the whole `stream_capacity`** for the output;
`maxbits = zfp_maximum_block_size_bits` (fixed-rate K: 64K bits; fixed-precision p: 72 + 64·min(p,32);
fixed-accuracy: 2120 bits whatever the tolerance) is both the bit budget and the **slot stride** of
the uncompacted stream; `encode3` = **hipMemset of blocks × maxbits** then `encode3_kernel<<<grid,128>>>`
on the NULL stream; variable-rate: `compact_stream` (`hip/variable.h:367-412`: per chunk of
`processors×1024` blocks `copy_length_kernel` + `hipcub::DeviceScan` + cooperative
`compact_stream_kernel`; then `align_stream_kernel<<<1,1>>>` + sync 8-B D2H; two hipFree);
`cleanup_device` = **sync pageable D2H of the payload + hipFree**; back on the host `encode_index_offset`
(`zfp.c:59-96`) does a **serial prefix sum over all blocks**; ARCTO appends the index and a 20-B
trailer (`:293-320`). **Decompress** `arctoZFPDecompress` (`:413-556`): host header parse, trailer
sniff (magic `AZFP`), variable-rate without trailer → CPU serial fallback (`:367-411`); one
`device_init()`; `zfp_internal_hip_decompress` (`interface.cpp:134-238`): **hipMalloc + sync H2D of the
payload** (`hip/device.h:161-171`), index H2D (`:185-195`, reads 8 B past the index — upstream
quirk), `decode3`: hipMalloc + memset of `d_offset`, `decode3_kernel<<<grid,128>>>`, **sync D2H 8 B +
hipFree**; three hipFree. Everything is on the NULL stream; `hipFree` implies device-wide sync.

**HIP backend structure.** One kernel per dimensionality, templated on Scalar; **one thread = one
zfp block** (64 values in 3-D), `block_idx = global thread id`, 128-thread groups; per thread:
gather 64 scalars (`encode3.h:11-17`, x-inner contiguous, lanes 16 B apart), `fwd_cast`,
`fwd_xform` lifting, `fwd_order` + `int2uint` into `ublock[64]`, `encode_ints` / `encode_ints_prec`
(`encode.h:235-302`): per bit plane build the 64-bit plane with 64 shift/and/or ops (`:251-252`),
write the first *n* bits, then the **bit-by-bit unary group-test loop** (`:258-260`); writer
`BlockWriter::write_word` = **64-bit `atomicAdd` per word** (`writer.h:19-20`, because adjacent
blocks share a word when `maxbits % 64 ≠ 0`; relies on the memset zero-fill). Decoder mirror with
`BlockReader` 8-B loads (`reader.h:20`) and `scatter3` 64 scalar stores. Private arrays
`fblock/iblock/ublock[64]` → ≥ 128 live VGPRs (float), ≥ 256 (double). HIP and CUDA backends are
feature-identical (variable-rate compaction, writer, 1–3-D, offset + hybrid index); neither has
4-D or reversible; the HIP one lacks only `hipMallocAsync` and managed-memory detection.

### 7.2 Pitfalls

1. `CUDART_VERSION` guards left by hipify (`encode.h:111,226,246,280`; `decode.h:36,163,191,225`) — benign (full unroll).
2. Warp-32 hard-coded hybrid-index path (`decode.h:415-432`: `/32`, `warp_idx*9`, `__shared__ uint64 offset[32]`, `__syncthreads` in divergent code) while kernels launch 128 threads → OOB LDS — broken upstream; ARCTO only emits `zfp_index_offset`, but a foreign trailer with type 3 would reach it → reject hybrid in the wrapper (ZFP-H8).
3. `__launch_bounds__(256, 1)` on `encode3/decode3` (`encode3.h:47`, `decode3.h:41`): on HIP "1" = min 1 wave per EU, i.e. no VGPR constraint → likely 1–2 waves/SIMD; the launch uses 128 threads anyway.
4. SM-centric constants: 48 KB LDS cap (`variable.h:329`; AMD allows 64 KB), `threads_per_sm = 1024` (`device.h:200`; CDNA CUs hold 2048).
5. 64-bit `atomicAdd` per emitted word (L2 atomics; only needed on slot-edge words; never when `maxbits % 64 == 0`).
6. Thread-per-block on wave64: the data-dependent unary loops run to the slowest of 64 blocks (vs 32); small fields under-fill (64³ = 4096 blocks = 64 wave64 for a 440-SIMD MI250X GCD — the shipped 1-MB fixtures are exactly that).
7. Strided scalar gathers/scatters: each VMEM instruction touches 64 lanes × 4 B at 16-B stride; 64 partial-line stores per thread in the decoder rely on L2 write combining; the 4-value row is a natural `dwordx4`.
8. Cooperative `grid.sync()` (software barrier on AMD) in compaction; all groups must be co-resident.
9. `malloc_async/free_async` are plain `hipMalloc/hipFree` (`share/device.h:95-105`).
10. `is_gpu_ptr` excludes managed memory (`share/device.h:83-87`).
11. 3 × 64-bit div + 3 × mod by runtime `bx,by,bz` per block (`encode3.h:78-80`, `decode3.h:90-92`) — hundreds of VALU ops.
12. Everything on the NULL stream; no stream parameter in `arcto/zfp.h:111-113`.

### 7.3 Gate first: ZFP-T1 <span class="chip kept">done: test_zfp_payload_exact, 308/308 byte-exact on gfx1100 and gfx942</span>

Current tests are tolerance-based (`test_zfp_bitcompat.cpp:9-12,143-146,196-199`; `test_zfp_canonical.cpp:143-150`)
on a 64×64×32 float field. Add a test comparing the **payload region** produced by the HIP path against
`zfp_exec_serial` byte-for-byte: fixed-rate K ∈ {8, 12.5, 16, 24} (non-integer rate exercises word
sharing), fixed-precision, fixed-accuracy; float/double/int32/int64; `nx % 4 ≠ 0`; 1-D/2-D; all-zero
blocks. Every kernel change below is gated by it.

### 7.4 Opportunities — compression (inside the fork)

| ID | Where | What | Mechanism | Impact | Risk | Effort | Measure |
|---|---|---|---|---|---|---|---|
| ZFP-C1 <span class="chip todo">open</span> | `encode3.h:11-17`, `encode2.h:11-16` (`gather3/2`) | When `sx == 1` and the row start is 16-B aligned, load each x-row as one `float4` / `double2`×2; scalar fallback for partial blocks | Lanes hold consecutive x-blocks → each row-load covers a contiguous 1 KB per wave64 (`global_load_dwordx4`), 4× fewer VMEM instructions, lower TA/TD pressure; MI300X (VMEM-issue bound) gains most | M on encode (input read is the largest stream at K ≤ 16) | needs `sy,sz` multiples of 4 elements; per-row fallback otherwise | S | `benchmark_zfp_single -3 512,512,512 -m fixed_rate -r 16`; `SQ_INSTS_VMEM_RD`, `FETCH_SIZE` |
| ZFP-C2 <span class="chip todo">open</span> | `writer.h:19-20` | Track the word index within the block; plain store for interior words, `atomicAdd` only for the slot's first/last word; none when `maxbits % 64 == 0` | Removes L2 RMW atomics for ≥ 87 % of words (K=16: 14 of 16); same trick upstream uses in `store_subchunk` (`variable.h:145-152`) | L–M alone; enabler for C3/C9 | pre-zeroing must remain | S | `TCC_ATOMIC`, encode time |
| ZFP-C3 <span class="chip todo">open</span> | `interface.cpp:86`, `encode3.h:86,133-134`, `variable.h:228` | Variable-rate: decouple the canonical bit budget `maxbits` from the uncompacted **slot stride** = `round_up(maxbits, 64)` → no word sharing, all plain stores, and the memset can go (compaction reads exactly `length` bits and masks the tail, `variable.h:64-71,103-104`). Fixed-rate keeps the exact stride (it *is* the layout) and either the memset or in-kernel zero padding | Fixed-accuracy float 3-D slot is 2120 bits → memset ≈ 1.03× input plus a 1.06×-input device allocation per call; dropping the memset saves a full HBM write pass | M (variable rate) | slots are scratch → bitstream untouched; keep `with_maxbits<>()` logic identical | S–M | T1; `hipMemset` disappears in the trace |
| ZFP-C4 <span class="chip todo">open</span> | `encode.h:251-252, 285-286`; decoder `decode.h:196-197, 230-231` | SWAR bit-plane transpose: build all planes once with a recursive 64×32 (float) / 64×64 (double) bit-matrix transpose on `uint64` words, then read `planes[k]`; decoder inverse at the end | ~8 K VALU ops/block for plane extraction today (64 × 4 ops × 32 planes) → ~0.5–1 K; pure data movement | M–H (encode/decode are ALU/latency bound with thread-per-block) | register pressure (`planes[32]` u64 = 64 VGPRs, `ublock` dies after) ≈ neutral; bit-exact | M | `SQ_INSTS_VALU`; T1 |
| ZFP-C5 <span class="chip todo">open</span> | `encode3.h:77-80`, `encode2.h:67-69`; `decode3.h:89-92`, `decode2.h:82-84` | 32-bit math when `blocks < 2^32`, or a 3-D grid (`blockIdx.y/z` ↔ `by/bz`) so no division is needed | 3 div + 3 mod in 64-bit ≈ 500–1000 VALU ops/block on AMD | L–M (5–10 % of per-block ALU) | none | S | `SQ_INSTS_VALU` |
| ZFP-C6 <span class="chip todo">open</span> | `encode3.h:47,121`, `decode3.h:41,131`; zfp HIP flags | Report resource usage per arch (BLD-2); sweep `__launch_bounds__(128, W)` W ∈ {2,3,4} and WG {64,128,256}; keep `-O3`; **never** `-ffast-math`/DAZ (would change `exponent()` on subnormals, `encode.h:55-72`); HIP's `-ffp-contract=fast` is harmless (no a·b+c in `fwd_cast`/`inv_cast`) | With waves_per_eu(1) the compiler may use ~200–256 VGPRs → 1–2 waves/SIMD on CDNA; doubles likely spill; RDNA3 less affected | unknown, potentially M | spills if over-constrained | S | resource report; `MeanOccupancyPerCU`; scratch traffic |
| ZFP-C8 <span class="chip todo">open</span> | `encode.h:256-260, 288-291` | ctz-based run-length emission of the unary group tests: after the first *n* bits, while `x ≠ 0` emit `1`, then `t = ctz(x)` zeros and a `1` (respecting the `n < BlockSize−1` truncation), advance `n` by `t+1`; `x == 0 && n < BlockSize` → emit `0`; use `write_bits` in O(popcount) steps, clamping to the fixed-rate budget (same truncated bit string) | Replaces up to 64 divergent `write_bit` iterations per plane with ≤ popcount iterations; on wave64 the loop runs to the slowest lane, so AMD gains more | potentially the largest kernel-side win (the coder dominates) | subtle corners (`n < BlockSize−1`, budget exhaustion mid-run) → exhaustive plane test vs the serial coder + T1 | M | T1 + plane test; time |
| ZFP-C9 <span class="chip todo">open</span> | new path in `encode3_kernel` / `BlockWriter` | Fixed-rate: a wave's 64 blocks occupy a contiguous `64 × maxbits` span → write words to LDS (per-block stride padded to W+1 to avoid 64-way bank conflicts), zero-fill in LDS, store the span with coalesced `dwordx4` plain stores → no atomics, no memset | Today each word-write instruction touches 64 distinct lines for 512 B useful (12.5 % line use) | M on encode | LDS: K=16 → 8 KB/wave (~8 waves/CU cap), K=32 → 16 KB; integer rates / aligned slots only | M | `TCC_REQ`, `WRITE_SIZE` |
| ZFP-C7 <span class="chip caution">later</span> | new kernel beside `encode3_kernel` | Wave-cooperative 3-D block encode: 16 lanes per block (one x-row per lane, `dwordx4`), lifting along y/z via DPP/`__shfl_xor` within the 16-lane group, per-plane bits via `__ballot` (a wave64 ballot *is* the 64-bit plane), serial coder on one lane per block unless combined with C8 | Kills the 192-VGPR-per-thread footprint, perfectly coalesced gathers, 16–64× more waves for small fields (64³: 4096 waves instead of 64) | H for small/medium fields; uncertain for large unless C8 lands | large; permutation before ballot; subtle | L | only after C1–C6/C8; T1 |

### 7.5 Opportunities — decompression

| ID | Where | What | Mechanism | Impact | Risk | Effort | Measure |
|---|---|---|---|---|---|---|---|
| ZFP-D1 <span class="chip todo">open</span> | `decode3.h:9-17, 85`, `decode2.h:11-16` | Row stores as `dwordx4` when `sx == 1` and aligned; zero `fblock` only on the all-zero-block path (`decode.h:300-306`) | 64 partial-line stores → 16 fully coalesced 1-KB wave stores; −64 `v_mov`/block | M (output write is the dominant stream) | alignment as C1 | S | `SQ_INSTS_VMEM_WR`, `WRITE_SIZE` |
| ZFP-D2 <span class="chip todo">open</span> | `reader.h:20` | (a) read 16 B per refill into a 2–4-word buffer; (b) LDS-stage the wave's contiguous input span (fixed-rate integer K) with coalesced loads, per-block stride W+1 | Lanes are one block apart (e.g. 128 B): each 8-B read instruction touches 64 lines for 512 B useful; TCP L1 (16 / 32 KB) cannot hold 64 lines × resident waves → ~8× L2 request amplification | M | variable-rate blocks are unaligned → (b) needs funnel shifts like `load_block` | S / M | `TCC_REQ` vs `FETCH_SIZE`, `TCC_HIT` |
| ZFP-D3 <span class="chip todo">open</span> | `ZFPBatch.cpp:280`, `zfp.c:59-96`, `device.h:185-195`, `zfp.c:1494-1505` | (a) granularity 4/8 (index ÷ g; decoder does g serial blocks/thread) or (b) store the 16-bit **length** table (0.25 bit/value) in the trailer, rebuild 64-bit offsets on device with a hipcub scan and hand zfp a device offset buffer (patch `zfp_decompress` / `setup_device_index_decompress`) | Offset index with g = 1 costs 1 bit/value (+12–50 % stream at typical 2–8 bit/value), a serial host prefix sum, two host memcpys and an 8 B/block H2D per decompress | ratio + decode setup time | only ARCTO's trailer changes (bump version byte); payload identical | M | ratio in `-m fixed_accuracy/fixed_precision`; decode time |
| ZFP-D4 <span class="chip kept">kept: fixed-rate bit count on host</span> | `decode{1,2,3}.h:146-148, 176-178` | Fixed-rate: `bits_read = blocks × maxbits` is known on host; variable-rate: persistent pinned + device scalar, async copy, single sync | Saves hipMalloc + memset + sync D2H + hipFree per decompress | small per call, large relative for small fields | none | S | API trace; 64³ decode wall time |
| ZFP-D5 = ZFP-C6 | `decode3.h:41` | launch-bounds sweep | decode holds `fblock/iblock` (aliased) + `ublock` ≈ 128 VGPRs (float) | unknown | spills | S | resource report |

### 7.6 Opportunities — host side (ARCTO wrapper + fork host code)

| ID | Where | What | Mechanism | Impact | Effort |
|---|---|---|---|---|---|
| ZFP-H1 <span class="chip kept">kept: host calls ×2–3 (64³)</span> | `hip/interface.cpp:41`, `hip/device.h:19-54`; `ZFPBatch.cpp:157, 207, 261, 538` | Do not set the HIP policy on throwaway streams (`serialize_header`, `GetMaxOutputSize`); guard `device_init()` with the existing `static bool initialized`; reuse a thread-local `zfp_stream` | Each init = hipMalloc + kernel + **hipHostMalloc** + sync memcpy + hipHostFree + hipFree (pinned registration is among the slowest ROCm calls); ARCTO triggers it 2×/compress, 1×/decompress, 1×/size query | large for small fields; measurable always | S |
| ZFP-H2 <span class="chip kept">(a) kept: pool allocs; 256³ ×3.6–4.6 on APU</span> | `share/device.h:95-105`, `hip/device.h:149-171, 185-195`; `ZFPBatch.cpp:252-253, 489-490` | (a) fork: `hipMallocAsync/hipFreeAsync` (ROCm ≥ 5.3); (b) ARCTO: grow-only **device** payload workspace (or a user device output), open the bitstream on that 8-B-aligned device pointer (zero-copy → no internal alloc/D2H), then one D2D/D2H copy behind the 12-B header; never point zfp at `d_out + header_bytes` (misaligned 64-bit atomics) | Per-call hipMalloc (≈ compressed max, up to 1.06× input) + hipFree (device sync) + sync pageable D2H; `benchmark_zfp_single.cu:86-92` reports 1.3–1.7× from pinned alone | M–L | S (a) / M (b) |
| ZFP-H3 <span class="chip todo">open</span> | ARCTO | Pinned + async transfers on a user stream once H2(b) owns the copy | pageable sync memcpy today (`hip/device.h:123,138,273`) | M | S–M |
| ZFP-H4 <span class="chip todo">open</span> | `hip/variable.h:367-412`, `hip/device.h:197-217` | Chunk = all blocks (offset array is 8 B/block) → one scan + one cooperative launch; `threads_per_sm = 2048`; LDS cap 64 KB (more tiles, fewer `grid.sync()`); persistent `d_offset`/cub temp | 2 M blocks → ~19 chunks on an MI250X GCD / ~7 on MI300X, each 3–5 launches + occupancy query + cooperative launch; 2 hipMalloc/hipFree per call | M for variable rate | S |
| ZFP-H5 <span class="chip todo">open</span> | `zfp.c:59-96`, `ZFPBatch.cpp:306, 504-511` | With D3(b) the host prefix sum and both index memcpys disappear; drop the duplicate header parse | µs-level | L | S |
| ZFP-H6 <span class="chip todo">open</span> | `share/device.h:83-87` | Accept `hipMemoryTypeManaged` as device (as CUDA does) | avoids silent staging of managed buffers | L | S |
| ZFP-H7 <span class="chip todo">open</span> | `zfp_exec_params_hip` (`zfp.h:92-95`), all launches/memsets/memcpys in `hip/` | Add `hipStream_t` to the exec params (ARCTO is the fork's only consumer) | enables overlap; removes NULL-stream serialisation | M in pipelines | M |
| ZFP-H8 <span class="chip todo">open</span> | `ZFPBatch.cpp:44-55, 523-525` | Fail fast on `ndims = 4`; require trailer `type == zfp_index_offset` | robustness (hybrid path is broken upstream) | — | S |

Suggested order: T1, H1, D4, H2(a), H4 (host-side, no bitstream risk, big wins on small/medium
fields) → C1/D1, C5, C3, C2, C6/D5 (small kernel changes) → C4, C8, D2, C9, D3, H2(b)/H3/H7 → C7
only with data from the above. Build with `-DZFP_WITH_HIP_PROFILE=ON` on the zfp target to print
kernel-only GB/s (`hip/shared.h:4-5`, `encode3.h:136-157`), which separates kernel time from the
host overheads; generate ≥ 256³–512³ cubes for throughput work and keep 64³ to quantify fixed
overheads. Reversible (lossless) ZFP is out of scope here; canonical CPU-only paths are
`third_party/zfp/src/template/rev*.c`.

### 7.7 Verify on hardware

VGPR/AGPR/scratch and occupancy of `encode3/decode3` for float and double per arch; per-call
rocprof API trace for 64³ and 512³ in all modes (how much is `device_init`, hipMalloc/hipFree,
memset, pageable memcpy vs kernels — quantifies H1/H2/D4 before kernel work); L2 request
amplification (`TCC_REQ` vs `FETCH/WRITE_SIZE`) to size C9/D2; `__launch_bounds__(128,{2,3,4})` and
WG size sweeps; cooperative-launch availability/cost on the deployed ROCm for gfx1100;
`hipMallocAsync` support and measured `hipFree`/`hipHostMalloc` cost; non-integer rates (atomics on
every slot edge) vs integer rates; unaligned `dwordx4` cost when `nx % 4 ≠ 0`; wave32 vs wave64 on
gfx1100 (zfp is wave-size agnostic); fixed-accuracy share of the 2120-bit-slot memset; whether
`perm_3[i]` constant-folds after unroll.

---

## 8. AMD architecture facts, AMD guidance, and literature

Every figure below carries a source; items that could not be opened are marked *unverified*.
Full citation keys are in the source list at the end of this section.

### 8.1 The numbers that drive the proposals

| Fact | gfx90a — MI210 / MI250(X), CDNA2 | gfx942 — MI300X, CDNA3 | gfx1100 — RX 7900 XT/XTX, RDNA3 |
|---|---|---|---|
| Wavefront | 64 [ISA200][HF] | 64 [HF] | 32 native; wave64 executes as 2 × wave32 [RDNA-WP][HIP-HW] |
| CUs | MI210 104; MI250 2 × 104; MI250X 2 × 110 [SPECS] | 304 = 8 XCDs × 38 CUs [CDNA3-WP p.7-8] | XTX 96 / XT 84 CUs; 2 CUs per WGP [RX7900XT][HIP-HW] |
| VGPR file per lane per SIMD | 512 unified (256 arch + 256 AGPR), allocated in groups of 8 [ISA200 §3.6.4][OLCF-MEM] | 512, "256 of each type … flexible" [ISA300 §3.6.4] | 1536 per SIMD (wave32); wave64 counts double [OCC][RDNA-WP p.20] |
| Max waves per SIMD | 8 [REGP][RCP-PIPE] | 8 (working assumption; not spelled out in the CDNA3 WP — confirm with `-Rpass-analysis`) | 16 [OCC] |
| **VGPR → waves/SIMD (CDNA)** | ≤ 64 → 8, ≤ 72 → 7, ≤ 80 → 6, ≤ 96 → 5, ≤ 128 → 4, ≤ 168 → 3, ≤ 256 → 2 [OLCF-RP p.4][REGP][ROBEY p.22] | same (512-lane file, granule 8) | wave32: ⌊1536 / VGPRs⌋ rounded to granule, max 16 [OCC] |
| SGPRs per wave | 104 [HF] | 104 [HF] | 106 [HF] |
| LDS | 64 KB/CU, 32 banks × 4 B, ≤ 64 KB per workgroup, wave64 LDS op serviced "over four cycles in waterfall" [ISA200 §11.1] | identical [ISA300 §11.1][CDNA3-WP p.8] | 128 KB per WGP, ≤ 64 KB per workgroup, 32 banks per 64 KB half [RDNA-WP p.18][CC-RDNA3]; CU mode splits it per CU pair [CC-RDNA3] |
| LDS bank-conflict cost | "as little as two cycles, or … as many as 64 cycles" [ISA200/300 §11.3.1] | same | "increase latency" [RDNA-PG] |
| Vector L1 | 16 KB/CU, 64-B lines (L2 lines 128 B) [OLCF-MEM][SPECS] | 32 KB/CU, 128-B lines [CDNA3-WP p.8] | L0 32 KB/CU, L1 256 KB per shader array [CC-RDNA3] |
| L2 | 8 MB per GCD, 32 channels, 128 B/clk per slice, 128-B lines [CDNA2-WP][RCP-L2] | 4 MB per XCD, 16 channels × 256 KB, 128-B reads / 64-B half-line writes, write-back, coherent within an XCD [CDNA3-WP p.9] | 6 MB [SPECS] |
| Infinity Cache (MALL) | none [MALL] | 256 MB memory-side, 128 channels, 17.2 TB/s peak (≈ 11.9 measured, ~218 ns) [CDNA3-WP p.9-10][CC-MI300X]; early spill from ~128 MB working set [MALL] | 96 MB (XTX) / 80 MB (XT) [RX7900XTX][RX7900XT] |
| DRAM | 1.6 TB/s per GCD [MI210] | 5.3 TB/s HBM3 [MI300X] | 960 / 800 GB/s GDDR6 [RX7900XTX][RX7900XT] |
| Widest per-lane load | `GLOBAL_LOAD_DWORDX4` (16 B) [ISA200 §9] | same [ISA300] | `GLOBAL_LOAD_B128` (*unverified*: RDNA3 ISA PDF not retrieved) |
| Non-temporal lowering | `nontemporal` → `glc=1 slc=1` ("streaming" L2 mode) [ISA200 §9][LLVM-MM] | → `nt=1` (CU "Miss Evict", L2 "Hit Stream", LLC "Hit Evict") [ISA300 §9.1.10.2][LLVM-MM] | → `slc=1 dlc=1` [LLVM-MM]; open LLVM bug for float/vector types [LLVM-BUG] |
| Cross-lane | `ds_bpermute/ds_permute` route through the LDS crossbar without writing LDS; DPP row ops at full VALU rate (2 wait states after a VALU write of the source); `ds_swizzle` needs no extra VGPR [XLANE] | same | same instruction families (*ISA-level detail unverified*) |
| Workgroup dispatch | one L2 per GCD | round-robin across the 8 XCDs [MI300-SYSOPT]; XCD-aware `blockIdx` swizzle raised L2 hit 43 % → 92 % [XCD-SWZ] | — |
| CU vs WGP mode | — | — | `-mcumode`; HIP converts `__launch_bounds__` 2nd arg as `(MIN_BLOCKS×MAX_THREADS)/(warpSize×2)` in CU mode, `/(warpSize×4)` in WGP mode [HIP-PORT][LLVM-USAGE] |

Derived: a wave64 load of 4 B/lane touches 256 B = two 128-B L2 lines (MI300X / MI2xx) or four 64-B L1
lines (MI2xx); a wave64 `dwordx4` load touches 1 KiB = eight 128-B lines — HIP: "the hardware can
combine all 64 thread requests into as few as 4-8 cache line requests" [HIP-HW]. On MI300X the L2
write path is half-line (64 B), so partially written 128-B lines are not penalised at the channel
level [CDNA3-WP p.9].

### 8.2 AMD's guidance, mapped to ARCTO

1. **Coalescing for 64 lanes** — consecutive lanes → consecutive addresses; stride kills effective bandwidth [HIP-PG][HIP-PERF]. → LZ4/Snappy literal and match copies should be lane-contiguous (LZ4-D1, SNP-D6/D8); Cascaded/zfp streaming loads lane-contiguous (CAS-S5, ZFP-C1/D1).
2. **16-byte vector access** — "Ensure `global_load_dwordx4` is used in the ISA … LDS load and store should use `_b128`" [MI300-TUNE]; naturally aligned `float4` [HIP-PG]. → CAS-S5, CAS-C2 (`ds_read_b128`), ZFP-C1/D1, LZ4-D3, SNP-D6.
3. **LDS bank conflicts** — pad by one element / swizzle; conflicts serialise by an integer factor; a wave64 LDS op is serviced per 16-lane group over four cycles [HIP-PG][HIP-PERF][ISA200/300]. → CAS-C2 (stride 8 dwords today), SNP-D5 (symbol queue), ZFP-C9/D2 (W+1 padding), LZ4 claim table (already conflict-free).
4. **Divergence / EXEC** — serialised branches, EXEC toggling [HIP-PG][HIP-PERF]. → keep token-parse rare paths out of hot loops; wave-uniform decisions via ballot + `readfirstlane` (SNP-D3/D4, LZ4-C9/D5).
5. **`__launch_bounds__` second argument = min waves per EU** [HIP-CPP][HIP-PORT]; `amdgpu_waves_per_eu` is a hint, default flat work-group size 1..1024 [LLVM-USAGE]; `amdgpu_num_vgpr` deprecated [CLANG-ATTR]; lab notes: occupancy 4 → 2 yet faster when spills vanished [LAP3]; "256 or 512 are generally the options to try for high scratch kernels" [OLCF-RP]. → CAS-S2, SNP-C4/D9, LZ4-D7, ZFP-C6, the existing LZ4 64-VGPR pin; fix the `amd_optimizations.h:98` comment.
6. **Register-pressure hygiene** — launch bounds, define near first use, avoid stack arrays and large kernel args, control unrolling, `__restrict__`, manual LDS spilling; check `-Rpass-analysis` / `.vgpr_spill_count` [REGP][ISA-BLOG][ROBEY]. → zfp's three `[64]` private arrays (ZFP-C4/C7), Cascaded `size_t` everywhere (CAS-S6), `ARCTO_KRESTRICT` already measured.
7. **Wave primitives on wave64** — 64-bit `__ballot`/`__activemask`; 32-bit mask shifts are a porting bug; `warpSize` not a compile-time constant before ROCm 7.0; shuffles = `ds_bpermute` (needs `s_waitcnt`), DPP at full rate; `__builtin_amdgcn_readfirstlane` for SGPR broadcasts [HIP-CPP][HIP-PORT][XLANE]. → SNP-D2/D3/D4, LZ4-C9/D5, the `main` mask-truncation defect (§4.2).
8. **hipCUB/rocPRIM over hand-rolled** — logical warps ≤ 64 must be 64-aware or a scan silently uses 32 lanes [HIPCUB][ROCPRIM-SCAN]. → CAS-S4, CAS-D2, SNP-D2.
9. **Non-temporal stores** — `__builtin_nontemporal_store` bypasses L2 for write-once data (1.19× on MI250X from one line) [LAP3]; verify `nt`/`slc` bits in the ISA [LLVM-BUG]. → LZ4-C7, CAS-S8; never on data that is re-read (decompressed output).
10. **L2/LLC-aware geometry** — re-indexing blocks to fit L2 raised hit 33 → 66 % on MI250X [LAP4]; per-XCD L2 on MI300X with round-robin dispatch; keep the per-XCD working set < 4 MB; MALL spills early (~128 MB) [MI300-SYSOPT][XCD-SWZ][MALL]. → candidate `blockIdx` swizzle (`xcd = wgid % 8`) for chunk batches on gfx942 (HLIF persistent grid and the batched grids) — listed as an experiment, not a first step.
11. **Atomics** — integer atomics fine on all three; avoid one global 64-bit atomic per small unit on one L2 channel; aggregate per wave/block [HF][CDNA2-WP]. → HLIF-3, ZFP-C2/C3.
12. **Occupancy is not the goal** — "Peak occupancy does not always mean peak performance"; issue several independent loads per lane (Little's law) [OCC][HIP-PERF]. → LZ4-D3 prefetch, SNP-C1, ZFP-D2.
13. **Workgroup size** — multiples of 64; 256 fills the 4 SIMDs [RDNA-PG][HIP-PG][NOD]. → CAS-S1 (128 → 256 on wave64).
14. **Compile-time checks** — `-Rpass-analysis=kernel-resource-usage`, `--save-temps`, `-mcumode`, `-mwavefrontsize64`, `--offload-arch` [REGP][ISA-BLOG][LLVM-USAGE]. → BLD-2/4/6.

### 8.3 Literature — what transfers under the "algorithm fixed" rule

| Work | Transferable (kernel-level) | Avoid (changes algorithm/format) |
|---|---|---|
| nvCOMP docs / GTC S21597 (Cascaded, LZ4) [NVC] | all three codecs are scan-dominated → hipCUB with 64-lane logical warps, re-tune items-per-thread so 256 threads cover the same tile | — |
| **Künas et al., ICCSA 2026** — "Characterizing Lossless GPU Data Compression Across AMD CDNA and RDNA Architectures" [KUNAS26] | baseline for ARCTO: MI300X up to 11× MI50 decompression; RDNA3 competitive for irregular-access compression; transfers dominate end-to-end → host staging + per-arch tuning | — |
| Sitaridi et al., Gompresso (ICPP 2016) [GOMP] | warp-cooperative match copy with intra-warp round-based dependency resolution (64-lane rounds on AMD) → LZ4-D1/D2, SNP-D8 | high-water-mark encoding |
| Ozsoy & Swany, CULZSS / pipelined LZSS [CULZSS] | stream-overlapped H2D/compute/D2H (+46 %) → HOST-3; LDS-resident window | LZSS token format |
| Zhang et al., GPULZ (ICS 2023) [GPULZ] | two-pass prefix sum + kernel fusion to cut LDS↔global traffic; arch-specific chunk size to fill 64 KB LDS → CAS tile fusion, LZ4 chunk sizing | multi-byte-symbol matching |
| Park et al., CODAG (arXiv 2307.03760) [CODAG] | **warp-per-chunk without dedicated prefetch warp**, warp-level sync instead of block-level, cache-line-granularity coalesced reads → the strongest argument for eventually re-shaping Snappy's 3-wave design around one wave64 per chunk (only after SNP-D1…D9 are measured) | — |
| Shavidze, "What Actually Serializes GPU LZ77 Decode" (arXiv 2608.10188, 2026) [SHAV] | parse, not copy, holds 64–72 % of decode time; 1-B-per-thread writes against 128-B lines have 4.4 % bus efficiency, coalesced 39× faster → prioritise wave-wide speculative parse + lane-contiguous writes (SNP-D3/D4/D7, LZ4-D1) | encoder-side offset/distance-history changes |
| Azami, Fallin, Burtscher, LC framework (ASPLOS 2025) [LC] | 16 KB chunks so two buffers fit in LDS, 512-B sub-chunks per warp, whole RLE/zero-elimination pipeline in shared memory between stages, shuffle-based exchange → CAS pipeline already LDS-resident; confirms per-wave sub-chunking and `ITEMS_PER_THREAD` scans (CAS-D2) | LC's own transforms |
| Knorr et al., ndzip-gpu (SC21) [NDZIP] | warp-cooperative **vertical (transposed) bit packing**: one lane per bit position, `__ballot` builds a 64-bit plane word — wave64 yields 64-bit words natively → CAS BitPack and zfp bit-plane emission (ZFP-C4/C7) | — |
| Shanbhag et al., Tile-based integer compression (SIGMOD 2022) [SHAN] | load a tile into LDS once, apply all decode stages in LDS, 2.2–2.6× vs nvCOMP decompression → validates CAS fused pipeline; only the pass structure transfers | their tile format |
| Afroozeh et al., FastLanes-GPU (DaMoN 2024); G-ALP (DaMoN 2025) [FL][GALP] | 1024-value vectors too large per warp (register pressure) → mini-vectors, wider blocks, one-value-per-thread → choose items-per-thread so a wave64 keeps ≤ 64 live VGPRs (CAS-D2/D6) | interleaved layouts |
| Huang et al., cuSZp2 (SC24) [CUSZP2] | single-kernel decoupled-lookback offsets instead of a separate scan kernel; on AMD beware 64-bit flag atomics hot-spotting one L2 channel → HLIF-3 | quantisation/encoding |
| Zhang et al., FZ-GPU (HPDC 2023) [FZ] | warp-level bitshuffle without conflicts, LDS fusion → ZFP-C4, CAS BitPack | lossy pipeline |
| Rivera et al. (IPDPS 2022); Weißenberger & Schmidt (ICPP 2018); Yamamoto (ICPP 2020) [HUFF] | online LDS sizing, divergence reduction (technique only — ARCTO has no Huffman stage) | gap arrays / self-sync |
| DietGPU (Meta) [DIET] | warp-per-4 KiB segments; saturation rule (segments ≈ resident warps) → batch sizing on 304-CU MI300X | ANS |
| zfp docs / releases [ZFPD] | one thread per 4^d block is register-heavy; 64-bit word streams match wave64 ballots → ZFP-C4/C7/C8 | variable-rate index formats outside the path |
| MGARD-X [MGARD] | per-backend tuned kernel templates (HIP/CUDA/SYCL) as a pattern for per-arch constants | — |
| Lurati et al., "Bringing Auto-tuning to HIP" (Euro-Par 2024) [AUTOTUNE] | auto-tuning impact ~10× on AMD vs ~2× on NVIDIA; CUDA-tuned configs are not optimal on AMD; small blocks + spatial tiling reduce registers on MI250X → sweep block size / items-per-thread / unroll per arch | — |
| Choudhary et al., XCD-aware swizzle (arXiv 2511.02132); SwizzlePerf [XCD-SWZ] | `xcd = wgid % 8; local = wgid / 8` remap → L2 hit 43 → 92 % on MI300X → experiment on HLIF/batched grids | — |
| Funasaka et al. (CCPE 2017) [FUNA]; Morfiadakis 2018 / Wesley 2022 [NEG] | negative results: cross-block synchronised LZ77 decode loses to independent chunks → keep one wave per chunk | GPU-designed formats |
| Micro-architecture characterisations: MALL is Open (SC'25 W); CDNA3 SC25; Ambati & Diep; chips&cheese [MALL][CC-*] | context for cache-level decisions (MALL early spill, per-XCD L2) | — |

### 8.4 Sources

[HF] rocm.docs.amd.com/projects/HIP/en/latest/reference/hardware_features.html ·
[HIP-HW] …/HIP/en/latest/understand/hardware_implementation.html ·
[HIP-PG] …/HIP/en/latest/how-to/performance_guidelines.html ·
[HIP-PERF] …/HIP/en/docs-7.2.0/understand/performance_optimization.html ·
[HIP-CPP] …/HIP/en/latest/reference/cpp_language_extensions.html ·
[HIP-PORT] …/HIP/en/latest/how-to/hip_porting_guide.html ·
[SPECS] rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html ·
[CDNA3-WP] amd.com/content/dam/amd/en/documents/instinct-tech-docs/white-papers/amd-cdna-3-white-paper.pdf ·
[CDNA2-WP] amd.com/content/dam/amd/en/documents/instinct-business-docs/white-papers/amd-cdna2-white-paper.pdf ·
[ISA300] amd.com/…/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf ·
[ISA200] amd.com/…/instruction-set-architectures/instinct-mi200-cdna2-instruction-set-architecture.pdf ·
[RDNA-WP] gpuopen.com/download/RDNA_Architecture_public.pdf ·
[RDNA3-ISA] amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf (*not retrieved*) ·
[RDNA-PG] gpuopen.com/learn/rdna-performance-guide/ · [OCC] gpuopen.com/learn/occupancy-explained/ ·
[XLANE] gpuopen.com/learn/amd-gcn-assembly-cross-lane-operations/ ·
[REGP] gpuopen.com/learn/amd-lab-notes/amd-lab-notes-register-pressure-readme/ ·
[LAP1–4] gpuopen.com/learn/amd-lab-notes/amd-lab-notes-finite-difference-docs-laplacian_part1/ (…part4) ·
[OLCF-RP] olcf.ornl.gov/wp-content/uploads/Intro_Register_pressure_ORNL_20220812_2083.pdf ·
[OLCF-MEM] olcf.ornl.gov/wp-content/uploads/03-MemoryHierarchy.pdf ·
[ROBEY] ccs.tsukuba.ac.jp/wp-content/uploads/sites/14/2025/09/10.-Advanced-HIP.pdf ·
[LLVM-USAGE]/[LLVM-MM] llvm.org/docs/AMDGPUUsage.html · [CLANG-ATTR] clang.llvm.org/docs/AttributeReference.html ·
[LLVM-BUG] github.com/llvm/llvm-project/issues/128408 ·
[RCP-L1]/[RCP-L2]/[RCP-LDS]/[RCP-PIPE] rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/cdna/{vector-l1-cache,l2-cache,local-data-share,pipeline-descriptions}.html ·
[MI300-TUNE] rocm.docs.amd.com/en/latest/how-to/rocm-for-ai/inference-optimization/workload.html ·
[ISA-BLOG] rocm.blogs.amd.com/software-tools-optimization/amdgcn-isa/README.html ·
[HIPCUB] rocm.docs.amd.com/projects/hipCUB/en/latest/ · [ROCPRIM-SCAN] rocm.docs.amd.com/projects/rocPRIM/en/latest/warp_ops/scan.html ·
[MI300X]/[MI210]/[MI250]/[RX7900XTX]/[RX7900XT] amd.com product pages ·
[CC-CDNA3]/[CC-MI300X]/[CC-RDNA3] chipsandcheese.com (secondary) ·
[MALL] dl.acm.org/doi/10.1145/3731599.3767487 · [MI300-SYSOPT] instinct.docs.amd.com/projects/amdgpu-docs/en/latest/system-optimization/mi300x.html (*not fetched*) ·
[XCD-SWZ] arxiv.org/abs/2511.02132 · [NOD] github.com/nod-ai/amd-shark-ai/blob/main/docs/amdgpu_kernel_optimization_guide.md (secondary) ·
[NVC] docs.nvidia.com/cuda/nvcomp/cascaded.html, …/lz4.html; GTC 2020 S21597 ·
[KUNAS26] link.springer.com/chapter/10.1007/978-3-032-30491-9_23 · [GOMP] arxiv.org/abs/1606.00519 ·
[CULZSS] web.cs.hacettepe.edu.tr/~aozsoy/papers/2011-ppac.pdf · [GPULZ] arxiv.org/abs/2304.07342 ·
[CODAG] arxiv.org/abs/2307.03760 · [SHAV] arxiv.org/abs/2608.10188, arxiv.org/abs/2606.18900 ·
[LC] userweb.cs.txstate.edu/~burtscher/papers/asplos25.pdf, github.com/burtscher/LC-framework ·
[NDZIP] dl.acm.org/doi/10.1145/3458817.3476224 · [SHAN] anilshanbhag.com/static/papers/gpufor_sigmod22.pdf ·
[FL] ir.cwi.nl/pub/34260/34260.pdf · [GALP] ir.cwi.nl/pub/35205/35205.pdf · [CUSZP2] dl.acm.org/doi/10.1109/SC41406.2024.00021 ·
[FZ] arxiv.org/abs/2304.12557 · [HUFF] arxiv.org/abs/2201.09118, dl.acm.org/doi/10.1145/3225058.3225076 ·
[DIET] github.com/facebookresearch/dietgpu · [ZFPD] zfp.readthedocs.io/en/latest/execution.html, github.com/LLNL/zfp/releases ·
[MGARD] github.com/CODARcode/MGARD/blob/master/doc/MGARD-X.md · [AUTOTUNE] arxiv.org/abs/2407.11488 ·
[FUNA] onlinelibrary.wiley.com/doi/abs/10.1002/cpe.4283 · [NEG] hgpu.org/?p=18542.

Research caveats: ACM/Springer full texts and several rocm.blogs.amd.com pages were not fetched
(abstracts/search listings only); the RDNA3 ISA PDF download returned an HTML stub; CDNA3's max
waves per SIMD and RDNA3 VGPR granularity should be confirmed on hardware (§8.1).

---

## 9. Roadmap and branch plan

### 9.1 Step zero (before any kernel change)

`BLD-2` — turn on `-Rpass-analysis=kernel-resource-usage` (+ `--save-temps`) for one build per
arch and record VGPR/SGPR/LDS/spills/occupancy for every kernel in §1.2. Half of the items below
are conditional on those numbers (every launch-bound item, LZ4-D7, SNP-D9, CAS-S2, ZFP-C6). Also
`BLD-1` (explicit `-O3 -DNDEBUG` for HIP) so no measurement is ever taken with device asserts on.

### 9.2 Start here — highest expected return per unit of effort

| # | Item | Codec | Effort | Expected | Why first |
|---|---|---|---|---|---|
| 1 | SNP-D1 real `s_sleep` in the three spin-waits | Snappy decomp | S | tens of % at realistic occupancy | pure scheduling; the AMD `clock()` fallback is a porting accident |
| 2 | CAS-C1 single-pass min/max in `get_for_bitwidth` | Cascaded comp | S | 20–40 % with `use_bp` | removes 62 of 64 collectives+barriers per u8 array |
| 3 | CAS-S1 256-thread blocks on wave64 (+ CAS-S2 launch bounds) | Cascaded both | S | 15–30 % / 10–25 % | doubles resident waves at the same LDS |
| 4 | CAS-D5 / CAS-S3 LDS shrink (alias `temp_count_array`) | Cascaded decomp | S–M | 10–30 % u8/u16 | 2 → 3 blocks per CU |
| 5 | LZ4-D7 decompress launch bounds, then LZ4-D2 modulo removal | LZ4 decomp wave64 | S / S–M | enabler / H on repetitive data | the untouched front on CDNA; the earlier vectorisation losses may be a VGPR-step artefact |
| 6 | ZFP-T1 gate, then ZFP-H1 + ZFP-D4 + ZFP-H2(a) | ZFP host | S each | large on small/medium fields | removes 2–3 `device_init` pinned allocations, one hipMalloc/memset/D2H/hipFree per call |
| 7 | SNP-D3 + SNP-D4 `readfirstlane` and all-lane uniform masks | Snappy decomp | S–M | 5–15 % | moves the decoder's broadcast chain off the LDS crossbar; frees VGPRs |
| 8 | LZ4-C5 widen `lengthOfMatch` | LZ4 comp | M | H on compressible data | cuts the dependent L2 round trips 4–16× |
| 9 | CAS-D1 balanced RLE expansion | Cascaded decomp | M | 10–50 % RLE-heavy | the data Cascaded is chosen for |
| 10 | SNP-D2 hipCUB DPP warp scan/reduce | Snappy decomp | M | 10–25 % short-symbol data | 6–12 dependent `bpermute` → DPP |
| 11 | ZFP-C1/D1 vectorised gather/scatter, ZFP-C5 drop 64-bit div/mod, ZFP-C3 aligned slots / no memset | ZFP kernels | S–M | M each | first bit-exact kernel wins |
| 12 | ZFP-C8 ctz run-length coder / ZFP-C4 SWAR transpose | ZFP kernels | M | potentially the largest zfp kernel win | after the above, gated by T1 + plane test |
| 13 | HLIF-1 16-B aligned chunk slots + dword scratch copy; HLIF-4 cap `max_ctas` | all HLIF | S | M for fast codecs; memory H | second full pass over compressed bytes today |

### 9.3 Branch plan (one branch per aggregation, off `main`; one commit per change)

| Branch | Contents (in order) | Gate |
|---|---|---|
| `opt/build-reports-2026-08` | BLD-1, BLD-2, BLD-7 | configure + build on all three archs; no perf claim |
| `opt/snappy-decomp-2026-08` | SNP-D1 → D3 → D4 → D5 → D2 → D7 → D9 → D6 → D8 → D11 (knobs) → D12 (experiment) → D10 | `test_snappy_app`, `test_snappy_batch_c_api`, `SnappyLargeTokens_test`; bytes identical; `benchmark_snappy_chunked` on 3 inputs × `-p 32768/65536` × `-x` |
| `opt/snappy-comp-2026-08` | SNP-C4 → C1 → C2 → C5 → C3 | same tests; byte-exact compressed output |
| `opt/cascaded-2026-08` | CAS-S2 → S1 → C1 → D5/S3 → D1 → D2 → S4 → S5 → C2/C3 → C4 → C5 → S6/S7/S8 (→ D3 if S3 lands) | `test_cascaded_batch` (+ new cases for `num_RLEs=0/>2`, `num_deltas=0`, non-default chunk), `test_cascaded`, C API; bytes identical |
| `opt/lz4-decomp-wave64-2026-08` | LZ4-D7 → D2 → D1 (re-test under D7) → D3 → D5 → D4 → D6; LZ4-C12 hardening | `test_lz4`, `test_random_lz4`, C API; round-trip bit-exact; `benchmark_lz4_chunked` zeros/TTI/random at saturation on gfx90a/gfx942, wave32 regression check on gfx1100 |
| `opt/lz4-comp-2026-08b` (continues the July/August lineage, off `opt/curated` or after its merge) | LZ4-C2 → C3 → C5 → C7 → C9 → C6 | bytes identical (C2/C3/C5/C9), ratio in exact bytes otherwise; `EXPERIMENT-LOG` entries E18+ |
| `opt/zfp-host-2026-08` (ARCTO side) | ZFP-T1 → H1 → D4 → H2(a) → H4 → H8 → H2(b)/H3 → H7 (fork ABI) | `test_zfp_*` + T1; `benchmark_zfp_single` 64³ and 512³ all modes |
| fork `cristianokunas/zfp` branch `amd-hip`: `opt/zfp-kernels-2026-08` | C1/D1 → C5 → C3 → C2 → C6/D5 → C4 → C8 → D2 → C9 → D3; C7 last | T1 byte-exact; `ZFP_WITH_HIP_PROFILE` kernel-only GB/s; move the submodule pin in ARCTO per kept commit |
| `opt/hlif-2026-08` | HLIF-1 → HLIF-4 → HLIF-2 → HLIF-3 → HLIF-5/6/7 → TMP-1 → HOST-1/2/3 → BENCH-1 | HLIF tests (`test_lz4`, `test_cascaded`, `test_snappy_app`); HLIF benchmark path |
| `opt/integration-2026-09` | stacked combination of the kept commits from the branches above (the "opt1+2" branch) | full matrix |

### 9.4 Measurement protocol per commit

Build wave64 (gfx90a/gfx942/gfx906 where available) and wave32 (gfx1100) from the same commit;
`ctest` green on both; compressed bytes identical unless the commit says otherwise (then report the
deviation in exact bytes); round trip bit-exact; throughput with the chunked benchmarks on at least
three inputs (highly compressible / structured TTI / random) at `-p 65536` and `-p 32768`, in the
undersubscribed regime and at saturation (`-x`), ≥ 10 repetitions with raw per-repetition values
(`ARCTO_PER_REP_CSV`); record hypothesis → prediction → verdict in the experiment log; revert with
an explicit commit when a knob does not pay. On the CUDA backend: build and run the same tests
(correctness only) so the portable build never regresses by construction; the nvCOMP comparison
numbers come from the CUDA-backend benchmark harness, which links nvCOMP.

---

## 10. Profiling recipe

- **Compile-time:** `-Rpass-analysis=kernel-resource-usage` (VGPR/SGPR/AGPR/LDS/occupancy/spills per
  kernel per `--offload-arch`), `--save-temps` (ISA: look for `global_load_dwordx4`, `ds_bpermute`,
  `v_readlane`, `s_barrier`, `v_rcp`/udiv expansions, `buffer/global_store_byte`). Read
  `.vgpr_count`, `.sgpr_count`, `.agpr_count`, `.group_segment_fixed_size`, `.private_segment_fixed_size`
  in the code-object metadata.
- **rocprof / rocprofv3 counters by question:** occupancy → `SQ_WAVES`, `SQ_BUSY_CYCLES`,
  `GRBM_GUI_ACTIVE`, derived `MeanOccupancyPerCU`; dependent-latency chains → `SQ_WAIT_INST_ANY`,
  `SQ_WAVE_CYCLES`; instruction mix → `SQ_INSTS_VALU`, `SQ_INSTS_SALU`, `SQ_INSTS_LDS`,
  `SQ_INSTS_VMEM_RD/WR`, `SQ_ACTIVE_INST_VMEM/LDS/VALU`; LDS → `SQ_LDS_BANK_CONFLICT`,
  `SQ_LDS_IDX_ACTIVE`, `SQ_LDS_DATA_FIFO_FULL`, `SQ_WAIT_INST_LDS`; L1/L2 → `TCP_TOTAL_CACHE_ACCESSES`,
  `TCC_HIT`, `TCC_MISS`, `TCC_REQ`, `TCC_EA_RDREQ`, `TCC_EA_WRREQ`, `TCC_EA_ATOMIC`, `FETCH_SIZE`,
  `WRITE_SIZE`; launch overhead → `rocprofv3 --kernel-trace` / `--hip-trace` gaps.
- **ROCm Compute Profiler (rocprof-compute, formerly omniperf)** for the roofline and the per-kernel
  panels once a candidate is chosen — `rocprof-compute profile -n run -- ./bench` then
  `rocprof-compute analyze -p workloads/run/<arch>/ --block 2.1 7.x 12.x 16.x 17.x`; its derived
  metrics map directly onto the questions here: *Wavefront occupancy* (waves/SIMD, limited by
  VGPR/LDS/workgroup), *LDS Bank Conflict Rate* (% of active LDS cycles spent on conflicts; target → 0
  for CAS-C2/SNP-D5/ZFP-C9), *vL1D Coalescing Efficiency* (25 % uncoalesced … 100 %; check copy
  kernels first), *L1/L2 Hit Rate*, *L2-Fabric read/write bandwidth*, *VALU Utilization* and branch
  efficiency [RCP-L1][RCP-L2][RCP-LDS][RCP-PIPE][RCP-CLI]. AMD's own lab notes recommend it over raw
  rocprof. **rocprof-sys (Omnitrace)** for timelines with host API calls (zfp's `device_init`,
  hipMalloc/hipFree, pageable memcpys). `rocprofv3 --list-avail`; `rocprofv3 --kernel-trace --stats
  --output-format csv -- ./bench`; `rocprofv3 --pmc SQ_WAVES GRBM_GUI_ACTIVE -- ./bench` (one
  `--pmc` group per pass). ATT traces (rocprof-compute-viewer) for `vmcnt`/`lgkmcnt` stall chains on gfx942.
- **Expected profiles:** LZ4 compress = dependent-chain (high `SQ_WAIT_INST_ANY`, LDS/VALU for the
  claim table); LZ4/Snappy decompress = VMEM-write-instruction bound on low-ratio data, VALU (`urem`)
  bound on RLE data, plus Snappy's LDS polling share; Cascaded = barrier/LDS-latency bound at 1–3
  waves/SIMD; zfp = VALU/latency bound with 1–2 waves/SIMD and L2 request amplification on the
  stream words.
- **Benchmarks:** `benchmark_<codec>_chunked -f <files> -p 65536 -w 2 -i 10 -c true` with `-x N`
  for saturation; `ARCTO_PER_REP_CSV`/`ARCTO_PER_REP_TAG` (on the `feat/per-rep-output` lineage) for
  raw per-repetition values; `benchmark_zfp_single -f cube.bin -3 NX,NY,NZ -m <mode> -r P -i 10 -w 2 [-c] [-D]`.

---

## Appendix A — Open questions for the maintainer

1. Is a multi-arch fat binary (gfx90a + gfx1100) a supported configuration, or is one build dir per
   wave size the intended mode? (decides BLD-6)
2. Were the `AMD_LAUNCH_BOUNDS_*` values chosen with HIP's waves-per-EU meaning or CUDA's blocks-per-SM?
3. Were published numbers produced with this CMake/clang configuration (implicit `-O3`) or an earlier hipcc build?
4. Is the blocking `hipMalloc` on AMD in `ManagerBase.hpp` (CUDART_VERSION gate) intentional?
5. Is the adaptive window meant to be re-chosen from the online rates (HOST-1)?
6. `include/arcto.h:106-186` declares a deprecated generic API that `src/arcto_api.cpp` never defines — remove or implement?
7. Three test CMakeLists still carry an NVIDIA EULA header and are absent from `NOTICES.md`.
8. Benchmark `-w` short flag collides with `--file_with_page_sizes`; `benchmarks/README.md` is stale (`build/benchmarks/`, `scripts/benchmark.sh`, missing `-P/-R/-A`).
9. `hip::hipcub` found but not linked — intentional?
10. `BatchManager` size estimate uses `uint32` per chunk while the layout uses `size_t` — known?
11. Where does the CI matrix (ROCm 6.1 / 7.x × gfx90a/gfx942/gfx1100) live?

## Appendix B — Test-coverage gaps to close before the corresponding kernel work

- LZ4: output-identity checks are manual (compressed size/bytes diff); add a test that hashes the
  compressed stream for the fixture files per wave size.
- Cascaded: no cases for `num_RLEs = 0`, `num_deltas = 0`, `num_RLEs > 2`, non-default `chunk_size`.
- ZFP: tolerance-based only → ZFP-T1.
- Snappy: no tests for odd chunk sizes, odd input/output alignments or `BATCH_COUNT` wrap-around;
  add a randomized compressibility sweep cross-checked with python-snappy (`snappy/snappy_example.py`)
  before SNP-C3/D6/D7/D8.
- HLIF: the wave-size guard exists only on the batched entry points (`opt/curated`); add it to the
  manager path.

## Appendix C — Review provenance

Sections 2–7 condense five static reviews (build/infra, LZ4, Snappy, Cascaded, ZFP) and one
source-gathering pass (§8) carried out on 2026-08-21/22 against `main` @ `e3e1a1f` with the
`opt/*` branches read for prior art. The full per-codec reviews (kernel inventories with per-kernel
launch/LDS/primitive details, and the longer rationale for each item) live next to this file:
`docs/reviews/lz4.md`, `docs/reviews/snappy.md`, `docs/reviews/cascaded.md`, `docs/reviews/zfp.md`,
`docs/reviews/build_and_shared_infra.md`, and the sourced AMD reference `docs/reviews/amd_sources.md`.
This map keeps the file:line anchors so each item can be re-derived from the source.
