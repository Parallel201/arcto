# ARCTO LZ4 codec — static review for AMD-GPU kernel optimization

Scope: `src/LZ4Kernels.hiph`, `src/LZ4Types.h`, `src/lowlevel/LZ4Batch.cpp`, `src/lowlevel/LZ4CompressionKernels.{h,hip}`, `src/highlevel/LZ4HlifKernels.{h,hip}`, `src/highlevel/LZ4Manager.{hpp,cpp}`, `include/arcto/lz4.{h,hpp}`, `src/arcto_common_deps/hlif_shared.hiph`, `src/arcto_hipcub.hiph`, `src/device_functions.hiph`, `src/HipUtils.h`, `src/amd_optimizations.h`, `src/arcto_device_types.h`, `include/cuda_shim/hip/*`, `CMakeLists.txt` (flag block), benchmarks and tests (skimmed). Static review only; no build, no run, no repo edits. Hardware figures (cache sizes, VGPR budgets) are from public AMD documentation and should be treated as approximate until verified on the target boxes.

Hard constraint respected throughout: every proposal below keeps the LZ77 hash-table + sliding-window matcher, the hash function, the hash-table size policy, the token/LSIC/offset encoding and the byte-exact LZ4 block output. Only execution/memory mechanics change. Where a proposal *would* alter compressed bytes it is flagged explicitly and marked out-of-scope.

---

## 1. Code map

### 1.1 Low-level (batched C API) path

| Layer | Symbol | Location |
|---|---|---|
| C API | `arctoBatchedLZ4CompressGetTempSize / CompressGetMaxOutputChunkSize / CompressAsync / DecompressGetTempSize / DecompressAsync / GetDecompressSizeAsync` | `include/arcto/lz4.h:106-243`, impl `src/lowlevel/LZ4Batch.cpp:71-224` |
| Host wrapper | `lz4BatchCompress` (dispatches on `arctoType_t`), `lz4BatchDecompress`, `lz4BatchGetDecompressSizes`, `lz4GetHashTableSize`, `lz4BatchCompressComputeTempSize`, `lz4DecompressComputeTempSize`, `lz4ComputeMaxSize` | `src/lowlevel/LZ4CompressionKernels.hip:142-318` (decl. `LZ4CompressionKernels.h:77-135`) |
| Kernels | `lz4CompressBatchKernel<T>` (T in {uint8_t, uint16_t, uint32_t}) | `LZ4CompressionKernels.hip:73-94` |
| | `lz4DecompressBatchKernel` (also used with `output_decompressed=false` to compute sizes) | `LZ4CompressionKernels.hip:96-136` |
| Device core | `compressStream<T>` | `src/LZ4Kernels.hiph:793-969` |
| | `decompressStream` + `BufferControl` | `src/LZ4Kernels.hiph:971-1097`, `353-518` |

Call chain (compress): `arctoBatchedLZ4CompressAsync` (`LZ4Batch.cpp:189`) -> `lz4BatchCompress` (`.hip:158`) -> `lz4CompressBatchKernel<T><<<batch_size, warpsize>>>` (`.hip:182-216`) -> `compressStream<T>` (one wave per chunk).
Call chain (decompress): `arctoBatchedLZ4DecompressAsync` (`LZ4Batch.cpp:89`) -> `lz4BatchDecompress` (`.hip:224`) -> `lz4DecompressBatchKernel<<<ceil(batch/2), (warpsize,2)>>>` (`.hip:236-247`) -> `decompressStream` (one wave per chunk, two chunks per block).

### 1.2 High-level interface (HLIF / manager) path

| Layer | Symbol | Location |
|---|---|---|
| Public | `LZ4Manager(uncomp_chunk_size, data_type, stream, device_id)` | `include/arcto/lz4.hpp:62-67`, `src/highlevel/LZ4Manager.hpp:161-174` |
| Manager | `LZ4BatchManager : BatchManager<LZ4FormatSpecHeader>` — `do_batch_compress/do_batch_decompress`, `compute_*_max_block_occupancy`, `compute_scratch_buffer_size`, `format_specific_init` | `LZ4Manager.hpp:67-157` |
| Host wrappers | `lz4HlifBatchCompress`, `lz4HlifBatchDecompress`, `batchedLZ4CompMaxBlockOccupancy`, `batchedLZ4DecompMaxBlockOccupancy` | `LZ4HlifKernels.hip:157-279` |
| Wrappers (device) | `lz4_compress_wrapper<T>` (hash table = `tmp_buffer + blockIdx.x*hash_table_size`), `lz4_decompress_wrapper` (LDS slice per `threadIdx.y`) | `LZ4HlifKernels.hip:76-121`, `123-155` |
| Generic kernels | `HlifCompressBatchKernel<CompressT,CompressorArg,chunks_per_block=1>`, `HlifDecompressBatchKernel<DecompressT,chunks_per_block>` (persistent CTAs, atomic work-stealing on `ix_chunk`) | `hlif_shared.hiph:229-252`, `419-447`, bodies `160-227`, `285-340` |

Grid for HLIF = `max_ctas` = `multiProcessorCount * hipOccupancyMaxActiveBlocksPerMultiprocessor(...)` (`LZ4HlifKernels.hip:222-279`); each CTA loops over chunks (`hlif_shared.hiph:185-226`, `317-339`). Compression writes each chunk into a per-CTA scratch slot, then `atomicAdd`s the output offset and copies into the final buffer (`copyScratchBuffer`, `hlif_shared.hiph:127-158`).

### 1.3 Data formats and memory layouts

* Types: `position_type = uint32_t`, `offset_type = uint16_t` (`LZ4Types.h:60-63`). `word_type = uint32_t`, `double_word_type = uint64_t` (`LZ4Kernels.hiph:111-112`).
* Hash table: per chunk (low-level) / per CTA (HLIF), `hash_table_size` entries of `offset_type` (2 B). Size policy: `min(roundUpPow2(max_chunk_size), MAX_HASH_TABLE_SIZE=1<<14)` (`LZ4CompressionKernels.hip:142-156`, `LZ4Kernels.hiph:151`). For the default 64 KB chunk: 16 384 entries = 32 KB. Entries store `pos & MAX_OFFSET` (16-bit wrapped position, `LZ4Kernels.hiph:736`); `convertIdx` (`619-632`) reconstructs the absolute position. `NULL_OFFSET = 0xFFFF` marks empty (`157`). Hash: `(__brev(key) + (key ^ 0xc375)) & (size-1)` (`557-561`).
* Temp space (low-level compress): `batch_size * HT_size * sizeof(offset_type)`, laid out `temp_space + bidx*hash_table_size` (`LZ4CompressionKernels.hip:91, 173-174, 287-296`). Decompress temp: `sizeof(chunk_header)*num_chunks` (`298-304`) but `lz4BatchDecompress` ignores `temp_ptr` (`229-230`) — legacy/unused.
* HLIF scratch: `max_comp_ctas * (hash_table_size*2 + max_comp_chunk_size)` (`LZ4Manager.hpp:147-151`); hash tables live *after* the per-CTA output slots (`hlif_shared.hiph:239-240`).
* Chunk output bound: `maxSizeOfStream(size) = roundUpTo(size + 1 + ceil(size/255), 8)` (`LZ4Kernels.hiph:198-202`); `MAX_CHUNK_SIZE = 16 MB` (`174`).
* Decompression LDS window: `DECOMP_INPUT_BUFFER_SIZE = LZ4_DECOMP_THREADS_PER_CHUNK * 8` bytes per chunk (512 B on wave64, 256 B on wave32), refilled when fewer than half remain (`137-145`, `988-991`).
* "Data type" specialization (`ARCTO_TYPE_CHAR/UCHAR/BITS -> uint8_t`, `SHORT/USHORT -> uint16_t`, `INT/UINT -> uint32_t`; `LZ4CompressionKernels.hip:185-219`, `LZ4HlifKernels.hip:167-189`): compression only. `T` sets the granularity at which the wave scans input: each lane loads one `T` (`LZ4Kernels.hiph:849-851`), words are assembled by `shuffleLiterals<T>` (`743-791`), `invalid_threads = 3/sizeof(T)` tail lanes cannot form a full 4-byte word (`861`), so a wave processes `warpsize-3` (uint8), `warpsize-1` (uint16) or `warpsize` (uint32) positions per round (`863-865`). Positions, offsets and match lengths are scaled back to bytes before encoding (`946-947, 954`), so the output is plain LZ4 regardless of `T`; the decompressor is type-agnostic. Default opts use CHAR (`include/arcto/lz4.h:84`; benchmark `benchmark_lz4_chunked.cu:38-40`).

---

## 2. Kernel inventory

Wave size: `warpsize` = 64 when `__HIP_PLATFORM_AMD__ && ENABLE_HIP_OPT_WARPSIZE64 && USE_WARPSIZE_64` (`LZ4Kernels.hiph:87-94`; CMake adds `ENABLE_HIP_OPT_WARPSIZE64` unless `USE_WARPSIZE_32`, `CMakeLists.txt:128-132`), else 32. Numbers below are given as wave64 / wave32.

| Kernel | File:line | Launch config | LDS | Wave primitives | Global access pattern | Regs / occupancy | Notes |
|---|---|---|---|---|---|---|---|
| `lz4CompressBatchKernel<T>` | `LZ4CompressionKernels.hip:73-94` | grid = `batch_size`, block = `(LZ4_COMP_THREADS_PER_CHUNK)` = 64/32 (`.hip:182-183`). No `__launch_bounds__`. One wave = one chunk. | Static: `warpMatchAny<uint32_t>::values[warpsize]` = 256 B / 128 B (`LZ4Kernels.hiph:227`, one instantiation since both call sites pass `uint32_t`). No dynamic LDS. | `__ballot` (`warpBallot`, 213-216, 877, 911, 609), `__shfl` broadcast (`SHFL1`, 891, 921), `__shfl_down` x2 (uint8) / x1 (uint16) (`760-762, 779`), emulated match-any via LDS loop (`218-245`), `__brevll/__clzll` on 64-bit masks. No hipcub, no cooperative groups. `SYNCWARP1()` is a no-op on AMD (`82-83`). | Input `next`: 1 x T per lane, coalesced 64 B/128 B/256 B (`849-851`). Hash gather: 64 random 2-B loads into a 32 KB table (`643`); `isValidHash` verifies with `readWord<word_type>` = `4/sizeof(T)` separate loads at a random position (`656`, `256-265`). Match extension: 2 coalesced loads per lane per 64 positions (`604-608`). Hash scatter: random 2-B stores (`736`). Hash clear: 256 iterations of 2-B strided stores for 16K entries (`815-818`). Literal copy: byte-granular coalesced (`582-584`). Token/offset: thread 0 byte stores (`677-680, 703-705`). | Unknown (needs `-Rpass-analysis=kernel-resource-usage`). 64-bit masks (`lane_mask_t`) cost 2 VGPRs each. Single-wave workgroups: occupancy on CDNA bounded by VGPRs (<=64 VGPR for 8 waves/SIMD) and by waves/CU (32). | Hot loop is a serial chain per round: load next -> match-any (LDS loop) -> ballot -> hash gather (L2/MALL) -> isValidHash data load -> ballot -> insert -> lengthOfMatch (dependent L2 trips per 64 positions) -> writeSequenceData. `#ifdef NVCOMP_22` (`699-706`) is not defined in CMake, so offset is written by thread 0 only. |
| `lz4DecompressBatchKernel` | `LZ4CompressionKernels.hip:96-136` | grid = `ceil(batch_size/2)`, block = `(warpsize, 2)` = 128/64 threads (`.hip:236-237`, `258-259`). No `__launch_bounds__`. | Static: `buffer[DECOMP_INPUT_BUFFER_SIZE*2]` = 1024 B / 512 B (`.hip:108`); per-wave slice `threadIdx.y*512` (`.hip:127`). | No shuffles/ballots in the default build (`WARP_READ_LSIC` is off, `365-403`). `syncCTA()` = no-op on AMD (`204-211`). Token parsing executed redundantly by all lanes. | Window refill: 1 x 8-B load per lane, coalesced 512 B (`484-488`), byte-wise fallback in the last <512 B (`490-496`). Literals: byte-granular copy LDS/global -> global (`520-528`, `1015-1024`). Match copy: byte-granular with `source[i % dist]` runtime modulo (`530-542`); source is global output (`1076-1080`) or LDS window (`1066-1070`). Offset read: 2 byte loads (`1040-1042`). | Unknown. LDS is negligible; occupancy will be VGPR-bound. | Two waves per block never synchronize (no `__syncthreads`), so `CHUNKS_PER_BLOCK` is only a launch-granularity choice. `OOB_CHECKING=1` adds two scalar compares per sequence (`1007-1012, 1053-1058`). |
| `HlifCompressBatchKernel<lz4_compress_wrapper<T>, LZ4CompressorArgs, 1>` | `hlif_shared.hiph:229-252`, LZ4 launch `LZ4HlifKernels.hip:164-186` | grid = `max_ctas` (= CUs x occupancy, `.hip:222-262`), block = 64/32, dyn LDS 0. No `__launch_bounds__`. Persistent: loops over chunks via `atomicAdd(ix_chunk)`. | Static: `ix_chunks[1]` (4 B), `output_status[1]` (4 B) + `values[warpsize]` (256/128 B) from compressStream. `extern __shared__ share_buffer[]` unused (0 B). | Same as compress kernel, plus `cg::this_thread_block().sync()` (= `__syncthreads`, single-wave block) x3 per chunk (`179, 205, 225`), `atomicAdd` u64 on `ix_output` and u32 on `ix_chunk` per chunk (`200-203, 223`). | Same as compress kernel. Scratch copy-out: `char4` loads + 4 byte stores per lane (`copyScratchBuffer`, `138-148`) when the scratch slot is 4-B aligned, else byte loop. Per-CTA hash table re-used across chunks (good L2 locality). | Unknown. | Scratch footprint = `max_ctas * (32 KB + ~66 KB)`; on a 304-CU part with 32 single-wave blocks/CU this is ~0.9 GB (see P8). |
| `HlifDecompressBatchKernel<lz4_decompress_wrapper, 2>` | `hlif_shared.hiph:419-447`, LZ4 launch `LZ4HlifKernels.hip:206-217` | grid = `max_ctas`, block = `(warpsize, 2)`, dyn LDS = 1024 B / 512 B (`.hip:208`). No `__launch_bounds__`. Persistent with `atomicAdd(ix_chunk)` per chunk. | Dynamic 1024/512 B (window per wave) + static `ix_chunks[2]`, `output_status[2]`. | `cg::tiled_partition<arcto::warpsize>(cta)` (`hlif_shared.hiph:383`) and `.sync()` per chunk (`314, 338`); `atomicAdd` u32 per chunk. | Same as decompress kernel. | Unknown. | `arcto::warpsize` comes from `arcto_device_types.h:43-44` (64 on AMD unless `USE_WARPSIZE_32`; 32 on CUDA), consistent with the global `warpsize` in `LZ4Kernels.hiph:91/99`. |

Build flags relevant to all four: Release = `-O3` (CMake default `-O3 -DNDEBUG` for HIP/CXX; explicit `-O3` at `CMakeLists.txt:152`), `-fPIC`, no `-ffast-math`, no `-mcumode`, no `-Rpass-analysis`, no `HIP_ENABLE_WARP_SYNC_BUILTINS`. CUDA backend defines `__HIP_PLATFORM_NVCC__/__HIP_PLATFORM_NVIDIA__` and compiles the same sources through `include/cuda_shim/hip/hip_runtime.h` (pure `#define` aliasing to CUDA runtime) — device intrinsics are therefore the *native CUDA* ones.

---

## 3. NVIDIA-isms and AMD pitfalls found

P1. **Warp-sync macros collapse to no-ops on AMD** — `LZ4Kernels.hiph:71-85`: the `__CUDACC_VER_MAJOR__ >= 9` branch picks `__shfl_sync/__ballot_sync/__syncwarp`; on HIP-Clang/AMD that macro is undefined, so `SYNCWARP1()`/`SYNCWARP(m)` expand to nothing and `syncCTA()` (`204-211`) is empty. Correctness currently relies on wave lockstep plus the compiler not reordering LDS/global accesses across the (absent) barrier (`warpMatchAny` 230-240, `BufferControl::loadAt` 499, `compressStream` 820, `insertHashTableWarp` 740, `decompressStream` 1072/1075). Works in practice (LDS ops from one wave are issued in order; same-array may-alias prevents reordering) but is fragile. `amd_optimizations.h:127-134` already provides `wave_barrier()` (`__builtin_amdgcn_wave_barrier`) for exactly this; LZ4 does not use it.

P2. **Emulated `__match_any_sync` via an O(warpsize) LDS loop** — `LZ4Kernels.hiph:218-245`: `__CUDA_ARCH__ >= 700` is never true on AMD, so each call does one LDS write + 64 (wave64) LDS reads + compares. It is called **twice per compression round** (`872`, `733`). On wave64 that is ~128 LDS reads + ~250 VALU ops per 61 input bytes — likely the single largest VALU/LDS cost in the compressor. No hardware equivalent on AMD, but a ballot/`readfirstlane` loop (iterations = #distinct values, data-dependent) or HIP's software `__match_any_sync` (ROCm >= 6.2 with `-DHIP_ENABLE_WARP_SYNC_BUILTINS`, to be verified) avoids LDS entirely.

P3. **Lane-mask truncation to `int` in the wave64 port (latent correctness/determinism defect, not just perf)** —
   * `numValidThreadsToMask` (`LZ4Kernels.hiph:717-720`) computes a 64-bit `LANE_MASK_FULL >> (64-n)` but **returns `int`**. For `n >= 32` (the normal case: 61 for uint8, 63 for uint16) the low 32 bits are all ones, the `int` is `-1`, and it sign-extends to an all-64-lanes `participants` mask in `warpMatchAny(const signed_lane_mask_t, T)`. Lanes `>= numValidThreads` never enter `warpMatchAny` (guards at `731`, `870`) so their `values[]` slots are **never written for uint8 (slots 61..63) / uint16 (slot 63)** and contain whatever LDS held before the workgroup started; the fallback loop (`233-239`) nevertheless compares against them. In the `next`-match use (`870-893`) a spurious high bit is harmless (a lane's own bit is always lower, so `CLZ(BREV(mask))` is unaffected). In `insertHashTableWarp` (`734`) the rule "highest matching lane inserts" can pick a non-participating lane, so **no lane inserts** for that slot — a missed hash insert; output stays valid LZ4 but compressed size becomes non-deterministic/slightly worse.
   * `const int match = warpMatchAny(...)` (`LZ4Kernels.hiph:732`) truncates the 64-bit match mask to 32 bits before `CLZ(match)` (which is `__clzll` after integer promotion). Duplicates spanning lanes 0-31 and 32-63 lose the "last match inserts" semantics (several lanes or none insert); if lane 31 is in the mask the promoted value is negative and `63 - CLZ = 63`, so nobody inserts.
   Both are nvCOMP `int`s that were correct for 32 lanes. Fix = widen to `signed_lane_mask_t`/`lane_mask_t` (3 lines). This *restores* the intended algorithm on wave64; it does not change it. Not an issue on the CUDA backend or under `USE_WARPSIZE_32`.

P4. **Runtime integer modulo in the decompression hot loop** — `coopCopyRepeat` (`LZ4Kernels.hiph:539-541`): `dest[i] = source[i % dist]` per byte per lane. AMD has no integer divider; a 32-bit `urem` by a runtime divisor is ~20-40 VALU instructions. This runs for every overlapped match (RLE-like data, zeros) — the cheapest data for LZ4 becomes the most VALU-expensive for the AMD decompressor.

P5. **Byte-granular copies everywhere** — literals (`575-585`, `520-528`), match copies (`530-542`), `readWord/writeWord` assembled from byte loads/stores (`247-265`), hash-table clear with 2-B stores (`815-818`). Each store instruction moves only 64 B per wave; CDNA/RDNA memory pipelines are happiest at dword/dwordx4 per lane. AMD global memory supports unaligned dword/dwordx4 loads and stores (unaligned-access mode is on by default in ROCm), so `memcpy`-style 4-/16-B accesses lower to single instructions on AMD while staying correct (byte-split) on NVIDIA.

P6. **Two `ds_bpermute` shuffles per round just to assemble a 4-byte key (uint8 path)** — `shuffleLiterals<uint8_t>` (`754-764`): the key for lane `t` is bytes `t..t+3`, which is exactly an unaligned dword load at `decompData + decomp_idx + t`. On AMD `__shfl_down` is an LDS-crossbar op (`ds_bpermute_b32`, tens of cycles + `lgkmcnt` wait); an unaligned `global_load_dword` is cheaper and removes a dependency on lanes `t+1..t+3`. Same for uint16 (one shuffle, `779`).

P7. **Wave-uniform broadcasts done with `__shfl`** — `SHFL1(match_location, first_match_thread)` (`891`, `921`): `first_match_thread` is derived from a ballot and is wave-uniform; on AMD this can be `v_readlane_b32` into an SGPR (`__builtin_amdgcn_readlane`) instead of a bpermute. Micro.

P8. **Occupancy-API-driven HLIF grid amplifies scratch memory on AMD** — `batchedLZ4CompMaxBlockOccupancy` (`LZ4HlifKernels.hip:222-262`) x `multiProcessorCount`, used for `max_comp_ctas` and hence `compute_scratch_buffer_size = max_ctas * (32 KB + max_comp_chunk_size)` (`LZ4Manager.hpp:147-151`). With 64-thread blocks the AMD occupancy answer can be up to 32 blocks/CU; on a 304-CU MI300X that is ~9.7k CTAs and ~0.9 GB of scratch for a 64 KB chunk size (MI250X GCD ~0.35 GB, RX 7900 XT ~0.26 GB). Not an NVIDIA-ism per se (the formula is generic) but the magnitude is AMD-specific. The test file already warns about it (`tests/test_lz4.cpp:253-254`).

P9. **Per-target wave size is a build-wide macro** — `USE_WARPSIZE_32` / `ENABLE_HIP_OPT_WARPSIZE64` (`CMakeLists.txt:128-132`, `LZ4Kernels.hiph:65-69, 87-103`, `arcto_device_types.h:28-44`) means one binary cannot correctly target both gfx90a/gfx942 (wave64) and gfx1100 (wave32 default) — the 64-lane assumptions (`DECOMP_INPUT_BUFFER_SIZE`, `numValidThreads`, `lane_mask_t`) would be wrong on the wave32 device. Per-target selection (`__AMDGCN_WAVEFRONT_SIZE__` or `__builtin_amdgcn_wavefrontsize()`, depending on ROCm version) would fix this. Out of scope for perf but worth noting for a multi-arch build.

P10. **Uncapped per-wave serial latency means small batches under-fill CDNA** — grid = one wave per chunk (`LZ4CompressionKernels.hip:182-183`). A 100 MB input at 64 KB chunks is 1600 waves; MI300X has 304 CUs x 4 SIMDs = 1216 SIMDs, so ~1.3 waves/SIMD: essentially no latency hiding; throughput scales with batch size until ~9.7k chunks. This is a measurement pitfall more than a code bug (see Section 5 "how to measure": use `-x` duplicate_data).

P11. **`hipOccupancyMaxActiveBlocksPerMultiprocessor` for the decompressor assumes dynamic LDS only** (`LZ4HlifKernels.hip:270-275`) — fine today (static LDS is tiny) but if D4 enlarges the window via a static array the call must stay consistent.

Not found (already fine): no 48 KB shared-memory assumptions; no hard-coded SM counts; no `__syncwarp` inside divergent code; no hipcub in LZ4; cooperative groups limited to `this_thread_block()` and `tiled_partition<warpsize>` (both supported by HIP); all `/`,`%` in `writeLSIC`/`token_type`/`convertIdx` are by compile-time constants and fold to mul/shift; LDS layouts are conflict-free (sequential dword writes, rotated reads `(t+d)&63`, broadcast token reads).

---

## 4. Existing AMD optimizations already applied (do not redo)

* Wave-64 typing of all lane masks: `lane_mask_t/signed_lane_mask_t = uint64_t/int64_t`, `BREV=__brevll`, `CLZ=__clzll`, `warpsize=64` (`LZ4Kernels.hiph:87-103`); `warp_mask_t` mirror in `arcto_device_types.h:35-44`; 64-bit `__ballot` handled by `BALLOT1` (`84`).
* Per-chunk thread counts follow the wave: `LZ4_COMP_THREADS_PER_CHUNK = LZ4_DECOMP_THREADS_PER_CHUNK = warpsize` (`118-123`), so wave64 processes 61/63/64 positions per compression round and copies 64 B per iteration in the decompressor; `DECOMP_INPUT_BUFFER_SIZE` scales with it (`137-138`).
* Invalid-tail lane count is type-, not wave-, dependent (`invalid_threads = 3/sizeof(T)`, `861`), and `numValidThreads` is clamped at chunk end (`863-865`) — correct for 64 lanes.
* HIP `__shfl/__shfl_down/__ballot` default `width=warpSize` used (no hard-coded 32 widths).
* HLIF grid sized from device CU count x measured occupancy (`LZ4HlifKernels.hip:222-279`) rather than SM constants.
* `hash()` keys are 32-bit regardless of wave size (comment at `560`).
* `amd_optimizations.h` exists (wave helpers, `wave_barrier`, `AMD_LAUNCH_BOUNDS_*`, reductions/scans) but **none of it is referenced by the LZ4 sources**; the AMD adaptations in LZ4 are the local macros above.
* Build: `-O3` Release; `ENABLE_HIP_OPT_WARPSIZE64` default, `USE_WARPSIZE_32` for gfx1100.

---

## 5. Optimization opportunities

Legend: Impact = qualitative (H/M/L) on the wall-time of the respective kernel for typical mixed data; CUDA risk = risk of breaking/slowing the experimental CUDA backend; Effort S/M/L.

Measurement for all items: `benchmark_lz4_chunked -f <file(s)> -p 65536 -w 2 -i 10 [-x N] [-c true]` (flags at `benchmark_template_chunked.cuh:1496-1547`; it times compress and decompress separately with events at `610-660`, `705-731`, and verifies round-trip on the last iteration, `755-774`). Use `-x 4..8` on MI300X/MI250X so the batch has >= ~10k chunks (see P10). Correctness: `test_lz4` (`[small]`, `[large]`), `test_random_lz4`, `test_lz4batch_c_api` (tags/cases at `tests/test_lz4.cpp:164-280`, `tests/test_random_lz4.cpp:182-206`, `tests/test_lz4batch_c_api.c:50-56`). For output-identity claims additionally diff compressed sizes/bytes before vs. after (the benchmark prints compressed size; `-c true` gives it as a column).

### 5.1 COMPRESSION (`compressStream<T>`, `lz4CompressBatchKernel`, `HlifCompressBatchKernel<lz4_compress_wrapper>`)

**C1 — Replace the LDS match-any emulation (and fix the mask truncation)** — `LZ4Kernels.hiph:218-245`, `717-720`, `732`.
What: (a) widen `numValidThreadsToMask` return and `insertHashTableWarp::match` to `signed_lane_mask_t`; (b) on AMD implement `warpMatchAny` with a ballot loop: `remaining = participants; while (remaining) { v = readfirstlane(val of lowest remaining lane); m = ballot(val == v) & participants; if (val == v) mine = m; remaining &= ~m; }` (or `__match_any_sync` from HIP when `HIP_ENABLE_WARP_SYNC_BUILTINS` is available). Iteration count = number of distinct values among participants; each iteration is a handful of SALU/VALU ops with no LDS traffic and no LDS allocation.
Mechanism: removes ~128 `ds_read_b32` + waits per round and the 256 B static LDS; on CDNA the VALU and SALU pipes do the work and the loop exits early for repetitive data (few distinct `next` words). For incompressible data (61 distinct words, and hashPos values are nearly always distinct in a 16K table) the cost is similar to the unrolled LDS loop, so the win is data-dependent — measure both variants. (a) alone removes the non-determinism and restores the intended hash-insert policy.
Impact: M (VALU/LDS bound part of the round), plus determinism. CUDA risk: none (keep `__match_any_sync` under `__CUDA_ARCH__ >= 700`). Preserves algorithm: yes — same mask semantics, same decisions; (a) makes wave64 behave as the 32-lane original did. Effort: S (a), M (b).
Measure: compress throughput; rocprof `SQ_INSTS_LDS`, `SQ_INSTS_VALU`, `SQ_INSTS_SALU`; verify identical compressed bytes across repeated runs after (a).

**C2 — Load the 4-byte key directly with one unaligned dword load (drop `shuffleLiterals`)** — `LZ4Kernels.hiph:848-854`, `754-781`.
What: for `T=uint8_t/uint16_t`, when `decomp_idx + threadIdx.x + 4 <= length` (bytes) read `next` via `__builtin_memcpy(&next, bytes + decomp_idx*sizeof(T) + threadIdx.x*sizeof(T), 4)`; keep the existing byte/short path only for the last few lanes of the chunk. Lanes `>= numValidThreads` are don't-care exactly as today.
Mechanism: one `global_load_dword` (unaligned OK on gfx9+) instead of a byte load + 2 `ds_bpermute_b32` (+`s_waitcnt lgkmcnt`) per round; the 64 lanes read 67 contiguous bytes — two cache lines, fully coalesced. Removes a cross-lane dependency from the critical chain.
Impact: M (every round). CUDA risk: low — `memcpy` lowers to 4 byte loads on NVIDIA (about the same as today's 1 load + 2 shuffles). Preserves algorithm: yes — identical `next` values for all valid lanes (little-endian assembly both ways), hence identical hashes/matches. Effort: S.
Measure: compress throughput; `--save-temps` ISA diff (no `ds_bpermute` in the loop).

**C3 — Unaligned dword/short access in `readWord`/`writeWord`** — `LZ4Kernels.hiph:247-265`, used at `656` (hash verification, hot), `1040-1042` (decomp offset read), `700/704` (offset write).
What: implement with `__builtin_memcpy` into a `T` (or a `packed, aligned(1)` struct load/store).
Mechanism: `isValidHash` currently issues 4 byte loads per lane per round for uint8 (2 for uint16) at a random position; one unaligned `global_load_dword` halves the VMEM instruction count of the dependent chain and frees VGPRs. On AMD the compiler emits the single instruction because unaligned access mode is on; on NVIDIA memcpy stays byte-wise.
Impact: L-M (reduces instruction count on the most latency-critical dependent load). CUDA risk: none. Preserves algorithm: yes (same values). Effort: S.

**C4 — Vectorize the per-chunk hash-table clear** — `LZ4Kernels.hiph:815-818`.
What: clear with `uint4` (dwordx4) stores when the table base is 16-B aligned (true for `hipMalloc` + 32 KB/2^k strides; keep the 2-B loop as fallback). 16 384 entries x 2 B = 32 KB = 32 iterations of 1 KB per wave instead of 256 iterations of 128 B.
Mechanism: 8x fewer store instructions on the per-wave critical path; bandwidth identical (the clear writes 50 % of the input size in bytes — 32 KB per 64 KB chunk — which is the same with or without vectorization). Optional variant: one `hipMemsetAsync` of the whole temp region before the launch removes the clear from the wave entirely (DMA/fill engine runs at full bandwidth), at the cost of an extra stream op; for HLIF (per-CTA table reused across chunks) the in-kernel clear must stay.
Impact: L (fixed ~2-4 us per wave; matters more for small chunks, e.g. `-p 32768`). CUDA risk: none (aligned `uint4` stores are fine on NVIDIA). Preserves algorithm: yes (same `NULL_OFFSET` fill). Effort: S.

**C5 — Widen `lengthOfMatch` to 4 (or 16) bytes per lane per dependent trip** — `LZ4Kernels.hiph:592-617`.
What: per lane compare `4/sizeof(T)`-byte (or 16-byte) words at `prev+i` and `next+i` (unaligned `memcpy` loads), ballot "any byte differs", then resolve the exact first differing byte inside the winning lane with `__builtin_ctz(a ^ b)/8`; clamp the last partial lane against `length - min_ending_literals` exactly as the scalar loop does. Optional: software-pipeline one iteration ahead (issue loads for `j+64*W` before the ballot of `j`).
Mechanism: today each 64-position step is a *dependent* L2/MALL round trip (loads -> ballot -> break), so a 64 KB run of zeros costs ~1024 serial memory latencies (~hundreds of us per chunk). Widening to 4 B per lane cuts the number of dependent trips 4x (16x with dwordx4); the output is the same match length.
Impact: H for highly compressible data (long matches), neutral for incompressible. CUDA risk: low with `memcpy` (byte loads on NVIDIA — same load count as today, fewer ballots); `dwordx4` on NVIDIA via memcpy becomes 16 byte loads per lane — acceptable for an experimental backend, or keep the 4-B variant there. Preserves algorithm: yes — identical `num_matches`. Effort: M.
Measure: compress throughput on `-x`-duplicated zero/RLE data vs. random data; rocprof `SQ_WAIT_INST_ANY`, `SQ_INSTS_VMEM_RD`.

**C6 — Vectorized literal copy (compression side)** — `LZ4Kernels.hiph:575-585` (called at `691-692`).
What: 4-B (AMD: unaligned dword) per lane per iteration via `memcpy`, byte tail. Shared with D1.
Mechanism: 4x fewer load/store instructions for literal runs; only matters for long literal runs (incompressible regions).
Impact: L-M (data-dependent). CUDA risk: none (`memcpy`). Preserves algorithm: yes. Effort: S.

**C7 — Non-temporal stores for compressed output (AMD only)** — stores at `LZ4Kernels.hiph:677-680`, `686`, `691-692`, `700-705`, `711`.
What: on `__HIP_PLATFORM_AMD__` use `__builtin_nontemporal_store` for `compData` writes (guard: plain store / `__stcs` on CUDA). `compData` is never re-read by the compressor.
Mechanism: the compressor's cache-sensitive working set is the 32 KB hash table + 64 KB input window per resident wave; on CDNA2 (8 MB L2/GCD, no MALL) ~900 resident chunks already exceed L2 several times over, so random 2-B hash gathers go to HBM with ~32x byte amplification. Keeping the write-once output stream out of L2 (streaming/`nt` policy) leaves more L2 for tables and windows. CDNA3's 256 MB Infinity Cache softens this; RDNA3 (6 MB L2 + 80-96 MB IC) sits in between.
Impact: L-M on CDNA2, L elsewhere; must be measured. CUDA risk: none with guard. Preserves algorithm: yes (same bytes, only cache policy). Effort: S.
Measure: rocprof `TCC_HIT/TCC_MISS`, `TCC_EA_RDREQ`, `TCP_TOTAL_CACHE_ACCESSES`.

**C8 — Hash table in LDS (experiment; occupancy trade-off, same output)** — `LZ4Kernels.hiph:815-818, 643, 736`; LDS carve-out in `lz4CompressBatchKernel` / `lz4_compress_wrapper`.
What: for `hash_table_size <= 16384`, keep the table as a 32 KB `__shared__ uint16_t[]` per wave (`ds_read_u16/ds_write_b16`), cleared with `ds_write_b128`; identical size, hash, NULL fill, and insert policy.
Mechanism: random gather/scatter latency drops from L2/MALL/HBM (~0.5-2 us dependent) to LDS (~100-200 cycles), and all hash traffic disappears from L2. Cost: 32 KB LDS per wave -> only 2 waves/CU on gfx90a/gfx942 (64 KB LDS) and ~2 per CU-equivalent on gfx1100 (128 KB/WGP), i.e. occupancy collapses from up to 8 waves/SIMD to 0.5. Whether the shorter per-round chain beats the lost overlap is unknown and data-dependent; on CDNA2 (no MALL) it is most likely to pay off, on CDNA3 least likely.
Impact: unknown (could be H or negative). CUDA risk: M (32 KB static smem is fine on NVIDIA but occupancy drops there too; keep under a macro). Preserves algorithm: yes — table contents/size/policy identical, output identical. Effort: M.
Measure: A/B with identical compressed bytes; rocprof occupancy (`MeanOccupancyPerCU`) and `SQ_LDS_BANK_CONFLICT`.

**C9 — Wave-uniform broadcasts via `readlane`; `ffs` instead of `CLZ(BREV())`** — `LZ4Kernels.hiph:891, 921` and `878, 884, 887, 912, 611, 734`.
What: under AMD, `match_location = __builtin_amdgcn_readlane(match_location, first_match_thread)` (lane id is uniform — comes from a ballot); `CLZ(BREV(m))` -> `__ffsll(m)-1` (`s_ff1_i32_b64` when `m` is the SGPR ballot result).
Mechanism: avoids `ds_bpermute` (LDS crossbar + `lgkmcnt` wait) and two 64-bit bit-reverse/clz VALU sequences per use. Micro, but inside the per-round chain.
Impact: L. CUDA risk: none with `#ifdef`. Preserves algorithm: yes. Effort: S.

**C10 — Launch bounds / VGPR budget** — kernel signatures at `LZ4CompressionKernels.hip:73-74`, `hlif_shared.hiph:233/256`.
What: after reading `-Rpass-analysis=kernel-resource-usage`, consider `__launch_bounds__(64, N)` (HIP's second argument = min waves per EU) to pin VGPRs at <= 64 (8 waves/SIMD on CDNA) if the compiler is slightly above a threshold (e.g. 68-72 VGPRs) without spilling. Note `AMD_LAUNCH_BOUNDS_*` in `amd_optimizations.h:104-113` target 256-1024-thread blocks and are not suitable as-is for these 64-thread kernels.
Impact: L-M (only if currently just over a VGPR step). CUDA risk: none (`__launch_bounds__(64, N)` on nvcc means min blocks/SM, harmless). Preserves algorithm: yes. Effort: S.

**C11 — Cap HLIF `max_comp_ctas` to bound scratch and launch cost** — `LZ4Manager.hpp:147-151`, `LZ4HlifKernels.hip:222-262`.
What: `max_ctas = min(CUs * occupancy, CUs * 8)` (or a byte budget). Grid sizing only; the persistent loop already handles any grid.
Mechanism: avoids ~1 GB scratch allocations on MI300X and thousands of idle persistent CTAs for small inputs; 8 single-wave CTAs/CU already equals the max VGPR-bound occupancy per SIMD x 2.
Impact: memory footprint H, perf L (avoids over-subscription tail). CUDA risk: none. Preserves algorithm: yes. Effort: S.

**C12 — Robustness: real wave barrier on AMD** — `LZ4Kernels.hiph:82-83` (`SYNCWARP1/SYNCWARP`).
What: on AMD expand to `__builtin_amdgcn_wave_barrier()` (= `arcto::amd::wave_barrier()`, `amd_optimizations.h:127-134`). Zero runtime cost (compiler scheduling barrier), prevents future reordering bugs around the LDS match loop / window refill.
Impact: none (safety). CUDA risk: none. Preserves algorithm: yes. Effort: S.

**C13 — Compile flags to A/B** — `CMakeLists.txt:128-160`.
`-Rpass-analysis=kernel-resource-usage` (report VGPR/SGPR/LDS/occupancy per kernel); `--save-temps` (ISA inspection); `-DHIP_ENABLE_WARP_SYNC_BUILTINS` (enables `__match_any_sync`, `__syncwarp`, `__shfl_sync` on ROCm >= 6.2 — lets the `__CUDACC_VER_MAJOR__`/`__CUDA_ARCH__` branches at `LZ4Kernels.hiph:71-85, 221` be re-pointed on AMD); `-mcumode` on gfx1100 (CU vs WGP mode — sometimes better for many small independent waves); `-mwavefrontsize64` on gfx1100 (would let RDNA3 use the 61-positions-per-round path; requires the wave64 build and is risky with hipcub/warpSize assumptions — experiment only). Impact: unknown until measured. Effort: S.

Not recommended / out of scope: changing `MAX_HASH_TABLE_SIZE` or the hash-size policy (changes compressed bytes); changing `ARCTO_TYPE_*` defaults (a user choice; note that `ARCTO_TYPE_INT` on 4-B-aligned data quadruples positions per round but also changes matches, hence output).

### 5.2 DECOMPRESSION (`decompressStream`, `lz4DecompressBatchKernel`, `HlifDecompressBatchKernel<lz4_decompress_wrapper,2>`)

**D1 — Vectorize literal copies and non-overlapping match copies** — `LZ4Kernels.hiph:520-528` (used at `1015-1024`, `553`).
What: each lane moves 4 B (AMD: one unaligned `global_load_dword`/`global_store_dword`; 16 B when both pointers are 16-B aligned) per iteration via `memcpy`; byte head/tail. Source may be LDS (`ctrl.rawAt`) or global; for the LDS source either keep byte loads or use unaligned `ds_read` (supported gfx9+), both behind the same `memcpy`.
Mechanism: the decompressor's output side is store-issue/latency bound: 64 B per store instruction today, so a 64 KB chunk of literals needs ~1024 load+store pairs on the wave's critical path; 4x-16x fewer instructions and more bytes in flight per `s_waitcnt`. CDNA L2 write-combines partial lines either way, so bandwidth is not the limiter — instruction count and dependent latency are.
Impact: H for low-ratio data, M for mixed. CUDA risk: none with `memcpy` (byte-wise there). Preserves algorithm: yes (pure copy). Effort: S-M.
Measure: decompress throughput vs. compression ratio of the input; rocprof `SQ_INSTS_VMEM_WR`, `SQ_WAIT_INST_ANY`.

**D2 — Remove the per-byte modulo from overlapped match copies** — `LZ4Kernels.hiph:530-542` (called at `1066-1070`, `1076-1080`).
What: `r = threadIdx.x % dist` once per lane, `step = blockDim.x % dist` once per wave; per iteration `r += step; if (r >= dist) r -= dist;` (valid because `r, step < dist`). Fast paths that keep the same bytes: `dist == 1` (broadcast byte -> dword/dwordx4 stores), `dist >= 4` (4 consecutive output bytes per lane with one conditional wrap each), and when `dist >= length` the existing no-overlap path (D1 vectorized). All reads stay within `source[0..dist)` as today, so no intra-copy ordering hazard is introduced.
Mechanism: replaces ~20-40 VALU per byte-lane (software `urem`) by ~3; for RLE-heavy data (e.g. zero pages, where LZ4 emits one sequence with a 64 KB match at `dist=1`) this is the whole kernel.
Impact: H for repetitive data, L otherwise. CUDA risk: none. Preserves algorithm: yes (identical index sequence). Effort: S-M.

**D3 — Larger, dwordx4-refilled input window with prefetch** — `LZ4Kernels.hiph:137-145`, `458-500`, `988-991`; LDS declarations at `LZ4CompressionKernels.hip:108`, `LZ4HlifKernels.hip:131, 208`.
What: grow `DECOMP_INPUT_BUFFER_SIZE` to e.g. `warpsize * 16..32` B (1-2 KB per wave; 2-4 KB per block — still negligible LDS), align `setAndAlignOffset` to 16 B, refill with one `uint4` per lane per 1 KB, and optionally issue the next window's global load into registers *before* the current window is exhausted (write to LDS at refill time).
Mechanism: a refill is a dependent global round trip every 256 B of compressed input today (prefetch distance = half of 512 B); for a 64 KB incompressible chunk that is ~256 serialized latencies. A 2 KB window with half-window prefetch cuts that 4x and serves more literal copies from LDS (`1016-1023` falls back to global when the run crosses the window end). Keep `batchedLZ4DecompMaxBlockOccupancy`'s `shmem_size` in sync (`LZ4HlifKernels.hip:270`).
Impact: M for low-ratio data, L for high-ratio. CUDA risk: none (aligned `uint4` loads). Preserves algorithm: yes (window is a cache of the same bytes). Effort: M.

**D4 — Unaligned short read for the offset** — `LZ4Kernels.hiph:1037-1043` via C3 (`readWord<offset_type>`): one `global_load_ushort`/`ds_read_u16` instead of two byte loads on the per-sequence chain. Impact: L. Effort: S (shared with C3).

**D5 — Scalarize wave-uniform decode state (AMD)** — `LZ4Kernels.hiph:994-1050`.
What: after the token byte / LSIC bytes are read (per-lane VGPR from LDS), pass them through `__builtin_amdgcn_readfirstlane` so `comp_idx`, `decomp_idx`, `num_literals`, `offset`, `match` live in SGPRs; the compiler then uses SALU for the sequence bookkeeping and frees VGPRs.
Mechanism: lower VGPR pressure (-> occupancy) and SALU/VALU overlap; also lets the OOB checks (`1008, 1054`) and loop bounds become scalar branches (`s_cbranch_scc`) instead of `exec`-mask manipulation.
Impact: L-M (needs the resource-usage numbers first). CUDA risk: none with `#ifdef` (identity on NVIDIA). Preserves algorithm: yes. Effort: S-M.

**D6 — Chunks per block / launch shape** — `LZ4Kernels.hiph:128`, `LZ4CompressionKernels.hip:236-237`, `LZ4HlifKernels.hip:207`.
What: `LZ4_DECOMP_CHUNKS_PER_BLOCK` = 1, 2 (today) or 4; the two waves never synchronize, so this is only dispatch granularity. On AMD, single-wave workgroups do not consume barrier resources; 4 waves/block halves the number of workgroups the dispatcher must launch for large batches.
Impact: L. CUDA risk: none. Preserves algorithm: yes. Effort: S (constant). Measure with `-x 8`.

**D7 — Launch bounds / VGPR budget** — as C10 for the 128-thread decompress block (`__launch_bounds__(128, N)`), after `-Rpass-analysis`. Impact: L-M conditional. Effort: S.

**D8 — Wave barrier robustness** — same as C12 for `syncCTA()` (`204-211`) around `loadAt` and the LDS-sourced overlapped copy (`1066-1072`).

Not recommended: non-temporal stores for the *decompressed* output — the output is re-read by subsequent match copies (`1076-1080`) and must stay cache-resident.

---

## 6. Open questions to verify on hardware

1. **Resource usage / occupancy**: compile with `-Rpass-analysis=kernel-resource-usage` (or `--save-temps` and read `.s` metadata) for `lz4CompressBatchKernel<uint8_t/uint16_t/uint32_t>`, `lz4DecompressBatchKernel`, both HLIF instantiations, on gfx90a, gfx942, gfx1100 (wave32). Key thresholds: <= 64 VGPRs for 8 waves/SIMD on CDNA; SGPR count; scratch (spills) = 0; LDS 256 B / 1 KB as computed above. Decides C10/D7 and whether D5 is worth it.
2. **Where the compressor's time goes**: rocprof/rocprofv3 counters per kernel — `SQ_WAVES`, `SQ_BUSY_CYCLES`, `SQ_WAIT_INST_ANY`, `SQ_INSTS_VALU`, `SQ_INSTS_SALU`, `SQ_INSTS_LDS`, `SQ_INSTS_VMEM_RD/WR`, `SQ_ACTIVE_INST_VMEM/LDS/VALU`, `SQ_LDS_BANK_CONFLICT`, `TCP_TOTAL_CACHE_ACCESSES`, `TCC_HIT`, `TCC_MISS`, `TCC_EA_RDREQ`, `TCC_EA_WRREQ`, derived `MeanOccupancyPerCU`, `VALUUtilization`, `MemUnitStalled`. Expectation: compress = high `SQ_WAIT_INST_ANY` (dependent L2/MALL chain) + high LDS instruction share (match-any loop); decompress = VMEM-write instruction bound on low-ratio data, VALU bound (`urem`) on RLE data. This ranks C1/C5 vs C7/C8 and D1 vs D2.
3. **Hash-table cache behaviour by architecture**: `TCC_HIT/MISS` and `TCC_EA_RDREQ` for the compress kernel at full occupancy on MI250X (no MALL) vs MI300X (MALL) vs 7900 XT — determines whether C7 (nt stores) or the C8 LDS experiment is worth pursuing.
4. **`__match_any_sync` availability**: does the installed ROCm provide `__match_any_sync` under `-DHIP_ENABLE_WARP_SYNC_BUILTINS` (introduced around ROCm 6.2)? If yes, C1(b) can be a one-line re-point of the `__CUDA_ARCH__ >= 700` condition; compare its codegen with the hand-written ballot loop.
5. **Determinism check for P3**: run the benchmark/test several times on wave64 and compare compressed sizes per chunk before and after widening the masks; any run-to-run variation confirms the stale-LDS read path is being hit.
6. **Per-target wave size**: does the installed compiler still define `__AMDGCN_WAVEFRONT_SIZE__` (deprecated in newer ROCm) or require `__builtin_amdgcn_wavefrontsize()`? Needed only if a single fat binary for gfx90a/gfx942 + gfx1100 is a goal (P9).
7. **Unaligned access lowering**: confirm with `--save-temps` that `__builtin_memcpy(&w, p, 4)` on an `const uint8_t*` produces a single `global_load_dword` (global) / `ds_read_b32` (LDS) on each target and does not split into bytes (would negate C2/C3/C5/D1/D4). Also confirm the CUDA backend (nvcc) compiles the same `memcpy` forms.
8. **Batch-size sensitivity**: sweep `-x 0,1,3,7` at `-p 65536` and `-p 32768` on MI300X to see where throughput saturates; if the 1600-chunk default is far below saturation, P10 dominates every kernel micro-optimization in the default benchmark and results should be reported at saturation.
9. **gfx1100 mode flags**: A/B `-mcumode` vs default WGP mode, and (separately, risky) a wave64 build, for both kernels.
10. **NDEBUG**: confirm the Release HIP flags include `-DNDEBUG` so the many device `assert()`s (e.g. `LZ4Kernels.hiph:599, 802, 914-915, 931-937`) are compiled out; a stray assert-enabled build would distort every measurement.
