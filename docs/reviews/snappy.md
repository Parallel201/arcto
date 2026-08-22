# ARCTO Snappy codec — AMD GPU kernel-optimization review (static)

Scope: `src/snappy/*`, `src/lowlevel/SnappyBatch*.{cpp,h,hip}`, `src/highlevel/SnappyHlif*`, `src/highlevel/SnappyManager.*`, shared device helpers (`src/device_functions.hiph`, `src/arcto_device_types.h`, `src/arcto_common_deps/hlif_shared.hiph`), `src/amd_optimizations.h`, the `snappy/` tool directory, tests and the chunked benchmark. No build, no run, no file modified. Targets considered: gfx90a (CDNA2, wave64, 64 KB LDS/CU, 32 LDS banks, 4 SIMD/CU, max 8 waves/SIMD), gfx942 (CDNA3, same shape plus Infinity Cache), gfx1100 (RDNA3, wave32 build via `-DUSE_WARPSIZE_32`, 2 CU/WGP, 128 KB LDS/WGP), plus the experimental CUDA backend (`include/cuda_shim/hip/`).

Context from coordinator: on `origin/opt/curated` the only Snappy change is a `HipUtils::check_wave_size()` guard at the batched entry points in `SnappyBatch.cpp`; the Snappy kernels themselves are otherwise unoptimized for AMD. Everything below is therefore a first-pass list. The hard constraint is respected throughout: every proposal leaves the algorithm (12-bit hash LZ77, literal/copy tokens, Snappy raw format, nvcomp's warp-specialised decoder) and the produced bytes unchanged; only execution/memory mechanics change.

---

## 1. Code map

### 1.1 Entry points and call chains

**Batched (low-level) C API → host wrapper → kernels**

| C API (`src/lowlevel/SnappyBatch.cpp`) | Host wrapper (`src/lowlevel/SnappyBatchKernels.hip`) | Kernel |
|---|---|---|
| `arctoBatchedSnappyCompressGetTempSize` (168-187) → `*temp_bytes = 0` | — | — |
| `arctoBatchedSnappyCompressGetMaxOutputChunkSize` (189-206) → `32 + n + n/6` (72-76) | — | — |
| `arctoBatchedSnappyCompressAsync` (208-245) | `gpu_snap` (162-178): `dim3 block(COMP_THREADS_PER_BLOCK)`, `grid(count)` | `snap_kernel` (65-82) → `snappy::do_snap` (`src/snappy/compression.hiph:281-385`) |
| `arctoBatchedSnappyDecompressGetTempSize` (84-102) → 0 | — | — |
| `arctoBatchedSnappyGetDecompressSizeAsync` (104-130) | `gpu_get_uncompressed_sizes` (200-213): `block(warpsize)`, `grid(count)` | `get_uncompressed_sizes_kernel` (84-134), one active thread |
| `arctoBatchedSnappyDecompressAsync` (132-166) | `gpu_unsnap` (180-198): `block(DECOMP_THREADS_PER_BLOCK)`, `grid(count32)` | `unsnap_kernel` (145-160) → `snappy::do_unsnap` (`src/snappy/decompression.hiph:195-211`) → `_do_unsnap<DecodeSymbols<TryDecodeStringOf2To3ByteSymbols, TryDecodeStringOf2To5ByteSymbols>, PrefetchByteStream, ProcessSymbols>` (106-190) |

Both batched paths need **zero temp space** (`SnappyBatch.cpp:94,179`); per-chunk state lives entirely in static LDS. One thread block per chunk; the grid is simply `count`.

**HLIF manager → hlif kernels**

`SnappyManager` (`include/arcto/snappy.hpp:62-67`, pimpl) → `SnappyBatchManager : BatchManager<SnappyFormatSpecHeader>` (`src/highlevel/SnappyManager.hpp:66-139`):
- `compute_compression_max_block_occupancy()` (96-99) → `snappyHlifCompMaxBlockOccupancy` (`SnappyHlifKernels.hip:161-176`): `hipOccupancyMaxActiveBlocksPerMultiprocessor(HlifCompressBatchKernel<snappy_compress_wrapper>, COMP_THREADS_PER_BLOCK, 0)` × `multiProcessorCount`.
- `compute_decompression_max_block_occupancy()` (101-104) → `snappyHlifDecompMaxBlockOccupancy` (178-193), same with `HlifDecompressBatchKernel<snappy_decompress_wrapper,1>` and `DECOMP_THREADS_PER_BLOCK`.
- `do_batch_compress` (111-117) → `snappyHlifBatchCompress` (`SnappyHlifKernels.hip:124-134`): `grid(max_ctas)`, `block(COMP_THREADS_PER_BLOCK)`, `HlifCompressBatchKernel<snappy_compress_wrapper>` (`hlif_shared.hiph:254-273`) → persistent loop `HlifCompressBatch` (160-227): `compress_chunk` → `snappy::do_snap` into a per-CTA scratch slot (`scratch_buffer + this_ix_chunk*max_comp_chunk_size`, 181), then `atomicAdd(ix_output, size)` for the output offset (200-202), `copyScratchBuffer` (127-158), next chunk via `atomicAdd(ix_chunk,1)` (223).
- `do_batch_decompress` (119-138) → `snappyHlifBatchDecompress` (136-159): `grid(max_ctas)`, `block(DECOMP_THREADS_PER_BLOCK)`, `HlifDecompressBatchKernel<snappy_decompress_wrapper,1>` (419-447) → `HlifDecompressBatch` (285-340): persistent loop, `decompress_chunk` → `snappy::do_unsnap`, chunk grabbing via `atomicAdd(ix_chunk,1)` (335), `cg::this_thread_block().sync()`.
- Temp/scratch for HLIF: `max_comp_ctas * max_comp_chunk_size` (`BatchManager.hpp:272`), plus `ix_chunk`/`ix_output` counters and the common header (`hlif_shared.hiph:108-125`). `SnappyFormatSpecHeader` is empty (`snappy.hpp:58-60`). `share_buffer` (dynamic LDS) is declared but unused by Snappy (`shmem_size = 0`, `SnappyHlifKernels.hip:167,184`).

### 1.2 Data formats

- **Snappy raw stream**: varint uncompressed length (1-5 bytes, limited to 2^31, `decompression.hiph:71-104`, `SnappyBatchKernels.hip:98-130`), then tag-byte-led elements (`symbol.hiph:41-70`): literal `xxxxxx00` (len-1 in upper 6 bits; 60..63 → 1..4 extra length bytes), copy-1 `xxxxxx01` (3-bit len-4, 11-bit offset), copy-2 `xxxxxx10` (6-bit len-1, 16-bit LE offset), copy-4 `xxxxxx11` (32-bit offset).
- **Compressor emission constraints** (`config.h:83-93`, `compression.hiph`): `HASH_BITS=12` (4096 × uint16 entries), `MAX_LITERAL_LENGTH=256` (literal tags ≤ 2 bytes), `MAX_COPY_LENGTH=64`, `MAX_COPY_DISTANCE=32768`; `StoreCopy` (129-151) emits only copy-1 (len<12, dist<2048) and copy-2; copy-4 is never produced but is accepted by the decoder (single-thread path, `decompression_decode.hiph:99-103`; test `SnappyLargeTokens_test.cpp:534`).
- **Internal LZ77 symbol** (`symbol.hiph:75-80`): `{int32 len, int32 offset}`; `offset>0` = copy distance, `offset≤0` = negated byte-stream cursor of the first literal byte. 8 bytes (12 with `ARCTO_PRINT_DEBUG_INFO`).
- **Decoder queue** (`decompression_state.hiph:56-63`): `batch[BATCH_COUNT*BATCH_SIZE]` = 4 × 64 symbols (wave64; 4 × 32 on wave32/CUDA), `batch_len[4]` (0 = empty, >0 = count, <0 = end), `batch_prefetch_rdpos[4]`, byte ring `buf[PREFETCH_SIZE=4096]`, `prefetch_wrpos/rdpos/end`.
- **Compressor state** (`compression_state.hiph:49-59`): pointers, `volatile literal_length/copy_length/copy_distance`, `uint16_t hash_map[4096]` (8 KB).

### 1.3 Warp-size plumbing

`arcto_device_types.h:28-44` defines `USE_WARPSIZE_64` on AMD unless `USE_WARPSIZE_32`, and `warp_mask_t = uint64_t/uint32_t`, `warpsize = 64/32`. `config.h:45-53,93,99,114` derive `BATCH_SIZE = 64/32`, `COMP_THREADS_PER_BLOCK = 2*warpsize`, `DECOMP_THREADS_PER_BLOCK = 3*warpsize`. CMake (`CMakeLists.txt:56,128-132`) adds `USE_WARPSIZE_32` or `ENABLE_HIP_OPT_WARPSIZE64`; Snappy never references `ENABLE_HIP_OPT_WARPSIZE64` (only `LZ4Kernels.hiph:87` does). The CUDA backend (`include/cuda_shim/hip/hip_runtime.h`) gets `warpsize=32`, `__shfl_sync/__ballot_sync/__syncwarp/__nanosleep` via `INDEPENDENT_THREAD_SCHEDULING` (`device_functions.hiph:123-149,176-177`).

### 1.4 Compressor structure (`compression.hiph:281-385`) — 2 waves per chunk

- Init (lane 0): write varint header, set `s->dst/end/…` (303-326); all threads clear the hash map with 4-byte LDS stores (330-334); `__syncthreads` (335).
- Main loop (338-378), one literal+copy pair per iteration, software-pipelined between the two waves:
  - All lanes read `literal_length/copy_length/copy_distance` from LDS (339-341); `__syncthreads` (342).
  - **WARP0** (`t < WARPSIZE`, 343-357): `StoreLiterals` (73-117; lane 0 writes the tag, all lanes copy bytes `dst[i]=src[i]` strided by 64), `StoreCopy` by lane 0 (352), `SYNCWARP()` (no-op on AMD), lane 0 publishes `s->dst`.
  - **WARP1** (`WARPSIZE ≤ t < 2*WARPSIZE`, 359-375): `FindFourByteMatch` (190-246): up to 4 rounds (wave64) of: `unaligned_load32(src+pos+t)` (2 aligned dword loads + `__funnelshift_r`, `device_functions.hiph:330-341`), `snap_hash`, `HashMatchAny` (12 ballots, 157-172), `SHFL1(data32, …)`, LDS `hash_map[hash]` lookup, verifying `unaligned_load32(src+offset)`, ballot, `ffs`, hash-map update; then `Match60` (251-269) extends the 4-byte match by up to 60 bytes with a single ballot on wave64.
  - `__syncthreads` (377).
- Epilogue: lane 0 writes `*device_out_bytes` and status (380-384).

### 1.5 Decoder structure (`decompression.hiph:106-190`) — 3 specialised waves per chunk

Roles (`decompression.hiph:164-182`): **WARP0** = `DecodeSymbols::apply` (lanes `t<64`), **WARP1** = `PrefetchByteStream::apply` (`t & 63`), **WARP2** = `ProcessSymbols::apply` (`t & 63`). They communicate only through the `volatile unsnap_queue_s q` in LDS and spin-wait on each other.

**Prefetcher (WARP1, `decompression_prefetch.hiph:71-129`)**: lane `t<pos` copies the first `align_bytes` so that `base+pos` becomes GROUPSIZE-aligned (64 B on wave64; 79-86). Loop: lane 0 publishes `prefetch_wrpos`, then polls `prefetch_rdpos` until at most `PREFETCH_SIZE - PREFETCH_SECTORS*GROUPSIZE` (= 4096-512) bytes are outstanding, or `prefetch_end` (91-109); `blen` is broadcast with `SHFL10` (110); full granule: each lane loads 8 bytes `base[pos + t + i*64]` and stores them byte-wise into `buf[(…)&4095]` (111-118); tail: byte loop (119-126).

**Decoder (WARP0, `decompression_decode.hiph:207-289`)**: per batch:
1. Lane 0 waits until `prefetch_wrpos ≥ min(cur + 5*BATCH_SIZE, end)` (`WAIT_FOR_PREFETCHER`, 163-172) and picks the batch slot pointer `b` (220-223).
2. `TryDecodeStringOf2To3ByteSymbols::apply` (`decompression_decode_strategies.hiph:118-185`), all 64 lanes: broadcast `cur`; each lane reads tag candidates at `cur+t`, `cur+t+64`, `cur+t+128` from the LDS ring and ballots "is a 3-byte symbol" (literal of len 2 or copy-2) → `v0,v1,v2` (130-137). Lane 0 folds the 192 bits through `get_len3_mask_64` (16 dependent `k_len3lut` lookups, `warp_scans.hiph:221-300`) and broadcasts the 64-bit mask (139). Each lane computes its symbol start `cur + 2t + popc(mask below t)` (140; `warp_scans.hiph:376-380`), reads its real tag, ballots "long symbol" (141-143); `batch_len` = index of the first long symbol (149). Lanes `t<batch_len` decode `(len, offset)` and write `b[t]` (153-158); wave prefix-sum of lengths (`WarpReduce<64,64>::prefix_sum`, 160); `bytes_left/dst_pos` broadcast from lane 0 (161-162); range check via ballot + `ffs` trims `batch_len` (165-168); `cur` is recomputed from the last accepted lane, lane 0 updates `dst_pos/bytes_left` (170-181); returns the stopping tag (184).
3. If the stop was caused by a literal of 3-4 chars and `batch_len < 62` (`decompression_decode.hiph:243`), `TryDecodeStringOf2To5ByteSymbols::apply` (227-324) runs: per round, lane tags → 2-bit length classes → two ballots → lane 0 `get_len5_mask_64` (`warp_scans.hiph:311-326`) → positions (`395-414`) → same write/prefix-sum/range-check dance, appending `batch_add` symbols; repeats while `batch_add ≥ 6`.
4. Lane 0 serially decodes the remainder (`decode_and_fill_batch_using_single_thread`, 72-150): long literals, copy-4, anything the parallel strategies rejected, one symbol per loop with 1-5 dependent LDS byte reads.
5. Lane 0 submits (`batch_len[batch]=n`, `batch_prefetch_rdpos[batch]=cur`, 183-197, 267-271); `batch_len/bytes_left` are broadcast (275-276); lane 0 waits until the *next* slot is free (`WAIT_FOR_SYMBOL_PROCESSOR`, 174-181, 278); exit on `bytes_left==0`, then set `prefetch_end`, `batch_len[batch]=-1`, error (283-288).

**Processor (WARP2, `decompression_process.hiph:79-263`)**: lane 0 polls `batch_len[batch]` (90-101); lanes `t<batch_len` load their symbol (106-111). Fast path when the first two distances are both > 8 (118-147): repeatedly prefix-sum lengths, build `stop_mask` (first symbol whose copy source overlaps the current output window) and `start_mask` (bit per symbol start within the 64-byte window, via `WarpReduce::sum` of `1<<bofs`), take `n` symbols, and every lane writes exactly one output byte from its symbol's source (copy from `out - dist` or literal from `literal_base`, 138-139); continue while `n ≥ 4`. Per-symbol loop (149-254): pairs of independent small symbols combined into ≤64 bytes (155-180); copies: up to 2×64 bytes per step with `pos % dist` for overlapping sources (181-205); literals: `LITERAL_SECTORS=4` × 64 bytes per step, read from the LDS ring if still resident else from global (206-252). End of batch: `SYNCWARP()` (no-op on AMD), lane 0 advances `prefetch_rdpos` and frees the slot (255-260).

### 1.6 The `snappy/` tool directory (not built into the library)

- `snappy/arcto_snappy.py` (27-178): pure-Python reference parser of a Snappy stream into `(cursor, nbytes, dst_pos, LZ77Symbol)` tuples; `parse_symbol` (103-178) is a line-by-line transliteration of `decode_and_fill_batch_using_single_thread` (same branches and range checks). Used to list the symbols the GPU decoder *should* produce when debugging.
- `snappy/len3_mask.py` (62-92) and `snappy/len3_mask_64.py` (64-98): bit-string reference model of the len3 mask semantics (consume 2 or 3 bits per symbol depending on the ballot bit), applied to three 64-bit ballot words; they assert known cases (`cur_t 141/172`, "app input 2") and generate random test vectors for the C++ test.
- `snappy/len3_mask_64.cpp` (143-221, tests 227-348): host copy of `get_len3_mask_64` with the same `k_len3lut` table as `decompression_decode_warp_scans.hiph:66-130`; validates the wave64 generalisation (16 lookups, 192-bit input) of nvcomp's 32-lane `get_len3_mask_32` (8 lookups). Note the LUT itself is **not** generated by these scripts; the generator loop is the comment at `warp_scans.hiph:54-63`.
- `snappy/snappy_example.py` (+ `requirements.txt: python-snappy`): compresses sample inputs with python-snappy, compares/decompresses hex dumps pasted from ARCTO runs (format compatibility check), and dumps symbol lists via `arcto_snappy.decode_symbols`; the embedded byte arrays correspond to the `tests/test_snappy_app.cpp` vectors.
Relation to kernels: they are developer aids for the wave64 port of the decoder's warp-scan (`len3/len5` masks) and the single-thread parser; nothing is included from `src/`.

---

## 2. Kernel inventory

LDS sizes computed from the struct layouts (8-byte pointer alignment; wave64 values first, wave32/CUDA in parentheses). LDS-limited occupancy assumes 64 KB/CU and 512 B allocation granularity on CDNA.

| Kernel | file:line | Launch config | LDS (static/dynamic) & layout | Wave primitives / hipcub | Global access pattern | Regs / occupancy hints | Notes |
|---|---|---|---|---|---|---|---|
| `snap_kernel` | `SnappyBatchKernels.hip:65-82`; body `compression.hiph:281-385` | block (128,1,1) = 2 waves (64 on wave32/CUDA); grid (count,1,1); `__launch_bounds__(COMP_THREADS_PER_BLOCK)` single arg; no dynamic LDS | static `snap_state_s` ≈ 8,248 B: 5 pointer/len fields (40 B) + 3 volatile u32 (12 B) + `uint16 hash_map[4096]` (8,192 B) (`compression_state.hiph:49-59`); random 2-byte LDS reads/writes into hash_map | `__ballot` (12× in `HashMatchAny` 157-172, 1× match 230, 1× `Match60` 259), `__shfl` ×1 divergent index (213), `__clzll/__ffsll`, `__funnelshift_r`, `__syncthreads` ×2 per iteration (342, 377), `SYNCWARP` no-op (355); no hipcub | WARP1: `unaligned_load32(src+pos+t)` → 2 dword loads/lane, 64 lanes cover ~260 B contiguous (coalesced); verification `unaligned_load32(src+offset)` random within 32 KB back (L1/L2); `Match60` byte loads `src1[t],src2[t]` contiguous. WARP0: `dst[i]=src[i]` byte loads+stores, 64 B per instruction; tag/copy bytes by lane 0 | LDS-limited: ⌊64K/8.5K⌋ = 7 blocks/CU = 14 waves/CU (3.5/SIMD); VGPR budget at that occupancy 128+ so registers unlikely to bind; per-block critical path is latency-bound (global→LDS→global dependency chain + 2 barriers per symbol pair) | Only cross-block parallelism; wave64 gives 64-byte match rounds (4 rounds per 256-byte literal window vs 8 on warp32, `compression.hiph:203,244`) |
| `unsnap_kernel` | `SnappyBatchKernels.hip:145-160`; body `decompression.hiph:106-190` | block (192,1,1) = 3 waves (96 on wave32/CUDA); grid (count32,1,1) (note TODO at 192); `__launch_bounds__(DECOMP_THREADS_PER_BLOCK)`; no dynamic LDS | static `unsnap_state_s` ≈ 6,256 B (5,232 B): header 32 B; `q`: 44 B scalars + `LZ77Symbol batch[256]` 2,048 B (1,024) at struct offset 76 (**4-byte aligned only**) + `uint8 buf[4096]` ring at offset 2,124 (dword-aligned) + `gpu_input_parameters` 32 B (`decompression_state.hiph:56-124`); all `q` accesses are `volatile` | `__ballot` (64-bit), `__shfl` broadcasts (`SHFL10`) and indexed (`SHFL1`), `__shfl_xor` (scan/reduce, `device_functions.hiph:228-268`), `__ffsll/__popcll/__clzll`, `clock()` as NANOSLEEP (176-181), `__constant__ k_len3lut` 1 KB (`warp_scans.hiph:66`), `__syncthreads` ×3 total (131,163,181), `SYNCWARP` no-op; no hipcub/CG | WARP1: 8 `global_load_ubyte` per lane per 512 B (contiguous, 64 B per instruction) → LDS byte stores. WARP2: byte stores `out[t]`, `out[i*64+t]` (contiguous 64 B per instruction); copy sources `out - dist` (random, recently written → L2 on CDNA); literals from LDS ring or `literal_base` bytes. WARP0: LDS-only after the header | LDS-limited ≈ 9 blocks/CU = 27 waves/CU; VGPR count is the likely binder (decoder holds 64-bit masks, LUT chain, 2-to-5 loop) — unknown statically; see §6 | Spin-wait trio; one chunk per block; `BATCH_SIZE=64` on wave64 doubles the symbol batch and the serial `get_len3_mask_64` chain (16 vs 8 lookups) |
| `get_uncompressed_sizes_kernel` | `SnappyBatchKernels.hip:84-134` | block (warpsize,1,1); grid (count,1,1); `__launch_bounds__(warpsize)` | none | none (thread 0 only) | 1-5 dependent byte loads per chunk | 1 of 64 lanes active; `count` workgroups dispatched | Trivial runtime but a free geometry win (§5 D-10) |
| `HlifCompressBatchKernel<snappy_compress_wrapper>` | `hlif_shared.hiph:254-273` (launch `SnappyHlifKernels.hip:124-134`) | block (128,1,1); grid (max_ctas = CUs × occupancy-API blocks/CU); **no `__launch_bounds__`**; dynamic LDS 0 | `do_snap` 8,248 B + `ix_chunks[1]` + `output_status[1]` (8 B) | `cg::this_thread_block().sync()`, 64-bit `atomicAdd` on `ix_output`, 32/64-bit `atomicAdd` on `ix_chunk` (200-202, 223) | as `snap_kernel` + `copyScratchBuffer` (127-158): `char4` loads, **byte stores** to the final buffer | persistent CTAs; occupancy comes from `hipOccupancyMaxActiveBlocksPerMultiprocessor` (`SnappyHlifKernels.hip:168-173`) | scratch slot per CTA computed once before the loop (181) |
| `HlifDecompressBatchKernel<snappy_decompress_wrapper,1>` | `hlif_shared.hiph:419-447` (launch `SnappyHlifKernels.hip:136-159`) | block (192,1,1); grid (max_ctas); **no `__launch_bounds__`**; dynamic LDS 0 | `do_unsnap` 6,256 B + 8 B | `cg::this_thread_block().sync()`, `atomicAdd(ix_chunk,1)` (335) | as `unsnap_kernel` | persistent CTAs | `chunks_per_block==1`, so the `tiled_partition<warpsize>` path (383) is not instantiated for Snappy |

---

## 3. NVIDIA-isms and AMD pitfalls found

Ordered roughly by expected performance relevance on AMD.

1. **Spin-waits without yielding** — `device_functions.hiph:176-181`: on anything but `__CUDA_ARCH__ >= 700`, `NANOSLEEP(d)` is `clock()`. On AMD that is an `s_memtime`/cycle-counter read (result discarded; possibly DCE'd), not a sleep. The four polling loops — decoder `WAIT_FOR_PREFETCHER` (`decompression_decode.hiph:168-171`) and `WAIT_FOR_SYMBOL_PROCESSOR` (178-180, hard-coded `NANOSLEEP(100)`), prefetcher (`decompression_prefetch.hiph:97-108`), processor (`decompression_process.hiph:92-96`) — therefore become tight `volatile` LDS polls that consume issue slots and LDS bandwidth on the SIMD they share with other chunks' productive waves. The AMD constants in `config.h:101-104` are identical to the NVIDIA ones and are ignored anyway. The right AMD primitive is `__builtin_amdgcn_s_sleep(n)` (~64·n cycles, yields the SIMD).

2. **`__shfl`-based broadcasts of wave-uniform values** — `SHFL10/SHFL1` (`device_functions.hiph:150-162`) lower to `ds_bpermute_b32` on AMD (LDS crossbar latency, two ops for 64-bit). Many call sites broadcast values that are wave-uniform or whose source lane is uniform: `decode_strategies.hiph:130,139,145,149,161-162,171-172,184,285,308-309,313-314`, `decompression_decode.hiph:275-276`, `decompression_prefetch.hiph:110`, `decompression_process.hiph:101,118,141,151-153,157`. `v_readfirstlane_b32` / `v_readlane_b32` (few cycles, result lands in an SGPR) are the AMD-native equivalents; `hipcc` does not pattern-match them from `__shfl`. Divergent-index shuffles (`compression.hiph:213`, `decompression_process.hiph:134-135,142-143`) must stay `bpermute`.

3. **Shuffle-tree scans/reductions instead of DPP** — `WarpReducePos64`/`WarpReduceSum64` (`device_functions.hiph:228-240`): 6 dependent `SHFL1`/`SHFL1_XOR` steps (12 `bpermute`s for the 64-bit `start_mask` sum). Used on the decoder hot path (`decode_strategies.hiph:160,306`) and in the processor's inner loop (`decompression_process.hiph:123,127`). rocPRIM/hipCUB `WarpScan`/`WarpReduce` use DPP row/bank ops on AMD (much lower latency); `amd_optimizations.h:294-318` scans are also `__shfl_up`-based (no DPP), so they would not help either.

4. **"Lane 0 computes, then broadcast" for SALU-able values** — `decode_strategies.hiph:139` (`SHFL10((t==0) ? get_len3_mask<…>(v0,v1,v2) : 0)`), 149 (`batch_len`), 285 (`len5_mask`), 145 (`b` pointer). The inputs are ballot results (SGPR-uniform); forcing a divergent `t==0` region turns a pure-SALU computation (`s_lshr_b64`, `s_bcnt1`, scalar `s_load` of the `__constant__` LUT) into VALU + a 64-bit `bpermute`. The code itself notes "identical: len3_mask = get_len3_mask(...)".

5. **LDS layout of the symbol queue** — `decompression_state.hiph:56-63`: `LZ77Symbol batch[]` starts at byte offset 76 of `unsnap_state_s` (32 + 44), i.e. 4-byte but not 8-byte aligned; `set()/get()` (`symbol.hiph:88-107`) are two separate `volatile` 32-bit accesses at an 8-byte stride across lanes → two `ds_write_b32`/`ds_read_b32` per lane with lanes `t, t+16` on the same bank (2-way conflict per 32-lane half) where a single aligned `ds_write_b64` would be conflict-free.

6. **Byte-granular prefetch** — `decompression_prefetch.hiph:113-118`: 8 `global_load_ubyte` + 8 `ds_write_b8` per lane per 512 B, even though the prefetcher deliberately aligns `base+pos` to 64 B on wave64 (79-86) so dword (or wider) global loads would be aligned and still lane-contiguous.

7. **Byte-granular output in the processor** — `decompression_process.hiph:139,174,199,203,228,250`: `out[…] = byte`; per wave instruction only 64 B are written. On AMD the dominant cost for literal-heavy data is instruction issue and per-request overhead rather than bandwidth; NVIDIA's original has the same shape but AMD permits unaligned dword global access (unaligned-access mode is the ROCm default), which the CUDA backend does not.

8. **Serial decoder reads one byte per LDS op** — `decompression_decode.hiph:84-134`: `read_byte(0,cur)`, `cur+1`, … up to 5 dependent `volatile` `ds_read_u8` + `s_waitcnt` per symbol while 63 lanes idle. On wave64 this path is entered whenever a batch hits a literal > 4 chars or copy-4.

9. **Integer modulo on the copy hot path** — `decompression_process.hiph:188,194` (`pos % dist` for overlapping copies). No hardware integer divide on AMD (≈30-40 VALU ops); same in the CUDA original, so not an AMD regression, but it inflates the per-symbol latency of RLE-like data.

10. **Launch bounds** — `SnappyBatchKernels.hip:65,84,145` use the single-argument form; the HLIF kernels (`hlif_shared.hiph:254,419`) have none. On HIP the second argument means *minimum waves per EU (SIMD)*, not *blocks per SM* as in CUDA — any portable macro must encode that difference.

11. **Single-thread, one-wave-per-chunk size kernel** — `SnappyBatchKernels.hip:84-134,207`: 64-lane workgroups with one active lane, `count` workgroups. Harmless but 64× lane waste and dispatch overhead for large batches.

12. **Compile-time wave size with no runtime guard (in this tree)** — `arcto_device_types.h:28-44`; gfx1100 compiles HIP kernels in wave32 by default, so a build without `-DUSE_WARPSIZE_32` (or without `-mwavefrontsize64`) silently breaks every 64-lane assumption (3-wave blocks spanning 6 hardware waves, 64-bit ballots/shuffles). The coordinator notes `origin/opt/curated` adds `HipUtils::check_wave_size()` at the batched entry points, which addresses exactly this; the HLIF path (`SnappyHlifKernels.hip`) would need the same guard.

13. **Strict-aliasing type punning** — `device_functions.hiph:206-209` and `warp_scans.hiph:335-348` (`*reinterpret_cast<uint32_t*>(&mask_64)`); only instantiated for the `GROUPMASK_T=uint32_t` on wave64 configuration (not used by `do_unsnap` today). Use `static_cast<uint32_t>`.

14. **Compressor byte-wise literal copy and 4-byte hash clear** — `compression.hiph:111-115` (`dst[i]=src[i]` byte loads/stores) and 330-334 (16 × `ds_write_b32` per thread); minor.

15. **Prefetch ring depth vs. wave64 granule** — `config.h:59-67,116-117`: the granule is `PREFETCH_SECTORS*GROUPSIZE` = 512 B on wave64 (256 B on CUDA) but the ring stayed 4 KB (8 granules instead of 16); the file's own TODO (117) flags it.

16. **HLIF scratch copy** — `hlif_shared.hiph:139-148`: `char4` load then four byte stores per thread into the final compressed buffer (shared infrastructure, affects HLIF Snappy compression time on AMD).

17. **`SYNCWARP()` is a no-op on AMD** (`device_functions.hiph:163-165`) — correct because AMD waves execute in lockstep (no ITS), and the call sites (`compression.hiph:355`, `decompression_prefetch.hiph:90`, `decompression_process.hiph:255`) do not rely on memory ordering beyond same-wave program order. Cross-lane RAW through global memory in the processor (`out[t] = *src` with `src` pointing into earlier output) relies on the per-wave in-order vector-memory pipeline — the same reliance as the CUDA original; existing tests exercise it.

18. **`volatile` everywhere** (`decompression_state.hiph:122`, `compression_state.hiph:55-57`) prevents the compiler from merging byte accesses into wider LDS ops; any vectorisation must be written explicitly with wider volatile types.

19. **LDS size assumptions**: none of the 48 KB NVIDIA kind; both kernels use < 9 KB static LDS and no dynamic LDS, so they fit every target comfortably. Cooperative groups are used only in the HLIF wrappers (`this_thread_block().sync()`, HIP-supported). No hipcub in the codec.

---

## 4. Existing AMD optimisations already applied in this codec

- Wave64 typing: `warp_mask_t=uint64_t`, `warpsize=64` (`arcto_device_types.h:35-44`); `BATCH_SIZE=64` on wave64 (`config.h:45-53`); block sizes `2*warpsize`/`3*warpsize` (`config.h:93,99`).
- 64-bit ballots and specialisations: `ballot1<uint64_t,64>` / `<uint32_t,64>` (`device_functions.hiph:199-209`); `WarpReduce<64,64>` and `<32,64>` with 64-lane shuffle trees (227-268); 64-bit `find_first_set_bit/num_set_bits/num_leading_zero_bits` (51-120).
- Wave64 warp-scans: `get_len3_mask_64` (16 LUT steps over 192 bits), `get_len5_mask_64`, `compute_symbol_position_via_len5_mask<uint64_t>` (`warp_scans.hiph:221-326,407-414`) plus the validating tools in `snappy/`.
- Compressor: ballot-loop `HashMatchAny` for targets without `__match_any_sync` (`compression.hiph:160-171`); single-ballot `Match60` when `GROUPSIZE==64` (260-262); GROUPSIZE-templated literal stores and hash clear; the hash-map initialisation fix noted at 327-329.
- Prefetcher aligns to `GROUPSIZE` (64 B) and prefetches `PREFETCH_SECTORS*GROUPSIZE` per round (`decompression_prefetch.hiph:79-96`); processor handles 2×64-byte copies and `LITERAL_SECTORS`×64-byte literals (`decompression_process.hiph:185-250`).
- Sync-less `__shfl/__ballot` on AMD, `SYNCWARP` no-op (`device_functions.hiph:150-174`); `NANOSLEEP` fallback (176-181).
- Nothing from `src/amd_optimizations.h` is used by the Snappy codec (only `DeltaGPU.hip`, `BitPackGPU.hip`, `RunLengthEncodeGPU.hip` include it). `ENABLE_HIP_OPT_WARPSIZE64` is not referenced by Snappy. Per coordinator, the only Snappy-related change on `origin/opt/curated` is the `check_wave_size()` guard in `SnappyBatch.cpp`.

---

## 5. Optimisation opportunities

Conventions: impact is a rough, unmeasured estimate for the kernel in question; "CUDA risk" refers to the experimental CUDA backend; effort S (<½ day), M (1-3 days incl. tests), L (>3 days). Every item keeps the compressed bytes and the decoded bytes identical, and keeps nvcomp's parse/emit logic; they change only how the same work is scheduled, synchronised, or moved through memory. Measurement for all: `benchmark_snappy_chunked -f <files> -p <chunk> -w 2 -i 10 [-c true]` (`benchmarks/benchmark_template_chunked.cuh:1496-1547`; throughput = uncompressed bytes / hipEvent-timed `arctoBatchedSnappy{Compress,Decompress}Async`, round trip verified on the last iteration, 755-774), plus the Catch2 tests (`tests/test_snappy_app.cpp`, `tests/test_snappy_batch_c_api.c`, `src/test/SnappyLargeTokens_test.cpp`; globbed by `tests/CMakeLists.txt:48-96` and `src/test/CMakeLists.txt:34-44`). Use at least three inputs: highly compressible (long copies), text-like (2-to-5-byte symbol mix), incompressible (long literals). Profile with `rocprofv3 --pmc SQ_WAVES SQ_BUSY_CYCLES SQ_INSTS_LDS SQ_LDS_BANK_CONFLICT SQ_WAIT_INST_LDS SQ_INSTS_VALU SQ_INSTS_SALU TCC_HIT TCC_MISS` and `-Rpass-analysis=kernel-resource-usage` (or `--save-temps` + `.vgpr_count/.sgpr_count/.lds_size` notes).

### 5.1 COMPRESSION (`snap_kernel`, HLIF compress)

**C-1 — Software-prefetch the data words for all `FindFourByteMatch` rounds**
- Where: `compression.hiph:207-244` (the `do … while` over up to 4 × 64-byte windows; `unaligned_load32(src + pos + t)` at 209).
- Change: load the (up to 4) per-round `data32` values up front (8 aligned dword loads per lane, the window is contiguous and already L1/L2-resident because it is the chunk being compressed), then run the hash/ballot/LDS/verify chain per round on registers. Equivalent loop, same order of evaluation of matches, same hash-map updates.
- Mechanism: removes one global-load latency (L1 hit ~100+ cycles on CDNA, L2 several hundred) from the start of rounds 2-4; overlaps memory with the ballot/LDS work of round 1. CDNA2 (16 KB L1, high L2 latency) benefits most; gfx942/gfx1100 less.
- Impact: single-digit to ~15 % on literal-heavy inputs (more rounds per iteration); negligible on highly compressible data. Effort M. CUDA risk: none (same code). Algorithm preserved: identical results, only load timing changes. Measure: compress throughput on incompressible input (e.g. random bytes) and text; `SnappyLargeTokens_test` byte-exactness tests (381-447) must still pass bit-for-bit.

**C-2 — One barrier per iteration via ping-pong of `literal_length/copy_length/copy_distance`**
- Where: `compression.hiph:338-378` (barriers at 342 and 377; state fields `compression_state.hiph:55-57`).
- Change: make the three fields a 2-entry array indexed by iteration parity; WARP1 writes slot `(i+1)&1` while everyone read slot `i&1` at the top of iteration *i*. The first `__syncthreads` (342) then becomes unnecessary (the slot being written was last read before the previous iteration's barrier). `s->dst` is produced and consumed by WARP0 only; `pos` is register-replicated; `FindFourByteMatch`'s intra-wave `copy_length` write/read (206, 233-236, 366) stays within WARP1.
- Mechanism: `s_barrier` for a 2-wave workgroup is cheap, but the loop runs thousands of times per chunk and the barrier also serialises the two waves' instruction streams; halving barrier count shortens the per-iteration critical path on all three targets.
- Impact: ~5-10 % compress throughput (estimate). Effort M (careful reasoning + tests). CUDA risk: none. Algorithm preserved: same symbol sequence; only the synchronisation schedule changes. Measure: benchmark + `SnappyLargeTokens_test` byte-exact tests + `test_snappy_batch_c_api`.

**C-3 — Dword-granular literal emission in `StoreLiterals` (AMD only)**
- Where: `compression.hiph:111-115` (`dst[i] = src[i]` strided by 64, up to 256 bytes).
- Change: on `__HIP_PLATFORM_AMD__`, let each lane move 4 bytes (`unaligned_load32` on the source side; store as an `__attribute__((aligned(1)))` `uint32_t` on the destination side, bounds-checked against `end`), with a byte tail; keep the byte loop on CUDA. Output bytes identical.
- Mechanism: 4× fewer VMEM instructions for literal bodies; unaligned dword global stores are legal on ROCm (SH_MEM_CONFIG unaligned mode; LLVM `+unaligned-access-mode` default on amdhsa) — verify in ISA that `global_store_dword` with `align 1` is emitted rather than a byte split.
- Impact: small in general (WARP0 is rarely the bottleneck); noticeable only for incompressible data. Effort M. CUDA risk: guard required (misaligned access faults on NVIDIA). Measure: incompressible-input compress throughput; byte-exactness tests.

**C-4 — Occupancy/launch-bounds audit**
- Where: `SnappyBatchKernels.hip:65`, HLIF `hlif_shared.hiph:254`.
- Change: add a per-platform macro, e.g. `__launch_bounds__(COMP_THREADS_PER_BLOCK, ARCTO_SNAPPY_COMP_MIN_WAVES)` with the HIP meaning "min waves per EU" (e.g. 4) and the CUDA meaning "min blocks per SM" (e.g. 4), after reading the actual VGPR/SGPR/LDS usage; give the HLIF kernels the same bounds.
- Mechanism: LDS caps `snap` at ≈7 blocks/CU (3.5 waves/SIMD); if the compiler spends > 128 VGPRs it drops below that; bounding it keeps LDS the only limiter. Impact: 0 to moderate depending on what the compiler did (unknown statically). Effort S. CUDA risk: none (semantics differ, so keep separate values). Measure: resource-usage remarks + benchmark across the three inputs.

**C-5 — Vectorised hash-map clear and minor LDS hygiene**
- Where: `compression.hiph:330-334`.
- Change: clear `hash_map` with `uint4` stores (4 per thread on wave64) — 16 → 4 `ds_write` per thread; `__align__(16)` already present (296).
- Impact: negligible per chunk (a few hundred cycles) but free. Effort S. CUDA risk: none.

**C-6 — HLIF `copyScratchBuffer` dword stores (shared infrastructure)**
- Where: `hlif_shared.hiph:139-148`.
- Change: store `char4`/`uint32_t` directly when the output offset is 4-byte aligned (AMD: always possible via unaligned stores; CUDA: keep the byte path when misaligned).
- Impact: reduces the HLIF copy phase's store-instruction count 4×; relevant for HLIF Snappy compression only. Effort S. CUDA risk: guard alignment.

Not proposed (would change the algorithm or output): shrinking `HASH_BITS`, changing `MAX_LITERAL_LENGTH`, giving WARP0 search work, multi-chunk blocks with shared barriers.

Grid-filling note (no code change): with one 2-wave block per chunk, MI300X (304 CUs × ~7 blocks) needs > 2,000 chunks in flight to be occupied; measure with `-p 32768` vs `-p 65536` and `-x` duplication to separate per-block latency from occupancy effects.

### 5.2 DECOMPRESSION (`unsnap_kernel`, HLIF decompress, size kernel)

**D-1 — Real sleeps in the spin-waits**
- Where: `device_functions.hiph:176-181` (`NANOSLEEP`), call sites `decompression_decode.hiph:168-171,178-180`, `decompression_prefetch.hiph:107`, `decompression_process.hiph:94`; constants `config.h:101-109`.
- Change: on AMD define `NANOSLEEP(ns)` as `__builtin_amdgcn_s_sleep(k)` with `k` derived from the three constants (start with 1 for decode/process, 2-4 for the prefetcher; `s_sleep` units are ~64 clocks); replace the literal `NANOSLEEP(100)` at 179 with `PROCESS_SLEEP_NS`-style constants so the AMD branch of `config.h` becomes meaningful. Optionally poll once before sleeping.
- Mechanism: a polling wave in a 3-wave workgroup otherwise burns issue cycles and LDS bandwidth on its SIMD; with ~6-9 blocks per CU, a large share of resident waves are pollers at any moment. `s_sleep` yields to the productive waves of other chunks on the same SIMD. Applies equally to CDNA2/3 and RDNA3.
- Impact: potentially the largest single win for decompression under realistic occupancy (tens of percent is plausible); near zero when only a few chunks are resident. Effort S. CUDA risk: none (`__nanosleep` path unchanged). Algorithm preserved: pure scheduling. Measure: decompress throughput vs. `k`; `SQ_WAIT_INST_LDS`, `SQ_INSTS_LDS`, `SQ_BUSY_CYCLES`.

**D-2 — DPP/hipcub warp scan and reduce instead of `WarpReducePos64/WarpReduceSum64`**
- Where: `device_functions.hiph:227-268`; users `decode_strategies.hiph:160,306`, `decompression_process.hiph:123,127`.
- Change: implement `WarpReduce<GROUPSIZE,WARPSIZE>::prefix_sum/sum` with `hipcub::WarpScan<T, GROUPSIZE>::InclusiveSum` and `hipcub::WarpReduce<T, GROUPSIZE>::Sum` (via `src/arcto_hipcub.hiph`, which maps to cub on CUDA); the `TempStorage` objects are tiny and can live in the existing LDS state (per role). Keep the `<32,64>` specialisation (a 32-lane logical warp in a 64-wave) — hipcub supports logical sub-warps.
- Mechanism: rocPRIM selects DPP row/bank ops (a few cycles each) on AMD versus six dependent `ds_bpermute` round trips (twelve for 64-bit); the processor's "combine small symbols" loop (118-147) executes scan + 64-bit reduce + ~5 shuffles per 64 output bytes, so it is latency-dominated by exactly these.
- Impact: 10-25 % on poorly compressible/short-symbol data (estimate); smaller elsewhere. Effort M (add temp storage, verify identical prefix semantics — inclusive scan, in-lane order). CUDA risk: low (cub path). Algorithm preserved: identical numeric results. Measure: benchmark on short-symbol-heavy input; `SQ_INSTS_LDS` should drop; tests.

**D-3 — `readfirstlane`/`readlane` for uniform broadcasts**
- Where: `device_functions.hiph:150-162`; sites listed in §3 item 2.
- Change: add AMD-only `SHFL10(v)` = `__builtin_amdgcn_readfirstlane` (two halves for 64-bit/pointers) and a new `SHFL1_UNIFORM(v, i)` = `__builtin_amdgcn_readlane(v, i)` used only where the source lane is wave-uniform (`batch_len-1`, `batch_add-1`, loop index `i`, `n-1`, `it` is **not** uniform). Callers are all-lanes-active at those points (verify each).
- Mechanism: moves broadcasts off the LDS crossbar (`bpermute`, ~LDS latency) onto VALU/SALU; results become SGPRs, shrinking VGPR pressure.
- Impact: 5-15 % on the decoder/processor critical path (many broadcasts per batch). Effort S-M (site audit). CUDA risk: none (CUDA keeps `__shfl_sync`). Algorithm preserved: same values. Measure: benchmark; ISA diff (`ds_bpermute_b32` count).

**D-4 — Compute wave-uniform masks on all lanes (drop the `t==0` + broadcast)**
- Where: `decode_strategies.hiph:139,149,285` and the `b` pointer broadcast (145) together with the `batch` update in `decompression_decode.hiph:267-271` (move `batch = (batch+1)&3` after the `batch_len` broadcast at 275 so every lane tracks it).
- Change: `len3_mask = get_len3_mask<…>(v0,v1,v2)` on all lanes; `batch_len = short_sym_mask ? ffs-1 : warpsize` on all lanes; `len5_mask` likewise; `b` computed uniformly.
- Mechanism: the inputs are ballot results (SGPRs); without the divergent region the whole `get_len3_mask_64` chain is SALU (`s_lshr_b64`, `s_bfe`, `s_load_dword` from the `__constant__` LUT through the scalar cache) and needs no `bpermute`. Also removes a 64-bit shuffle and a pointer shuffle per batch.
- Impact: small-to-medium (a few hundred cycles per batch); pairs well with D-3. Effort S. CUDA risk: none (redundant lane work is free). Algorithm preserved: identical masks. Measure: ISA check for `s_load_dword` + `s_lshr_b64` in the mask chain; benchmark.

**D-5 — 8-byte-aligned `LZ77Symbol` queue with single 64-bit LDS accesses**
- Where: `decompression_state.hiph:56-63` (add padding so `batch[]` sits at an 8-byte offset, or reorder fields), `symbol.hiph:88-107` (`set/get` via one `volatile uint64_t` store/load, 4-byte fields packed).
- Mechanism: `ds_write_b64`/`ds_read_b64` from 64 lanes at an 8-byte stride is conflict-free full-rate; today it is 2×32-bit ops with 2-way conflicts. Also halves the volatile LDS instruction count in `set()` (strategies 157, 303) and `get()` (process 108).
- Impact: small (LDS is not the bottleneck) but free and it also shrinks the decoder's instruction stream. Effort S. CUDA risk: none. Algorithm preserved: storage only.

**D-6 — Dword-granular prefetch**
- Where: `decompression_prefetch.hiph:111-118` (full-granule path).
- Change: keep the lane→byte mapping contiguous but load `uint32_t` per lane: lane `t` loads `base + pos + 4t + 256i` (i = 0,1 on wave64; `base+pos` is 64-byte aligned by construction, 79-86, and `blen` granules keep it so), then writes its 4 bytes to `buf[(pos + 4t + 256i + j) & 4095]` (4 `ds_write_b8`, conflict-free: lanes hit consecutive banks) — or one `ds_write_b32` when `(pos & 3) == 0` (true only when the chunk's varint header length happens to restore dword alignment; keep the byte fallback). The ring contents are byte-identical.
- Mechanism: 8 `global_load_ubyte` → 2 `global_load_dword` per lane per 512 B (4× fewer VMEM instructions/address computations, same coalescing), more bytes in flight per instruction. For wave32/CUDA (`GROUPSIZE=32`, 32-byte alignment) the same mapping gives 128 B per load instruction.
- Impact: small-to-medium for inputs where the prefetcher is on the critical path (large, poorly compressible chunks). Effort M. CUDA risk: none (aligned dword loads). Measure: incompressible input; `TA_BUSY`/VMEM instruction counts.

**D-7 — Dword window reads in the single-thread decoder**
- Where: `decompression_decode.hiph:84-134` (and `unsnap_queue_s::read_byte`, `decompression_state.hiph:70-75`).
- Change: lane 0 reads two aligned dwords of the ring covering `cur..cur+7` (ring index `(cur & ~3) & 4095` and the next dword, wrap-safe since 4096 % 4 == 0; `buf` itself is at a dword-aligned offset) and extracts `b0..b4` with shifts/`__funnelshift_r`; parsing logic unchanged.
- Mechanism: 1-5 dependent `ds_read_u8`+`s_waitcnt` per symbol → 2 independent `ds_read_b32`; cuts the serial path's per-symbol LDS latency roughly in half. Matters on AMD because LDS latency is higher than NVIDIA's shared memory and the serial path runs with 63 idle lanes on every batch that contains a literal > 4 chars or a copy-4.
- Impact: medium on text-like/well-compressed inputs (serial path dominates), negligible on short-symbol streams. Effort S-M. CUDA risk: none. Measure: text/log-file input; tests `decompress_large_literal`, `decompress_long_2B/4B_match_case`, `test_snappy_app_1/2`.

**D-8 — Wider output stores in `ProcessSymbols` (AMD only, staged)**
- Where: `decompression_process.hiph:206-252` (literals), 181-205 (copies), 155-180 and 131-145 (combine paths).
- Change (stage 1, literals only): when reading from `literal_base` or the LDS ring, let each lane move 4 bytes (`out + 4t`, `unaligned`-tolerant loads/stores on AMD; byte path retained on CUDA and for tails). Stage 2 (optional): copies with `dist ≥ 4` and no overlap within the lane's 4 bytes. Output bytes identical.
- Mechanism: 4× fewer VMEM store instructions; helps gfx942/gfx1100 where the byte-store issue rate rather than HBM bandwidth limits literal-heavy streams. Unaligned dword global access is legal under ROCm's default unaligned mode; verify ISA emits `global_store_dword` with `align 1`. LDS-ring source reads stay byte-wise (ring index may be misaligned) unless `(dist&3)==0`.
- Impact: medium for incompressible data; zero for copy-heavy data. Effort M-L (alignment, tails, the three code paths). CUDA risk: must be guarded. Measure: incompressible input; full round-trip tests including odd-sized chunks and odd output alignments (add a test with deliberately misaligned `device_out_ptr`).

**D-9 — Launch bounds / waves-per-EU and resource audit**
- Where: `SnappyBatchKernels.hip:145`, `hlif_shared.hiph:419` (none).
- Change: after reading actual VGPR counts, add `__launch_bounds__(DECOMP_THREADS_PER_BLOCK, MIN_WAVES)` with platform-specific second argument (HIP: waves/EU, e.g. 6-7 on CDNA if the kernel fits in ≤ 72 VGPRs, 4 if not), same for the HLIF kernel; `-Rpass-analysis=kernel-resource-usage` to confirm no spills (`.vgpr_spill_count`).
- Mechanism: LDS allows ≈9 blocks/CU (27 waves); the decoder's 64-bit masks and loops make register pressure the probable binder; D-3/D-4 (SGPR-ising uniform values) and D-2 reduce VGPR use and may raise occupancy on their own. Impact: unknown until measured, possibly large. Effort S. CUDA risk: none with separate values.

**D-10 — `get_uncompressed_sizes_kernel` geometry**
- Where: `SnappyBatchKernels.hip:84-134,200-213`.
- Change: one chunk per thread, 256 threads per block, `grid = ceil(count/256)`; same varint parsing.
- Mechanism: removes 63/64 idle lanes and ~256× fewer workgroups to dispatch. Impact: tiny in absolute terms (µs), visible only for very large batches. Effort S. CUDA risk: none.

**D-11 — Compile-time tunables for wave64 (flags only)**
- Where: `config.h:59-72,116-117`.
- Change: evaluate `-DLOG2_PREFETCH_SIZE=13` (8 KB ring restores the 16-granule depth nvcomp had with 256-byte granules; LDS rises to ≈10.3 KB/block, still ≥6 blocks/CU), `-DPREFETCH_SECTORS=4` (256-byte granules, same depth, less per-lane register use) and `-DLITERAL_SECTORS=2/8`.
- Mechanism: ring depth governs how far the prefetcher can run ahead of the decoder; on wave64 it is half as deep as designed. Impact: small-medium, input dependent. Effort S. CUDA risk: none (separate defaults if desired).

**D-12 — Experiment: 32-lane decode groups inside a 64-wide wave**
- Where: `decompression.hiph:203-210` (template arguments).
- Change: instantiate `DecodeSymbols<TryDecodeStringOf2To3ByteSymbols<uint32_t, warp_mask_t>, TryDecodeStringOf2To5ByteSymbols<uint32_t, warp_mask_t>>` (supported by `ballot1<uint32_t,64>`, `WarpReduce<32,64>`, `get_len3_mask<uint32_t,uint64_t>`, `read_byte` masking), keep prefetch/process at 64; fix the punning casts (§3 item 13) first.
- Mechanism: halves the serial `get_len3_mask` chain (8 lookups) and the per-batch LDS traffic at the cost of half the lanes; nvcomp's parallel strategies were tuned for 32 symbols per batch, so it is unclear a priori whether 64-wide decode is a net win on AMD. Impact: unknown (could go either way). Effort S to try. CUDA risk: none (CUDA stays 32/32).

**D-13 — Speculative (only if profiling shows L2-bound copy reads): LDS window of recent output for short-distance copies**
- Where: `decompression_process.hiph:138,173,188-194`.
- Idea: mirror the last N KB of decoded output in LDS and serve copies with `dist ≤ N` from it; output identical. LDS budget makes this an L-effort, occupancy-costly change (N = 4 KB → ≈10.3 KB/block; N = 16 KB → 2-3 blocks/CU). Justified only if `TCC_HIT`-based latency measurements show `out - dist` loads are the limiter on CDNA2 (write-through L1, L2 latency). Listed for completeness, not recommended as a first step.

Hygiene (not performance): `static_cast` instead of pointer punning (`device_functions.hiph:208`, `warp_scans.hiph:337,347`); extend the wave-size guard to the HLIF manager path.

---

## 6. Open questions to verify on hardware

1. **Register/occupancy reality**: VGPR/SGPR/AGPR counts, spills and `.lds_size` for `snap_kernel`, `unsnap_kernel` and both HLIF kernels on gfx90a/gfx942/gfx1100 (`-Rpass-analysis=kernel-resource-usage`); achieved waves/SIMD (`rocprofv3` `MeanOccupancyPerCU`/`SQ_LEVEL_WAVES`). Decides C-4/D-9 and whether D-3/D-4 buy occupancy.
2. **How much time is spent polling**: `SQ_WAIT_INST_LDS`, `SQ_INSTS_LDS`, `SQ_BUSY_CYCLES` before/after D-1; best `s_sleep` values per role and per architecture (RDNA3 clocks differ).
3. **Does the compiler already scalarise the `get_len3_mask_64` chain** (look for `s_load_dword`/`s_lshr_b64` vs. `global_load_ubyte`/`v_lshrrev_b64` in the ISA) and does it ever turn uniform `ds_bpermute` into `v_readlane`? Determines the size of D-3/D-4.
4. **Copy-source latency**: are `out - dist` loads hitting L1 (TCP) or L2 on gfx90a/gfx942 (`TCP_TOTAL_CACHE_ACCESSES`, `TCC_HIT/MISS`, `TCC_EA_RDREQ`), and what fraction of decode time is in the copy path for typical data? Gates D-13.
5. **LDS bank conflicts** (`SQ_LDS_BANK_CONFLICT`) for the 8-byte-stride symbol queue (D-5) and the strategies' strided tag reads.
6. **Unaligned access**: confirm `+unaligned-access-mode` is on for the build's targets and that `global_store_dword`/`global_load_dword` with `align 1` are emitted (C-3, D-8); confirm behaviour on gfx1100.
7. **Wave64 vs. wave32 on gfx1100**: the README says gfx1100 "may require" `USE_WARPSIZE_32`; compare a `-mwavefrontsize64` wave64 build (64-symbol batches) with the wave32 build for this codec; also D-12 (32-lane decode groups on CDNA).
8. **Prefetch ring tuning** (D-11): 4 KB/8 KB ring, 4/8/16 sectors, `LITERAL_SECTORS` on each target; whether `WAIT_FOR_PREFETCHER`'s `5*BATCH_SIZE` (320 B on wave64) lookahead ever stalls.
9. **Chunk-size / occupancy curve**: throughput vs. `-p` (16-256 KB) and batch size on MI300X (304 CUs) vs MI250X vs RX 7900 XT, to separate per-block latency improvements from grid-filling effects; the HLIF `hipOccupancyMaxActiveBlocksPerMultiprocessor` result on ROCm for the two kernels (sanity-check against the LDS-derived bound).
10. **hipcub/rocPRIM DPP selection** for `WarpScan<int,64>` / `WarpReduce<uint64_t,64>` on gfx90a/gfx942 and for 32-lane logical warps on gfx1100 (D-2); verify no extra LDS is required beyond the TempStorage objects.
11. **Correctness hardening for the vectorised paths**: run the three test executables plus the benchmark round-trip on chunks with odd sizes, odd input/output alignments, and `BATCH_COUNT` wrap-around; add a randomized compressibility sweep (python-snappy cross-check as in `snappy/snappy_example.py`) before enabling C-3/D-6/D-7/D-8.
