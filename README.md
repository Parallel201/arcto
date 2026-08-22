# ARCTO

Batched compression for **AMD GPUs** (HIP/ROCm), covering lossless byte-level codecs
(LZ4, Snappy, Cascaded) and floating-point compression (ZFP), tuned for RDNA3 and
CDNA2/CDNA3.

**What is new here**
- **ZFP on AMD GPUs** — FIXED_RATE / FIXED_PRECISION / FIXED_ACCURACY running on device,
  bitstream-compatible with canonical zfp. No vendor library provides this on ROCm.
- **Adaptive host-side staging** (`arctoHostBatchAdaptive`) — profile-driven tiled windows
  with an online cost model, keeping peak pinned allocation bounded on multi-GB inputs;
  window sizes tuned per architecture (gfx906 / gfx90a / gfx942 / gfx1100).
- **Cross-vendor evaluation path** — an experimental CUDA backend builds the same HIP
  sources unmodified, so AMD and NVIDIA numbers come from one codebase.
- **Validated on three AMD architectures**, wave64 and wave32.

### Upstream provenance

Branch names move; these commits do not. Recorded so the translation can be
diffed against its ancestor, and so the NVIDIA baseline can be reproduced,
without keeping upstream checkouts in the workspace.

| Upstream | Role | Commit | Tag / branch | Date |
|---|---|---|---|---|
| [NVIDIA/nvcomp](https://github.com/NVIDIA/nvcomp) | Ancestor ARCTO was translated from | `a6e4e64a177e07cd2e5c8c5e07bb66ffefceae84` | `v2.2.0` / `branch-2.2` | 2022-02-07 |

```bash
git clone https://github.com/NVIDIA/nvcomp.git && cd nvcomp
git checkout a6e4e64a177e07cd2e5c8c5e07bb66ffefceae84   # the 2.2 ancestor
```

## Codecs

| Codec | Type | Notes |
|-------|------|-------|
| LZ4 | lossless, byte-level | batched low-level + high-level manager APIs |
| Snappy | lossless, byte-level | batched low-level + high-level manager APIs |
| Cascaded | lossless (RLE + Delta + BitPack) | best for structured/numerical data |
| ZFP | floating-point (3D) | FIXED_RATE / FIXED_PRECISION / FIXED_ACCURACY on GPU via vendored [LLNL zfp](https://github.com/LLNL/zfp); bitstream-compatible with canonical zfp |

ANS, GDeflate and Bitcomp wrappers from nvCOMP 2.2 are present but require
external proprietary libraries (`ENABLE_ANS` / `ENABLE_GDEFLATE` /
`ENABLE_BITCOMP`); they are not built by default.

## Host-side staging

For large batched inputs, two helper APIs reduce H2D transfer cost:

- `arctoHostBatch` (`include/arcto/host_batch.h`) — coalesces all chunks into
  one pinned host buffer, single `hipMemcpyAsync` upload.
- `arctoHostBatchAdaptive` (`include/arcto/host_batch_adaptive.h`) —
  profile-driven tiled windows with an online cost model, for multi-GB inputs
  where peak pinned allocation must stay bounded; per-architecture tuned
  window sizes (gfx906/gfx90a/gfx942/gfx1100).

## Supported GPUs

| Architecture | GPU | Wave size |
|--------------|-----|-----------|
| gfx1100 (RDNA3) | Radeon RX 7900 XT/XTX | 64 or 32 — build with `-D USE_WARPSIZE_32=ON` |
| gfx90a (CDNA2) | Instinct MI210/MI250 | 64 |
| gfx942 (CDNA3) | Instinct MI300X | 64 |

Other gfx9+/RDNA targets should work but are not part of the test matrix.

## Requirements

- ROCm >= 6.1
- CMake >= 3.21
- C++17 compiler

## Building

```bash
cmake -S . -B build \
  -D CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake \
  -D CMAKE_HIP_ARCHITECTURES="gfx90a;gfx942" \
  -D CMAKE_BUILD_TYPE=Release -D BUILD_TESTS=ON -D BUILD_BENCHMARKS=ON
cmake --build build -j

# RDNA3 (wave32) — REQUIRED for gfx1100: add -D USE_WARPSIZE_32=ON
```

Tips: select a GPU with `HIP_VISIBLE_DEVICES=<id>`; if no GPU is present at
configure time, set `CMAKE_HIP_ARCHITECTURES` explicitly.

For cross-vendor comparison studies, an experimental CUDA backend
(`-D CUDA_BACKEND=ON`) builds the same HIP sources unmodified on non-AMD
hardware; it exists to benchmark ARCTO against vendor libraries, not as a
supported target.

## Usage

Low-level batched C API:

```c
#include <arcto/lz4.h>

size_t temp_bytes;
arctoBatchedLZ4CompressGetTempSize(num_chunks, max_chunk_size, opts, &temp_bytes);

size_t max_out_bytes;
arctoBatchedLZ4CompressGetMaxOutputChunkSize(max_chunk_size, opts, &max_out_bytes);

arctoBatchedLZ4CompressAsync(
    device_in_ptrs, device_in_sizes, max_chunk_size, num_chunks,
    temp_buffer, temp_bytes, device_out_ptrs, device_out_sizes, opts, stream);
```

High-level C++ manager:

```cpp
#include <arcto/lz4.hpp>
#include <arcto/arctoManager.hpp>

arcto::LZ4Manager manager{chunk_size, ARCTO_TYPE_CHAR, stream};
manager.configure_compression(uncompressed_size);
manager.compress(device_input, device_output, stream);
manager.decompress(device_input, device_output, stream);
```

## Tests and benchmarks

```bash
cd build && ctest                       # unit + end-to-end round-trip tests
./build/bin/benchmark_lz4_chunked -f data.bin -p 65536 -w 2 -i 10
./build/bin/benchmark_zfp_single  -f field.bin        # ZFP modes + fidelity metrics
```

Useful benchmark flags: `-p` chunk size, `-P` pinned coalesced input,
`-A` adaptive tiled staging, `-R` per-phase host cost report, `-c` CSV output.

## Roadmap

- [ ] GPU-native lossless (reversible) ZFP for 3D float fields — canonical zfp
      only implements reversible mode on the serial CPU backend
- [ ] Kernel optimization pass for CDNA/RDNA (occupancy, warp primitives,
      vectorized copies)
- [ ] Multi-GPU support

## License

ARCTO as a whole is distributed under the [MIT License](LICENSE).
It contains code under other licenses, tracked file-by-file in
[NOTICES.md](NOTICES.md):

- Code derived from NVIDIA nvCOMP 2.2 — [BSD 3-Clause](NVCOMP_2_2_LICENSE)
  or Apache-2.0 (per-file headers preserved).
- `tests/catch.hpp` — Boost Software License 1.0.
- `third_party/zfp` (git submodule) — BSD 3-Clause, LLNL.

## Acknowledgments

- NVIDIA nvCOMP team for the original implementation
- AMD for ROCm/HIP
- LLNL for zfp
