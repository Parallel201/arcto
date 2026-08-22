# ARCTO Benchmarks

Throughput benchmarks for the batched (low-level) APIs: one `benchmark_<codec>_chunked`
executable per codec plus `benchmark_zfp_single` for ZFP. All of them time only the
`arctoBatched*Async` calls (HIP events around the asynchronous call), verify the round trip
on the last iteration, and report uncompressed-bytes throughput.

## Building

Benchmarks are built when `BUILD_BENCHMARKS=ON` is passed to CMake (see the top-level
README for the full configure line). Binaries land in **`build/bin/`**, next to the test
executables, and install to `bin/benchmarks/`.

```bash
# AMD (wave64 parts: gfx90a / gfx942; add -D USE_WARPSIZE_32=ON for gfx1100)
cmake -S . -B build -D CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake \
  -D CMAKE_HIP_ARCHITECTURES="gfx90a;gfx942" -D CMAKE_BUILD_TYPE=Release \
  -D BUILD_BENCHMARKS=ON
cmake --build build -j

# NVIDIA (experimental CUDA backend; the chunked benchmarks then measure nvCOMP,
# which is the cross-vendor baseline)
cmake -S . -B build-cuda -D CUDA_BACKEND=ON -D CMAKE_CUDA_ARCHITECTURES="80;90" \
  -D BUILD_BENCHMARKS=ON
```

## Executables

- `benchmark_lz4_chunked`, `benchmark_snappy_chunked`, `benchmark_cascaded_chunked` —
  always built.
- `benchmark_gdeflate_chunked`, `benchmark_bitcomp_chunked`, `benchmark_ans_chunked` —
  only useful when the corresponding external library is found (`ARCTO_EXTS_ROOT`).
- `benchmark_zfp_single` — ZFP modes (fixed rate / precision / accuracy) with fidelity
  metrics; see `-h`.

## Running a chunked benchmark

```bash
HIP_VISIBLE_DEVICES=0 ./build/bin/benchmark_lz4_chunked -f data.bin -p 65536 -w 2 -i 10 -c true
```

| Flag | Long form | Meaning | Default |
|---|---|---|---|
| `-g N` | `--gpu` | GPU device number | 0 |
| `-f FILE...` | `--input_file` | input file(s) (required) | — |
| `-p N` | `--chunk_size` | chunk size in bytes when splitting the input | 65536 |
| `-x N` | `--duplicate_data` | clone the chunk list N times (fills large GPUs) | 0 |
| `-w N` | `--warmup_count` | warm-up iterations (not timed) | 1 |
| `-i N` | `--iteration_count` | timed iterations to average | 1 |
| `-c true` | `--csv_output` | CSV instead of text | false |
| `-t true` | `--tab_separator` | tabs instead of commas in CSV | false |
| `-F true` | `--file_with_page_sizes` | input files are pages, each prefixed by an int64 size | false |
| `-P true` | `--pinned_input` | one pinned, coalesced host buffer + single bulk H2D (`arctoHostBatch`) | false |
| `-A true` | `--adaptive_tiled` | adaptive tiled staging (`arctoHostBatchAdaptive`); exclusive with `-P` | false |
| `-R true` | `--report_phases` | extra CSV columns with host-side phase costs (`t_alloc_ms`, `t_memcpy_h2h_ms`, `peak_pinned_bytes`, …) | false |

Environment variables: `ARCTO_PER_REP_CSV=<path>` appends one row per timed repetition
(throughput and time for compression and decompression) to `<path>`, with
`ARCTO_PER_REP_TAG=<label>` identifying the configuration — use them for sweeps that need
the raw per-repetition values rather than mean ± stddev.

Tip: with one wave or block per chunk, a 100 MB input at 64 KB chunks is only 1600 chunks;
use `-x` (e.g. `-x 8`…`-x 512`) so the batch is large enough to saturate the GPU before
comparing kernel changes.

## Output

Text mode prints the number of files, uncompressed/compressed bytes, ratio and the two
throughputs. CSV mode (`-c true`) prints one header row and one data row:

```
Files,Duplicate data,Size in MB,Pages,Avg page size in KB,Max page size in KB,Ucompressed size in bytes,Compressed size in bytes,Compression ratio,Compression throughput (uncompressed) in GB/s,Decompression throughput (uncompressed) in GB/s,Compression time (ms),Decompression time (ms),Transfer H2D (ms),Transfer D2H (ms),Total time (ms),Avg chunk time (ms),Comp throughput stddev (GB/s),Decomp throughput stddev (GB/s),Comp time stddev (ms),Decomp time stddev (ms)
```

(`-R` appends `t_alloc_ms, t_memcpy_h2h_ms, peak_pinned_bytes, adaptive_window_bytes, adaptive_num_windows`.)

## Test data

`tests/data/*.bin` ships six 1 MB fixtures (synthetic zeros / binary / random and three
64³ float TTI fields) that double as a compressibility ladder. Larger inputs:

```bash
dd if=/dev/urandom of=test_100mb.bin bs=1M count=100          # incompressible
base64 /dev/urandom | head -c 100M > test_text_100mb.txt       # compressible
```

## Profiling

```bash
rocprofv3 --kernel-trace --stats -- ./build/bin/benchmark_lz4_chunked -f test.bin   # AMD
nsys profile ./build/bin/benchmark_lz4_chunked -f test.bin                           # NVIDIA backend
```

## Troubleshooting

- *Binary not found*: configure with `-D BUILD_BENCHMARKS=ON`; binaries are in `build/bin/`.
- *GPU not detected*: `rocminfo | grep gfx` (AMD) / `nvidia-smi` (NVIDIA); select with
  `HIP_VISIBLE_DEVICES` / `CUDA_VISIBLE_DEVICES` or `-g`.
- *Out of memory*: reduce `-p` (chunk size) or `-x`.
- *ROCm version*: minimum ROCm 6.1; the reference measurements use ROCm 7.0.1 in the
  project's toolchain container.
