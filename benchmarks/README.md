# ARCTO Benchmarks

This directory contains performance benchmarks for ARCTO compression algorithms.

## Building Benchmarks

To build benchmarks, enable the `BUILD_BENCHMARKS` option when configuring with CMake:

### For AMD GPUs (HIP backend):
```bash
mkdir build && cd build
CMAKE_PREFIX_PATH=/opt/rocm/lib/cmake cmake .. \
  -D BUILD_BENCHMARKS=ON \
  -D CMAKE_HIP_ARCHITECTURES="gfx90a;gfx942"  # MI300X uses gfx942
make -j$(nproc)
```

### For NVIDIA GPUs (CUDA backend):
```bash
mkdir build && cd build
cmake .. \
  -D BUILD_BENCHMARKS=ON \
  -D CUDA_BACKEND=ON \
  -D CMAKE_CUDA_ARCHITECTURES="80;90"
make -j$(nproc)
```

Benchmark executables will be created in `build/benchmarks/` or installed to `bin/benchmarks/`.

## Available Benchmarks

- `benchmark_lz4_chunked` - LZ4 compression algorithm
- `benchmark_snappy_chunked` - Snappy compression algorithm
- `benchmark_cascaded_chunked` - Cascaded compression algorithm
- `benchmark_gdeflate_chunked` - GDeflate compression algorithm
- `benchmark_bitcomp_chunked` - Bitcomp compression algorithm
- `benchmark_ans_chunked` - ANS compression algorithm

## Running Benchmarks

### Individual Benchmark

Run a single benchmark on a specific file:

```bash
./build/benchmarks/benchmark_lz4_chunked \
  -g 0 \
  -f /path/to/test/file.bin \
  -i 10 \
  -w 2 \
  -c true
```

**Options:**
- `-g, --gpu <N>` - GPU device number (default: 0)
- `-f, --input_file <path>` - Input file(s) to benchmark (required)
- `-i, --iteration_count <N>` - Number of iterations to average (default: 1)
- `-w, --warmup_count <N>` - Number of warmup iterations (default: 1)
- `-c, --csv_output <true|false>` - Output in CSV format (default: false)
- `-t, --tab_separator <true|false>` - Use tabs instead of commas (default: false)
- `-p, --chunk_size <N>` - Chunk size for splitting data (default: 65536)
- `-x, --duplicate_data <N>` - Duplicate chunks N times (default: 0)

### Using the Benchmark Script

The `benchmark.sh` script automates running benchmarks on multiple files:

```bash
./scripts/benchmark.sh <algorithm> <data_directory> [gpu_id]
```

**Example:**
```bash
# Run LZ4 benchmark on all files in /data/testset on GPU 0
./scripts/benchmark.sh lz4 /data/testset 0
```

**Supported algorithms:** `lz4`, `snappy`, `cascaded`, `gdeflate`, `bitcomp`, `ans`

## GPU-Specific Notes

### AMD MI300X (gfx942)
- 304 Compute Units
- 192 GB HBM3 memory
- 8 TB/s memory bandwidth
- Wave size: 64 threads

To select specific GPU:
```bash
export HIP_VISIBLE_DEVICES=0
./benchmark_lz4_chunked -f test.bin
```

### AMD MI50 (gfx906)
- 60 Compute Units
- 16 GB or 32 GB HBM2

### AMD RX 7900 XT (gfx1100)
- 84 Compute Units
- 20 GB GDDR6
- Note: May require `-D USE_WARPSIZE_32=ON` when building

### NVIDIA GPUs
Use `CUDA_VISIBLE_DEVICES` to select GPU:
```bash
export CUDA_VISIBLE_DEVICES=0
./benchmark_lz4_chunked -f test.bin
```

## Output Format

### Standard Output
```
----------
files: 1
uncompressed (B): 104857600
comp_size: 52428800, compressed ratio: 2.00
compression throughput (GB/s): 15.23
decompression throughput (GB/s): 42.18
```

### CSV Output (`-c true`)
```
Files,Duplicate data,Size in MB,Pages,Avg page size in KB,Max page size in KB,Ucompressed size in bytes,Compressed size in bytes,Compression ratio,Compression throughput (uncompressed) in GB/s,Decompression throughput (uncompressed) in GB/s
1,0,100.00,1600,64.00,64.00,104857600,52428800,2.00,15.23,42.18
```

## Creating Test Data

Generate random test data:
```bash
# 100 MB random file
dd if=/dev/urandom of=test_100mb.bin bs=1M count=100

# Text-based data (more compressible)
base64 /dev/urandom | head -c 100M > test_text_100mb.txt
```

## Performance Profiling

### ROCm Profiler (rocprof)
```bash
rocprof --stats ./benchmark_lz4_chunked -f test.bin
```

### NVIDIA Nsight Systems
```bash
nsys profile ./benchmark_lz4_chunked -f test.bin
```

## Optimization Features Tracking

This benchmark infrastructure supports tracking optimization improvements across features:

- **Feature 1**: Baseline benchmark infrastructure
- **Feature 4+**: AMD-specific optimizations (see main README for details)

Compare performance between branches:
```bash
# Baseline
git checkout feature/add-benchmark-infrastructure
./scripts/benchmark.sh lz4 /data/testset > baseline.csv

# Optimized
git checkout optimize/amd-wavefront-aware
./scripts/benchmark.sh lz4 /data/testset > optimized.csv

# Compare
diff baseline.csv optimized.csv
```

## Troubleshooting

### Benchmark binary not found
Make sure you built with `-D BUILD_BENCHMARKS=ON`

### GPU not detected
```bash
# AMD
rocminfo | grep "Name:"

# NVIDIA
nvidia-smi
```

### Out of memory errors
Reduce chunk size or file size:
```bash
./benchmark_lz4_chunked -f test.bin -p 32768  # 32KB chunks instead of 64KB
```

### ROCm version compatibility
This project has been tested with ROCm 7.0.1. For older versions, you may need to enable workarounds:
```bash
cmake .. -D BUILD_BENCHMARKS=ON -D CG_WORKAROUND=1
```
