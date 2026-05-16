# HIP Compression Toolkit

A high-performance GPU compression library for AMD GPUs using HIP/ROCm, featuring optimized implementations of LZ4, Snappy, and Cascaded compression algorithms.

## Features

- **LZ4**: Fast lossless compression algorithm
- **Snappy**: High-speed compression developed by Google
- **Cascaded**: Multi-stage compression combining Delta, RLE, and Bit-packing
- **ZFP** *(planned)*: Floating-point array compression

### Optimizations

- Wave64 optimizations for AMD CDNA/RDNA architectures
- Batched compression/decompression APIs
- Async stream support for overlapped operations
- Support for multiple GPU architectures

## Supported GPUs

| Architecture | GPU Models |
|--------------|------------|
| `gfx1100` | Radeon RX 7900 XT/XTX (RDNA 3) |
| `gfx942` | AMD Instinct MI300X (CDNA 3) |
| `gfx90a` | AMD Instinct MI250/MI250X (CDNA 2) |
| `gfx908` | AMD Instinct MI100 (CDNA 1) |
| `gfx906` | AMD Instinct MI50/MI60 |

## Building

### Prerequisites

- ROCm 5.4+ (6.0+ recommended for MI300X)
- CMake 3.18+
- GCC 9+ or Clang 14+

### Build Instructions

```bash
mkdir build && cd build

# For AMD GPUs
cmake .. \
  -D CMAKE_PREFIX_PATH=/opt/rocm \
  -D CMAKE_HIP_ARCHITECTURES="gfx90a;gfx942;gfx1100" \
  -D BUILD_BENCHMARKS=ON \
  -D BUILD_TESTS=ON

make -j$(nproc)
```

### Build Options

| Option | Default | Description |
|--------|---------|-------------|
| `BUILD_BENCHMARKS` | ON | Build benchmark executables |
| `BUILD_TESTS` | ON | Build test suite |
| `CMAKE_HIP_ARCHITECTURES` | gfx90a | Target GPU architectures |
| `BUILD_SHARED_LIBS` | ON | Build shared library |

## Usage

### C API (Low-level Batched)

```c
#include <arcto/lz4.h>

// Get temp buffer size
size_t temp_bytes;
arctoBatchedLZ4CompressGetTempSize(
    num_chunks, max_chunk_size, opts, &temp_bytes);

// Get max output size per chunk
size_t max_out_bytes;
arctoBatchedLZ4CompressGetMaxOutputChunkSize(
    max_chunk_size, opts, &max_out_bytes);

// Compress asynchronously
arctoBatchedLZ4CompressAsync(
    device_in_ptrs, device_in_sizes, max_chunk_size, num_chunks,
    temp_buffer, temp_bytes,
    device_out_ptrs, device_out_sizes,
    opts, stream);

// Decompress
arctoBatchedLZ4DecompressAsync(
    device_in_ptrs, device_in_sizes,
    device_out_sizes, device_actual_out_sizes, num_chunks,
    temp_buffer, temp_bytes,
    device_out_ptrs, device_statuses, stream);
```

### C++ API (High-level Manager)

```cpp
#include <arcto/lz4.hpp>
#include <arcto/arctoManager.hpp>

// Create LZ4 manager
arcto::LZ4Manager manager{chunk_size, ARCTO_TYPE_CHAR, stream};

// Configure compression
manager.configure_compression(uncompressed_size);

// Compress
manager.compress(device_input, device_output, stream);

// Decompress
manager.decompress(device_input, device_output, stream);
```

## Benchmarks

### Running Benchmarks

```bash
# Single algorithm
./build/bin/benchmark_lz4 -f /path/to/data.bin -i 10

# Using benchmark script
./scripts/benchmark.sh lz4 /path/to/test_data/ 0
```

### Benchmark Options

| Option | Description |
|--------|-------------|
| `-f, --input_file` | Input file path (required) |
| `-g, --gpu` | GPU device ID (default: 0) |
| `-i, --iterations` | Number of iterations (default: 10) |
| `-w, --warmup` | Warmup iterations (default: 2) |
| `-p, --chunk_size` | Chunk size in bytes (default: 65536) |
| `-c, --csv` | Output in CSV format |


## Directory Structure

```
hip-compression-toolkit/
├── CMakeLists.txt
├── README.md
├── LICENSE
├── include/
│   └── arcto/
│       ├── lz4.h
│       ├── snappy.h
│       ├── cascaded.h
│       └── shared_types.h
├── src/
│   ├── lz4/
│   ├── snappy/
│   └── cascaded/
├── benchmarks/
│   ├── benchmark_lz4.cu
│   ├── benchmark_snappy.cu
│   └── benchmark_cascaded.cu
├── tests/
│   ├── test_lz4.cpp
│   ├── test_snappy.cpp
│   └── test_cascaded.cpp
└── scripts/
    └── benchmark.sh
```

## Roadmap

- [x] LZ4 compression
- [x] Snappy compression
- [x] Cascaded compression
- [x] Benchmarks
- [ ] ZFP floating-point compression
- [ ] Multi-GPU support
- [ ] Python bindings

## License

This project is based on NVIDIA nvcomp, ported to AMD HIP.
See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## Acknowledgments

- NVIDIA nvcomp team for the original implementation
- AMD ROCm team for HIP runtime
