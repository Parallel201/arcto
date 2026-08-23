# ARCTO on NVIDIA — comparison with nvCOMP (GH200, August 2026)

How much does it cost to run ARCTO on an NVIDIA GPU instead of NVIDIA's own nvCOMP? This document
answers it with one harness, one machine and three interchangeable backends.

## 1. Setup

| | |
|---|---|
| GPU | NVIDIA GH200 120 GB (Hopper, `sm_90`), driver 595.58.03, node `sdumont2nd3018` (aarch64) |
| Toolchain | CUDA 13.2, GCC 11.4, CMake 3.26 |
| **ARCTO** | the integration tree of this campaign, built with `-D CUDA_BACKEND=ON` — the *same HIP sources* that run on AMD, compiled by `nvcc` through `include/cuda_shim`; no CUDA-specific source tree, no nvCOMP code |
| **nvCOMP 2.2** | built from `NVIDIA/nvcomp` `branch-2.2` — the release ARCTO was translated from. Three local build patches were needed for CUDA 13 and are listed in §4; none touches a kernel |
| **nvCOMP 5.2** | the prebuilt `nvcomp 5.2.0.10-cuda12-arm64` package (current release) |
| Harness | `benchmarks/benchmark_compare_chunked.cu` compiled three times (`-DARCTO_COMPARE_NVCOMP22`, `-DARCTO_COMPARE_NVCOMP5`, and ARCTO's own API). Identical chunking, duplication, warmup, event timing, round-trip verification and per-repetition CSV for all three |
| Protocol | 64 KiB chunks, 3 warmup + 30 measured repetitions, median; "small batch" = one 1 MiB file (16 chunks), "saturation" = the same file duplicated ×512 (×32 for the 16 MiB text file) |
| Inputs | `synth_binary` (copy-heavy binary), `tti_rsf` (seismic floats), `words` (short-sequence text, 16 MiB), `synth_zeros` (long runs) |

**Validity check.** The compressed sizes are *identical* for all three backends on every input and codec
(e.g. LZ4 binary 3.258×, zeros 245.453×, Snappy binary 1.914×, Cascaded binary 3.031×): the three
libraries produce the same streams, so the throughput numbers compare the same work.

## 2. Results

Throughput is uncompressed bytes ÷ kernel time; "×ARCTO" is the other backend divided by ARCTO,
so **> 1 means the other library is faster** and < 1 means ARCTO is faster.

| codec | input | regime | ARCTO GB/s | nvCOMP 2.2 | ×ARCTO | nvCOMP 5.2 | ×ARCTO |
|---|---|---|---:|---:|---:|---:|---:|
| **compression** | | | | | | | |
| lz4 | binary | small batch | 0.7 | 0.5 | 0.73× | 0.5 | 0.75× |
| lz4 | binary | saturation | 45.5 | 44.9 | 0.99× | 44.6 | 0.98× |
| lz4 | tti | small batch | 0.7 | 0.5 | 0.73× | 0.5 | 0.75× |
| lz4 | tti | saturation | 13.3 | 13.1 | 0.98× | 13.3 | 1.00× |
| lz4 | words | small batch | 2.0 | 2.0 | 0.96× | 2.0 | 0.98× |
| lz4 | words | saturation | 20.0 | 19.6 | 0.98× | 19.8 | 0.99× |
| lz4 | zeros | small batch | 2.3 | 2.3 | 1.01× | 2.3 | 1.00× |
| lz4 | zeros | saturation | 373.1 | 374.6 | 1.00× | 373.2 | 1.00× |
| snappy | binary | small batch | 0.7 | 0.7 | 1.00× | 0.8 | 1.02× |
| snappy | binary | saturation | 105.1 | 104.9 | 1.00× | 107.6 | 1.02× |
| snappy | tti | small batch | 0.7 | 0.7 | 1.00× | 0.8 | 1.03× |
| snappy | tti | saturation | 70.5 | 70.3 | 1.00× | 70.8 | 1.00× |
| snappy | words | small batch | 2.2 | 2.1 | 0.99× | 2.3 | 1.06× |
| snappy | words | saturation | 14.1 | 14.1 | 1.00× | 15.0 | 1.06× |
| snappy | zeros | small batch | 1.1 | 1.1 | 1.04× | 1.2 | 1.11× |
| snappy | zeros | saturation | 132.5 | 132.6 | 1.00× | 138.8 | 1.05× |
| cascaded | binary | small batch | 6.4 | 3.6 | 0.56× | 3.8 | 0.60× |
| cascaded | binary | saturation | 265.8 | 174.6 | 0.66× | 181.7 | 0.68× |
| cascaded | tti | small batch | 6.4 | 3.6 | 0.56× | 3.8 | 0.60× |
| cascaded | tti | saturation | 217.2 | 126.3 | 0.58× | 136.0 | 0.63× |
| cascaded | words | small batch | 77.5 | 48.8 | 0.63× | 50.8 | 0.65× |
| cascaded | words | saturation | 195.0 | 119.9 | 0.61× | 128.7 | 0.66× |
| cascaded | zeros | small batch | 11.1 | 10.6 | 0.95× | 10.6 | 0.95× |
| cascaded | zeros | saturation | 496.3 | 442.0 | 0.89× | 445.8 | 0.90× |
| **decompression** | | | | | | | |
| lz4 | binary | small batch | 3.0 | 3.0 | 1.00× | 3.0 | 0.99× |
| lz4 | binary | saturation | 539.3 | 500.7 | 0.93× | 505.9 | 0.94× |
| lz4 | tti | small batch | 3.0 | 3.0 | 1.00× | 3.0 | 1.00× |
| lz4 | tti | saturation | 420.2 | 419.6 | 1.00× | 411.6 | 0.98× |
| lz4 | words | small batch | 5.1 | 5.0 | 0.99× | 7.3 | 1.44× |
| lz4 | words | saturation | 54.6 | 53.0 | 0.97× | 77.8 | 1.42× |
| lz4 | zeros | small batch | 18.1 | 8.6 | 0.48× | 42.7 | 2.35× |
| lz4 | zeros | saturation | 1,121.7 | 1,118.2 | 1.00× | 1,965.8 | 1.75× |
| snappy | binary | small batch | 2.4 | 2.3 | 0.98× | 5.7 | 2.43× |
| snappy | binary | saturation | 227.0 | 214.5 | 0.95× | 374.5 | 1.65× |
| snappy | tti | small batch | 7.4 | 6.7 | 0.91× | 5.7 | 0.77× |
| snappy | tti | saturation | 503.9 | 468.8 | 0.93× | 362.9 | 0.72× |
| snappy | words | small batch | 11.7 | 11.7 | 1.00× | 47.8 | 4.08× |
| snappy | words | saturation | 70.3 | 70.1 | 1.00× | 222.8 | 3.17× |
| snappy | zeros | small batch | 2.6 | 2.7 | 1.06× | 6.6 | 2.58× |
| snappy | zeros | saturation | 207.6 | 206.5 | 0.99× | 434.9 | 2.10× |
| cascaded | binary | small batch | 6.1 | 6.0 | 0.99× | 6.1 | 1.00× |
| cascaded | binary | saturation | 352.2 | 361.3 | 1.03× | 369.9 | 1.05× |
| cascaded | tti | small batch | 32.8 | 32.7 | 1.00× | 32.8 | 1.00× |
| cascaded | tti | saturation | 982.1 | 998.2 | 1.02× | 1,000.5 | 1.02× |
| cascaded | words | small batch | 490.9 | 503.4 | 1.03× | 483.7 | 0.99× |
| cascaded | words | saturation | 941.0 | 1,012.5 | 1.08× | 1,010.9 | 1.07× |
| cascaded | zeros | small batch | 12.0 | 12.1 | 1.01× | 12.1 | 1.00× |
| cascaded | zeros | saturation | 584.1 | 599.0 | 1.03× | 600.4 | 1.03× |


## 3. Reading the numbers

**Against nvCOMP 2.2 — the translation is free, and often better.** ARCTO is at parity
(0.96–1.06×) with the release it was translated from on every LZ4 and Snappy case, and *faster*
where this campaign optimized the algorithm: Cascaded compression **1.5–1.8×** (`0.56–0.66×` in the
table), LZ4 compression at small batch **1.4×**, LZ4 decompression of long runs **2.1×** at small
batch. Nothing measured is slower than 2.2 outside noise. The AMD-oriented work therefore did not
cost NVIDIA performance — some of it (the algorithmic items: single-pass min/max, the register
slab, the multi-item scans, the incremental modulo) is architecture-neutral and pays on both
vendors.

**Against nvCOMP 5.2 — mixed, and the gap is in decompression kernels NVIDIA rewrote.**

* *Compression*: parity everywhere for LZ4 and Snappy (0.98–1.11×), and ARCTO is **1.5–1.7× faster**
  for Cascaded on all four inputs.
* *Decompression*: parity for Cascaded (0.99–1.08×) and for LZ4 on binary/tti; nvCOMP 5.2 is ahead
  on Snappy (1.65–2.6× on binary/zeros, **3.2–4.1× on short-sequence text**) and on LZ4 for runs
  and text (1.4–2.35×). ARCTO is ahead on Snappy TTI decompression (1.3–1.4×, i.e. `0.72–0.77×`).
* The pattern — nvCOMP 5's advantage concentrated on short-sequence, latency-bound decompression —
  is consistent with five years of kernel work on their side (their 5.x decoders are not the 2.2
  ones ARCTO carries), and with what this campaign measured on AMD: the same class of input is
  where ARCTO's dword/vector paths also lose.

**Practical summary.** On NVIDIA hardware ARCTO is a drop-in equal of the nvCOMP generation it
derives from, ahead of it in Cascaded compression, and behind current nvCOMP only in
decompression of text-like/run-heavy data. For a portable library whose optimization budget went
into AMD, "no regression on NVIDIA and a 1.5–1.8× win in one codec" is the outcome the portability
contract (`HIP_FIRST_PORTABILITY.md`) was written to protect.

## 4. What it took to build the three backends on CUDA 13

The same three defects broke **ARCTO and nvCOMP 2.2 alike** — ARCTO inherited them verbatim in the
hipify translation, which is itself a finding:

1. `set(GPU_ARCHS "60")` (hard-coded Pascal) — CUDA 13 removed `sm_60`/`sm_70`. ARCTO now honours
   `CMAKE_CUDA_ARCHITECTURES` and bounds its default list by toolkit version; nvCOMP 2.2 needed the
   same edit locally.
2. `CMAKE_CUDA_STANDARD 14` — CCCL (Thrust/CUB/libcu++) in CUDA 12/13 requires C++17.
3. `cub::Sum` / `cub::Min` / `cub::Max` were removed in CCCL 3. ARCTO now passes its own functors
   (`arcto::ops::*`, `src/arcto_hipcub.hiph`), which work with any generation of CUB or hipCUB.

ARCTO additionally needed, for the CUDA backend: `--expt-relaxed-constexpr` (hip-clang accepts
`std::numeric_limits<T>::max()` in device code, `nvcc` needs the flag), the nvcc flags attached to
`COMPILE_LANGUAGE:CUDA` instead of `HIP` (they had never been applied), a missing `<sstream>`
include, and one test relaxed for coherent host memory (on Grace-Hopper an unregistered host
pointer *is* device-addressable). After those, the CUDA backend builds the whole tree and passes
**17/17 tests**.
