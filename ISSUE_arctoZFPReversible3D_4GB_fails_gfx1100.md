# `arctoZFPReversible3D` produces non-lossless reconstruction at ≥4 GB input on gfx1100

## Summary

`arctoZFPReversible3D` (the GPU-native ZFP reversible path) **fails to
produce bit-exact reconstruction** for float32 inputs at or above 4 GiB
on `gfx1100` (RX 7900 XT, 20 GiB VRAM). Smaller inputs work correctly.
The bug is **size-threshold-bound and reproducible**, and is independent
of array shape and host-buffer placement.

## Affected entry point

- `arctoZFPReversible3D` (`include/arcto/zfp_reversible_3d.h`)
- exercised by `benchmark_zfp_single -m reversible`
  (`benchmarks/benchmark_zfp_single.cu`, dispatch around line 256)

## Reproduction

Environment:
- arcto branch: `feature/scaling-instrumentation` (current `HEAD`)
- container: `arcto_gfx1100_v3.sif` (Ubuntu 22.04, ROCm 7.0.1)
- GPU: RX 7900 XT (gfx1100, 20 GiB GDDR6)
- input: middle-extracted TTI seismic slices, contiguous float32

Command (sweep harness `sweep_canonical.sh`, Block B):

```bash
singularity exec --rocm arcto_gfx1100_v3.sif bash -c \
  "LD_LIBRARY_PATH=/ssd/cakunas/arcto/build_canon/lib:\$LD_LIBRARY_PATH \
   /ssd/cakunas/arcto/build_canon/bin/benchmark_zfp_single \
     -f /ssd/cakunas/testdata/canonical/tti_<SIZE>.bin \
     -3 <NX,NY,NZ> -m reversible \
     -c -w 1 -i 3"
```

Observed `max_abs_diff` (L∞ round-trip error) per size:

| input size | shape           | `max_abs_diff` | PSNR (dB) | ratio | verdict      |
|------------|-----------------|----------------|-----------|-------|--------------|
| 10 MiB     | 128 × 128 × 160 | 7.28e-12       | 211       | 1.50× | bit-exact    |
| 100 MiB    | 256 × 256 × 400 | 7.28e-12       | 203       | 0.99× | bit-exact    |
| 1 GiB      | 512 × 512 × 1024| 7.28e-12       | 205       | 0.96× | bit-exact    |
| **4 GiB**  | **1024 × 1024 × 1024** | **1.28**  | **8.11** | **18.83×** | **broken** |

`max_abs_diff = 7.28e-12` at the small sizes is float32 round-off, i.e.
effectively zero — reversible is working there. At 4 GiB the error
jumps to 1.28 (on data with amplitude range ≈ 0.72), so the
reconstruction is essentially noise. The inflated ratio (18.83× for
"lossless") suggests the encoder is processing degraded input and
compressing aggressively without being able to reverse it.

## Causes ruled out

| hypothesis | how ruled out |
|---|---|
| corrupt input data | the first 1 GiB of `tti_4gb.bin` is bit-equal to `tti_1gb.bin` (which works); shared `head_md5`; source is the correct 72 GiB `TTI.rsf@`, extracted timestep-aligned at offset 100 |
| array-shape sensitivity | repro at `1024,1024,1024` AND at `512,512,4096`; both fail with `max_abs_diff ≈ 1.27` |
| pinned-host destination buffer | flag `-D` (pinned `hipHostMalloc` output buffer) gives identical `max_abs_diff = 1.28`; not buffer-related |
| `-D` ZFP shape mismatch | the input file is exactly `4 * 4 GiB` bytes = `1.07 * 10^9` float32; reshape to 1024³ is exact and aligned to ZFP's 4×4×4 block boundary |

The failure correlates **only with input size**, between 1 GiB and 4 GiB.

## Suspected cause

Likely a size-threshold bug in `arctoZFPReversible3D`. Candidate root
causes worth investigating (in priority order):

1. **Offset arithmetic overflowing `int32`** in block-index or
   chunk-index computation. 4 GiB / 4 B = 2³⁰ floats; 16.8 M blocks
   × `int` accumulator counters can overflow if any intermediate
   product crosses 2³¹.
2. **Working-memory allocation silently truncating** at the
   GDDR6 ceiling. RX 7900 XT has 20 GiB; 4 GiB input + 2-3× scratch
   may approach the cap and trigger a fallback path that does not
   preserve bit-exactness.
3. **Block-count overflow in the metadata header** that ZFP writes at
   the start of the compressed stream.

## Impact

- Lossless ZFP path is unusable above 1 GiB on gfx1100 in current `HEAD`.
- The fixed-rate, fixed-precision, and fixed-accuracy modes are
  **not affected** at any input size.
- Byte-level codecs (LZ4, Snappy, Cascaded) via the chunked template
  are **not affected**.
- Other architectures (gfx906, gfx90a, gfx942) **not yet tested**;
  see follow-up below.

## Follow-up

- [ ] Reproduce on MI50 (gfx906, 32 GiB), MI210 (gfx90a, 64 GiB),
      MI300X (gfx942, 192 GiB) to determine whether the bug is
      gfx1100-specific or cross-arch.
- [ ] Add an int-width audit pass over `src/zfp/zfp_reversible_3d.*`
      and any helpers it calls; promote candidates to `size_t` /
      `int64_t`.
- [ ] Add a stress test in `tests/` that runs reversible at
      ≥ 4 GiB input with a synthetic float pattern, asserting
      `max_abs_diff < 1e-6`.
