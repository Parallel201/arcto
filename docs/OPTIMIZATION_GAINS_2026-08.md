# ARCTO on AMD — gains summary (August 2026)

Companion to `OPTIMIZATION_RESULTS_2026-08.md` (results by category) and `AMD_OPTIMIZATION_MAP.md`
(the plan). This document is the *numbers-only* view the write-up needs: what each optimization
bought on its own, what each branch bought in total, and what the shipped heads deliver against
`main` — plus the comparison with the LZ4 branch that already carried gains before this campaign.

## 0. How to read the tables

*Branch names in the section headings are the lineages as they were developed; they are all merged
into `opt/aggregate` now (see `INTEGRATION_AND_CLEANUP_2026-08.md` §1). `bench/baseline-2026-08` is
the tag `baseline/post-hipify-2026-08`.*

* Every figure is a ratio of **medians of 30 repetitions** (per-repetition CSV of
  `benchmark_*_chunked -p 65536`), **later ÷ earlier**, so ×1.10 = +10 % throughput. Δ tables
  compare each commit with its predecessor *in the same sweep* (rows marked † join two sweeps of
  the same node when the predecessor was only measured in the earlier one).
* `sat` = saturation (`-x 512`, or `-x 32` for the 16 MB text-like file); `x0` = one 1 MB file,
  16 chunks (latency regime). Δ cells show the geometric mean over the inputs with the min–max in
  brackets; end-to-end tables are per input.
* Inputs: Snappy/LZ4 `synth_binary` (copy-heavy binary), `tti` (seismic floats, literal-heavy),
  `words` (short-sequence text), LZ4 also `synth_zeros` (long runs); Cascaded `ints_mixed`
  (runs + ramps + small-range noise, 32-bit), `synth_zeros`, `tti`.
* Nodes: gfx1100 = RX 7900 XT (RDNA3, wave32), gfx942 = MI300A (CDNA3, wave64), ROCm 7.0.1.
  On the MI300A, saturation medians of the *same binary* moved ±5–30 % between runs on the shared
  node; the x0 rows there are the dependable ones. "Bytes identical" held for every kept commit
  (exact-bytes ladder + round-trip tests).
* "Product of increments" multiplies the per-commit geometric means; "end-to-end" is the head
  measured against its base in one sweep. They agree to within the drift noted above.

## 1. Snappy (`opt/snappy-2026-08`; base = `main`)

#### Snappy decompression lineage — decompression, Δ vs previous commit (geometric mean over inputs, min–max); inputs: synth_binary, tti, words

| ID | change | gfx1100 sat | gfx1100 x0 | gfx942 sat | gfx942 x0 |
|---|---|---|---|---|---|
| SNP-D1 | s_sleep in spin-waits | ×1.00 (1.00–1.01) | ×1.00 (1.00–1.01) | ×1.01 (0.97–1.02) | ×1.00 (0.99–1.01) |
| SNP-D4 | uniform state on all lanes | ×1.00 (0.99–1.00) | ×1.00 (1.00–1.00) | ×1.01 (0.99–1.02) | ×1.00 (1.00–1.00) |
| SNP-D3 | readfirstlane/readlane broadcasts | ×1.04 (1.03–1.05) | ×1.06 (1.04–1.07) | ×1.04 (1.02–1.07) | ×1.07 (1.04–1.10) |
| SNP-D5 | 64-bit symbol queue | ×1.01 (1.01–1.02) | ×1.01 (1.01–1.03) | ×0.97 (0.94–0.99) | ×0.99 (0.96–1.04) |
| SNP-D7a | dword window ring reads | ×1.00 (1.00–1.01) | ×1.00 (1.00–1.00) | ×1.01 (1.00–1.01) | ×1.00 (1.00–1.00) |
| SNP-D7b | per-lane symbol window (†) | ×1.00 (0.99–1.00) | ×1.00 (1.00–1.01) | ×1.00 (0.98–1.03) | ×1.00 (1.00–1.01) |
| SNP-D2 | rocPRIM DPP scans | ×1.02 (1.01–1.03) | ×1.03 (1.02–1.04) | ×1.01 (0.99–1.05) | ×1.05 (1.00–1.09) |
| SNP-D6 | dword prefetch | ×1.03 (1.01–1.06) | ×1.00 (1.00–1.00) | ×1.01 (1.00–1.02) | ×1.00 (0.99–1.00) |
| SNP-D10 | one chunk/thread size kernel | ×1.00 (0.99–1.01) | ×1.00 (1.00–1.00) | ×1.01 (1.00–1.02) | ×1.00 (1.00–1.00) |
| SNP-D8 | dword literals+copies | ×1.09 (0.98–1.17) | ×1.07 (0.98–1.14) | ×0.94 (0.80–1.06) | ×0.95 (0.78–1.13) |
| SNP-D8s+D11s | split copies→wave32; LITERAL_SECTORS 8 on wave32 (†) | ×1.02 (1.00–1.05) | ×1.03 (1.00–1.10) | ×1.08 (1.00–1.20) | ×1.11 (0.99–1.31) |
| **product of increments** | | ×1.23 | ×1.24 | ×1.09 | ×1.18 |

#### Snappy compression lineage — compression, Δ vs previous commit (geometric mean over inputs, min–max); inputs: synth_binary, tti, words

| ID | change | gfx1100 sat | gfx1100 x0 | gfx942 sat | gfx942 x0 |
|---|---|---|---|---|---|
| SNP-C5 | 16-B hash clear (fix) | ×1.00 (1.00–1.00) | ×1.00 (1.00–1.00) | ×1.01 (1.00–1.02) | ×1.01 (0.99–1.03) |
| SNP-C2 | one barrier/pair (reverted) | ×0.98 (0.97–1.00) | ×0.97 (0.96–0.99) | ×0.97 (0.94–1.00) | ×0.96 (0.92–1.00) |
| SNP-C1 | prefetch match word (reverted) | ×0.96 (0.93–0.99) | ×0.96 (0.93–1.00) | ×0.94 (0.93–0.96) | ×0.95 (0.93–0.95) |
| SNP-C3 | dword literal store (reverted) | ×1.00 (1.00–1.01) | ×1.00 (1.00–1.00) | ×0.99 (0.99–1.00) | ×0.98 (0.95–1.00) |
| **product of increments** | | ×0.95 | ×0.93 | ×0.92 | ×0.91 |

Compression: C5 is kept (correctness + ≈ +1–3 % on wave64); C2, C1, C3 were reverted, so the
branch's compression delta is C5 alone (×1.00 gfx1100, ×1.01–1.02 gfx942), not the product above.

#### Snappy (integration head eb4024f vs main ced8428) — end-to-end (head vs base, same sweep), per input

| input | regime | gfx1100 comp | gfx1100 decomp | gfx942 comp | gfx942 decomp |
|---|---|---|---|---|---|
| synth_binary | sat | ×1.00 | ×1.24 | ×1.01 | ×1.08 |
| synth_binary | x0 | ×1.00 | ×1.23 | ×1.02 | ×1.12 |
| tti | sat | ×1.00 | ×1.39 | ×0.99 | ×0.99 |
| tti | x0 | ×1.00 | ×1.39 | ×0.99 | ×1.23 |
| words | sat | ×1.00 | ×1.07 | ×1.02 | ×1.15 |
| words | x0 | ×1.00 | ×1.10 | ×1.02 | ×1.21 |

**Total, Snappy branch vs `main`:** decompression gfx1100 ×1.24 / ×1.39 / ×1.07 (binary / tti /
words, saturation = small batch), gfx942 ×1.08 / ≈1.00 / ×1.15 at saturation and ×1.12 / ×1.23 /
×1.21 at small batch; compression ×1.00 gfx1100, ×1.01–1.02 gfx942.

## 2. Cascaded (`opt/cascaded-2026-08`; base `777135f` = `main` kernels + tests/benchmark fixes)

#### Cascaded lineage — compression, Δ vs previous commit (geometric mean over inputs, min–max); inputs: ints_mixed, synth_zeros, tti

| ID | change | gfx1100 sat | gfx1100 x0 | gfx942 sat | gfx942 x0 |
|---|---|---|---|---|---|
| CAS-S2 | launch bounds | ×1.00 (1.00–1.00) | ×1.00 (1.00–1.00) | ×0.97 (0.95–1.01) | ×1.00 (1.00–1.01) |
| CAS-S1 | 256-thread blocks (wave64) | ×1.00 (1.00–1.00) | ×1.00 (1.00–1.00) | ×1.45 (1.27–1.64) | ×1.43 (1.18–1.67) |
| CAS-C1 | single-pass min/max | ×1.40 (1.05–1.70) | ×1.41 (1.06–1.71) | ×1.37 (1.14–1.57) | ×1.41 (1.16–1.69) |
| CAS-S4 | BLOCK_SCAN_WARP_SCANS | ×1.00 (1.00–1.01) | ×1.00 (1.00–1.01) | ×1.17 (1.16–1.19) | ×1.01 (1.00–1.02) |
| CAS-D5 | LDS shrink (count scratch alias) | ×1.00 (1.00–1.00) | ×1.00 (1.00–1.01) | ×1.00 (1.00–1.00) | ×0.99 (0.99–1.00) |
| CAS-D2 | multi-item delta scan | ×1.00 (1.00–1.00) | ×1.00 (0.99–1.00) | ×1.00 (1.00–1.00) | ×1.00 (1.00–1.02) |
| CAS-D1 | balanced RLE fill (block vote) | ×1.00 (1.00–1.00) | ×0.97 (0.90–1.01) | ×1.00 (1.00–1.00) | ×1.00 (1.00–1.00) |
| CAS-C4 | incremental bitpack indices | ×1.00 (1.00–1.01) | ×1.00 (1.00–1.01) | ×1.00 (0.99–1.01) | ×1.00 (0.99–1.01) |
| CAS-C2 | RLE slab in registers | ×1.13 (1.10–1.15) | ×0.95 (0.78–1.19) | ×1.14 (1.08–1.17) | ×1.16 (1.10–1.22) |
| CAS-S6 | 32-bit RLE scan | ×1.01 (1.01–1.02) | ×1.07 (0.99–1.25) | ×1.03 (1.02–1.04) | ×1.03 (1.02–1.04) |
| CAS-S5 | 16-B copies (all sites) | ×1.26 (1.18–1.41) | ×1.07 (1.01–1.16) | ×1.16 (1.11–1.22) | ×1.06 (1.05–1.07) |
| CAS-H1 | deterministic padding | ×0.99 (0.98–1.00) | ×0.96 (0.91–0.98) | ×0.96 (0.95–0.97) | ×0.96 (0.95–0.97) |
| CAS-D1w+H1b | wave-local RLE fill; cheaper zeroing | ×1.00 (1.00–1.01) | ×0.99 (0.82–1.18) | ×1.02 (1.01–1.02) | ×1.01 (1.01–1.02) |
| **product of increments** | | ×2.02 | ×1.42 | ×3.00 | ×2.52 |

#### Cascaded lineage — decompression, Δ vs previous commit (geometric mean over inputs, min–max); inputs: ints_mixed, synth_zeros, tti

| ID | change | gfx1100 sat | gfx1100 x0 | gfx942 sat | gfx942 x0 |
|---|---|---|---|---|---|
| CAS-S2 | launch bounds | ×1.00 (1.00–1.01) | ×1.00 (1.00–1.00) | ×0.97 (0.93–1.00) | ×1.03 (0.94–1.11) |
| CAS-S1 | 256-thread blocks (wave64) | ×1.00 (0.99–1.00) | ×1.00 (1.00–1.00) | ×1.19 (1.08–1.27) | ×1.30 (1.00–1.53) |
| CAS-C1 | single-pass min/max | ×1.00 (1.00–1.00) | ×1.00 (0.98–1.00) | ×1.01 (0.97–1.07) | ×1.00 (0.96–1.05) |
| CAS-S4 | BLOCK_SCAN_WARP_SCANS | ×1.01 (1.00–1.03) | ×1.01 (1.00–1.03) | ×1.08 (1.00–1.15) | ×0.97 (0.87–1.03) |
| CAS-D5 | LDS shrink (count scratch alias) | ×1.09 (0.98–1.17) | ×1.00 (1.00–1.00) | ×1.03 (0.99–1.12) | ×1.00 (0.90–1.13) |
| CAS-D2 | multi-item delta scan | ×1.02 (0.98–1.09) | ×1.02 (1.00–1.05) | ×1.04 (1.01–1.09) | ×1.04 (1.01–1.08) |
| CAS-D1 | balanced RLE fill (block vote) | ×1.86 (0.93–6.84) | ×1.88 (1.00–6.67) | ×1.37 (0.91–2.87) | ×1.59 (0.95–4.32) |
| CAS-C4 | incremental bitpack indices | ×1.00 (0.99–1.01) | ×1.01 (1.00–1.03) | ×0.97 (0.95–0.99) | ×0.96 (0.89–1.01) |
| CAS-C2 | RLE slab in registers | ×1.01 (1.00–1.02) | ×0.87 (0.68–1.02) | ×1.01 (1.01–1.02) | ×1.08 (1.00–1.21) |
| CAS-S6 | 32-bit RLE scan | ×1.00 (0.98–1.00) | ×0.99 (0.98–1.02) | ×0.97 (0.93–0.99) | ×0.98 (0.96–0.99) |
| CAS-S5 | 16-B copies (all sites) | ×0.99 (0.98–1.00) | ×0.92 (0.78–0.99) | ×0.95 (0.87–1.04) | ×0.99 (0.96–1.06) |
| CAS-H1 | deterministic padding | ×1.00 (0.99–1.00) | ×0.97 (0.89–1.03) | ×0.91 (0.77–1.00) | ×0.98 (0.96–1.02) |
| CAS-D1w+H1b | wave-local RLE fill; cheaper zeroing | ×0.97 (0.85–1.08) | ×0.91 (0.79–1.09) | ×1.00 (0.91–1.10) | ×0.97 (0.86–1.06) |
| **product of increments** | | ×2.02 | ×1.37 | ×1.52 | ×2.02 |

Notes: CAS-S1 is a wave64-only change (no-op on gfx1100); CAS-D1's decompression cell averages
one ×6.8 (zeros) with two ≈ 1.0 inputs; the x0 `synth_zeros` compression rows on gfx1100 are
bimodal (8–14 GB/s) from CAS-D1 on (the decompression phase got 7× shorter and the 1 MB
compression timing sees clock ramping) — read zeros at saturation. CAS-H1/H1b (byte-determinism)
costs ≈ −1…−4 % compression; CAS-H2/H3 are correctness fixes with no measurable cost.

#### Cascaded (head ddf1fd0 vs base 777135f) — end-to-end (head vs base, same sweep), per input

| input | regime | gfx1100 comp | gfx1100 decomp | gfx942 comp | gfx942 decomp |
|---|---|---|---|---|---|
| ints_mixed | sat | ×2.16 | ×1.28 | ×3.17 | ×1.43 |
| ints_mixed | x0 | ×2.07 | ×1.16 | ×2.60 | ×1.40 |
| synth_zeros | sat | ×1.61 | ×6.43 | ×2.10 | ×2.49 |
| synth_zeros | x0 | ×0.91 | ×4.32 | ×1.60 | ×3.86 |
| tti | sat | ×2.32 | ×0.98 | ×3.82 | ×0.90 |
| tti | x0 | ×2.26 | ×0.84 | ×3.58 | ×1.54 |

**Total, Cascaded branch vs base:** compression gfx1100 ×2.1–2.3 (ints/tti), ×1.6 zeros; gfx942
×2.6–3.8 (ints/tti), ×1.6–2.1 zeros. Decompression gfx1100 ints ×1.16–1.28, zeros ×4.3–6.4, tti
≈ ×1.0; gfx942 ints ×1.40–1.43, zeros ×2.5–3.9, tti ×1.54 at small batch (saturation in the noise
band).

## 3. LZ4 decompression on wave64 (`opt/lz4-decomp-wave64-2026-08`; base = `origin/opt/curated`)

#### LZ4 wave64 decompression lineage — decompression, Δ vs previous commit (geometric mean over inputs, min–max); inputs: synth_binary, tti, words, synth_zeros

| ID | change | gfx1100 sat | gfx1100 x0 | gfx942 sat | gfx942 x0 |
|---|---|---|---|---|---|
| LZ4-D7 | launch bounds | ×0.99 (0.97–1.00) | ×0.92 (0.79–1.00) | ×1.01 (1.00–1.03) | ×1.00 (0.99–1.00) |
| LZ4-D2 | incremental repeat index | ×1.00 (1.00–1.00) | ×1.00 (1.00–1.00) | ×0.89 (0.83–0.99) | ×0.99 (0.98–1.01) |
| LZ4-D5 | readfirstlane parsing (reverted) (†) | ×1.01 (1.00–1.03) | ×0.99 (0.95–1.01) | ×0.97 (0.94–1.00) | ×0.97 (0.92–0.99) |
| LZ4-D1w | vec copies on wave64 (decomp side) | ×1.00 (0.98–1.02) | ×1.00 (1.00–1.01) | ×1.08 (0.77–1.46) | ×1.42 (0.83–2.18) |
| **product of increments** | | ×1.00 | ×0.92 | ×0.94 | ×1.36 |

LZ4-D1w's cell averages two ×2.2 inputs with words ×0.8 and zeros ≈ ×1.0 at small batch (and the
same pattern at saturation) — the per-input rows below are the meaningful ones. LZ4-D5 was
reverted; D7 is neutral and kept as documentation of the real block size; D2 is exact and kept.

#### LZ4 (head 34d036a vs curated base 433772a) — end-to-end (head vs base, same sweep), per input

| input | regime | gfx1100 comp | gfx1100 decomp | gfx942 comp | gfx942 decomp |
|---|---|---|---|---|---|
| synth_binary | sat | ×1.00 | ×1.00 | ×1.00 | ×1.35 |
| synth_binary | x0 | ×1.00 | ×1.00 | ×1.00 | ×2.28 |
| tti | sat | ×1.00 | ×1.01 | ×1.00 | ×1.12 |
| tti | x0 | ×1.00 | ×1.00 | ×1.00 | ×2.20 |
| words | sat | ×0.99 | ×1.00 | ×1.00 | ×0.75 |
| words | x0 | ×1.00 | ×1.00 | ×1.00 | ×0.80 |
| synth_zeros | sat | ×1.00 | ×1.00 | ×1.00 | ×1.37 |
| synth_zeros | x0 | ×1.00 | ×1.00 | ×1.01 | ×1.17 |

### LZ4 — comparison with the previous branch (`origin/opt/curated`) and accumulated vs `main`

Measured in one sweep per node (`lz4prior`, 30 reps): `main` (+ the per-repetition CSV in the benchmark, code identical to `main`), `opt/curated` (your E17 lineage + the two benchmark fixes), and the wave64 decompression head. Ratios are later ÷ earlier; the last two columns are the accumulated result of both branches against `main`.

| node | input | regime | curated vs main comp | curated vs main decomp | head vs curated comp | head vs curated decomp | **head vs main comp** | **head vs main decomp** |
|---|---|---|---|---|---|---|---|---|
| gfx1100 | binary | sat | ×3.35 | ×1.08 | ×1.00 | ×1.00 | ×3.35 | ×1.08 |
| gfx1100 | binary | x0 | ×3.18 | ×1.53 | ×1.00 | ×1.00 | ×3.18 | ×1.53 |
| gfx1100 | tti | sat | ×4.72 | ×1.26 | ×1.00 | ×1.00 | ×4.72 | ×1.26 |
| gfx1100 | tti | x0 | ×3.16 | ×1.87 | ×1.00 | ×1.00 | ×3.16 | ×1.87 |
| gfx1100 | words | sat | ×1.44 | ×0.71 | ×1.00 | ×0.99 | ×1.44 | ×0.70 |
| gfx1100 | words | x0 | ×1.59 | ×0.72 | ×1.00 | ×1.00 | ×1.58 | ×0.72 |
| gfx1100 | zeros | sat | ×1.17 | ×0.83 | ×1.00 | ×1.01 | ×1.17 | ×0.84 |
| gfx1100 | zeros | x0 | ×1.01 | ×3.05 | ×1.00 | ×1.00 | ×1.01 | ×3.06 |
| gfx942 | binary | sat | ×11.10 | ×1.02 | ×1.00 | ×1.27 | ×11.14 | ×1.30 |
| gfx942 | binary | x0 | ×12.31 | ×0.99 | ×1.00 | ×2.25 | ×12.29 | ×2.23 |
| gfx942 | tti | sat | ×13.54 | ×0.85 | ×1.00 | ×1.36 | ×13.50 | ×1.16 |
| gfx942 | tti | x0 | ×12.32 | ×1.04 | ×1.00 | ×2.19 | ×12.32 | ×2.27 |
| gfx942 | words | sat | ×4.00 | ×1.04 | ×1.00 | ×0.75 | ×4.01 | ×0.78 |
| gfx942 | words | x0 | ×4.44 | ×0.97 | ×1.00 | ×0.81 | ×4.45 | ×0.78 |
| gfx942 | zeros | sat | ×1.14 | ×0.75 | ×1.01 | ×1.01 | ×1.15 | ×0.76 |
| gfx942 | zeros | x0 | ×1.06 | ×1.01 | ×1.00 | ×1.17 | ×1.07 | ×1.18 |

What the curated lineage recorded in its own commits (your measurements, different files/shapes,
quoted for the comparison): E11 LDS-resident hash table +25 % gfx90a / +9 % gfx1100 / +300 % gfx906
over the packed global table, and with the small-table default 5.1× / 31.4× / 17.9× compression
over the inherited configuration on gfx90a / gfx1100 / gfx906; df13f11 LDS claim-table twin search
2.4× compression at 64 KB chunks on MI300X; 2e2b74e hot-loop rework +16 % gfx1100; e3293a1 64-VGPR
pin +5.8 % MI300X; 61218f5 wave32 copy vectorisation / doubling repeat / warp LSIC: decompression
146 → 151–159 GB/s and zeros +20 % on gfx1100 (wave64 kept the original loops: every variant tried
then cost 5–20 % on MI300X); 74ebdd7 (E17) wave32 gating ≈ +10.6 % gfx1100. The table above confirms
the compression side on this protocol (×3.2–4.7 gfx1100, ×11–13.5 MI300A on binary/TTI; ×1.4–4.4
on words; zeros ×1.06–1.17) and shows where that branch left decompression: +8…+87 % on gfx1100 with
a −28 % cost on short-sequence text (the wave32 vectorisation — the same trade this campaign met on
wave64), ≈ unchanged on the MI300A with zeros −24 % at saturation.

**Where the two branches meet.** The curated lineage bought compression; this campaign's LZ4
branch bought wave64 decompression on top of it (×2.2 binary/TTI at small batch, ×1.2–1.3 at
saturation, zeros +18 % at small batch) without touching compression, so the accumulated
`head vs main` column is the product of the two: on the MI300A, compression ×11–13.5 *and*
decompression ×2.2 / ×1.2–1.3 on binary/TTI, words decompression ×0.78 (the register-growth cost
documented in LZ4-D1s), zeros ×1.18 at small batch / ×0.76 at saturation (curated's saturation
loss on long runs is not recovered by the doubling repeat on wave64 — an open item).


## 4. ZFP host path (`opt/zfp-host-2026-08` + vendored-zfp branch; base = `main` ZFP wrapper + LLNL `cccbb9d`)

#### ZFP host path — call time vs base (median of 3 runs × 30 iterations; × = time ratio, < 1 is faster)

| step | gfx1100 64³ rate | gfx1100 64³ prec | gfx1100 256³ rate | gfx1100 256³ prec | gfx942 64³ rate | gfx942 64³ prec | gfx942 256³ rate | gfx942 256³ prec |
|---|---|---|---|---|---|---|---|---|
| base | c ×1.00 / d ×1.00 | c ×1.00 / d ×1.00 | c ×1.00 / d ×1.00 | c ×1.00 / d ×1.00 | c ×1.00 / d ×1.00 | c ×1.00 / d ×1.00 | c ×1.00 / d ×1.00 | c ×1.00 / d ×1.00 |
| +T1 test (no code) | c ×1.10 / d ×1.06 | c ×1.08 / d ×1.07 | c ×1.04 / d ×1.03 | c ×1.00 / d ×1.00 | c ×1.22 / d ×1.22 | c ×0.97 / d ×0.98 | c ×0.74 / d ×0.77 | c ×1.00 / d ×0.99 |
| +H1a no exec policy on throwaway streams | c ×0.74 / d ×1.13 | c ×0.89 / d ×1.13 | c ×0.98 / d ×1.03 | c ×0.93 / d ×1.01 | c ×0.95 / d ×1.27 | c ×0.79 / d ×1.01 | c ×0.96 / d ×0.99 | c ×1.26 / d ×1.20 |
| +H1b warm-up once, H2a async pool allocs, D4 no offset round trip | c ×0.30 / d ×0.48 | c ×0.51 / d ×0.49 | c ×0.82 / d ×0.91 | c ×0.91 / d ×1.20 | c ×0.59 / d ×0.73 | c ×0.66 / d ×0.75 | c ×0.22 / d ×0.27 | c ×0.83 / d ×0.44 |

(c = compress call, d = decompress call; base 64³ ≈ 0.46–0.74 ms compress / 0.33–0.41 ms decompress; 256³ 2.6–3.8 ms / 0.9–3.2 ms)


The T1 row (no code change) shows this benchmark's run-to-run band (±10–25 %); H1a alone is
inside it; the last row is the branch total: 64³ calls ×2–3.3 faster on both parts, 256³ fixed-rate
×3.6–4.6 on the APU (no PCIe copies, so the HIP API calls were the whole cost) and ×1.1–1.2 on the
discrete RX 7900 XT. ZFP-T1 (308 byte-exact cases) held at every step on both nodes.

## 5. One-line totals (shipped heads vs `main`)

| Branch | What | gfx1100 compression | gfx1100 decompression | gfx942 compression | gfx942 decompression |
|---|---|---|---|---|---|
| `opt/snappy-2026-08` | 10 kept decompression items + C5 | ×1.00 | ×1.07–1.39 | ×1.01–1.02 | ×1.08–1.23 (x0), ≈1.0–1.15 (sat) |
| `opt/cascaded-2026-08` | 13 kept items + 3 correctness fixes | ×1.6–2.3 | ×1.0 (tti) – ×6.4 (zeros); ints ×1.16–1.28 | ×1.6–3.8 | ×1.4 (ints) – ×3.9 (zeros); tti ×1.5 (x0) |
| `opt/lz4-decomp-wave64-2026-08` | decompression only, over `opt/curated` | = curated | = curated (identical) | = curated | binary ×2.3 / ×1.35, tti ×2.2 / ×1.1, zeros ×1.2 / ×1.4, words ×0.8 / ×0.75 (x0 / sat) |
| `opt/zfp-host-2026-08` (+ fork branch) | host path | call time ×0.30–0.91 | ×0.48–0.91 (×1.2 on one noisy row) | ×0.22–0.83 | ×0.27–0.75 |


Accumulated LZ4 vs `main` (curated × this branch): gfx1100 compression ×3.2–4.7 (binary/TTI), ×1.4–1.6
words, ×1.17 zeros; decompression ×1.08–1.87 binary/TTI, ×0.71 words, zeros ×3.1 (x0) / ×0.84 (sat) —
all from curated (this branch is configuration-identical on wave32). gfx942 compression ×11–13.5
binary/TTI, ×4.0–4.4 words, ×1.1–1.15 zeros (curated); decompression binary ×2.22 (x0) / ×1.30
(sat), TTI ×2.27 / ×1.16, zeros ×1.18 / ×0.76, words ×0.78 / ×0.78 (this branch × curated).

## 6. Final: the integrated tree vs `main` as it came out of hipify

Everything above measures one branch against its own base. This section measures **the shipped
tree** — all optimization branches merged plus the cleanup pass (`chore/cleanup-2026-08`) — against
`bench/baseline-2026-08`, which is `main` with *only* the benchmark instrumentation added (the
per-repetition CSV and the `-x` duplication fix; no library change whatsoever). Same protocol as
§0, one sweep per codec per GPU, 30 repetitions, exact-bytes ladder identical between the two
commits and every test green on both parts.

Ratios are head ÷ baseline, so **> 1 is faster**.

| codec | input | regime | gfx1100 comp | gfx1100 decomp | gfx942 comp | gfx942 decomp |
|---|---|---|---:|---:|---:|---:|
| lz4 | binary | small batch | ×3.19 | ×1.53 | ×12.30 | ×2.22 |
| lz4 | binary | saturation | ×3.36 | ×1.00 | ×11.08 | ×1.50 |
| lz4 | tti | small batch | ×3.16 | ×1.87 | ×12.32 | ×2.24 |
| lz4 | tti | saturation | ×4.84 | ×1.27 | ×13.37 | ×1.32 |
| lz4 | words | small batch | ×1.59 | ×0.72 | ×4.45 | ×0.81 |
| lz4 | words | saturation | ×1.43 | ×0.68 | ×3.65 | ×0.74 |
| lz4 | zeros | small batch | ×1.01 | ×3.06 | ×1.07 | ×1.16 |
| lz4 | zeros | saturation | ×1.16 | ×0.81 | ×1.11 | ×1.06 |
| | | | | | | |
| snappy | binary | small batch | ×1.00 | ×1.23 | ×1.02 | ×1.12 |
| snappy | binary | saturation | ×1.00 | ×1.24 | ×1.01 | ×1.02 |
| snappy | tti | small batch | ×0.98 | ×1.41 | ×0.99 | ×1.22 |
| snappy | tti | saturation | ×1.00 | ×1.37 | ×1.00 | ×1.11 |
| snappy | words | small batch | ×1.00 | ×1.10 | ×1.03 | ×1.18 |
| snappy | words | saturation | ×1.00 | ×1.08 | ×1.02 | ×1.16 |
| | | | | | | |
| cascaded | ints | small batch | ×2.37 | ×1.30 | ×2.63 | ×1.36 |
| cascaded | ints | saturation | ×2.20 | ×1.26 | ×3.19 | ×1.39 |
| cascaded | zeros | small batch | ×1.03 | ×4.38 | ×1.60 | ×3.74 |
| cascaded | zeros | saturation | ×1.65 | ×6.21 | ×2.10 | ×2.55 |
| cascaded | tti | small batch | ×2.06 | ×0.83 | ×3.63 | ×1.45 |
| cascaded | tti | saturation | ×2.38 | ×0.98 | ×3.83 | ×1.49 |
| | | | | | | |

ZFP (call time, lower is better; ratio = baseline ÷ head, so > 1 is faster):

| shape | mode | gfx1100 compress | gfx1100 decompress | gfx942 compress | gfx942 decompress |
|---|---|---:|---:|---:|---:|
| 256³ | fixed_rate | ×1.23 | ×1.09 | ×3.43 | ×2.86 |
| 256³ | fixed_precision | ×1.09 | ×0.82 | ×1.50 | ×2.70 |
| 64³ | fixed_rate | ×3.28 | ×2.12 | ×1.65 | ×1.39 |
| 64³ | fixed_precision | ×1.96 | ×2.01 | ×1.53 | ×1.35 |

### Reading it

* **Cascaded** is the largest win on both parts — compression ×2.2–2.4 (gfx1100) and ×2.6–3.8
  (gfx942), decompression ×1.26–1.30 (ints) and ×4.4–6.2 (zeros) on gfx1100, ×1.36–1.49 and
  ×2.5–3.7 on gfx942. The two levers are algorithmic (single-pass min/max, register-resident RLE
  slab, multi-item scans, wave-local run expansion) and structural (256-thread blocks and the LDS
  shrink on CDNA).
* **LZ4** compression carries the earlier `opt/curated` lineage: ×3.2–4.8 on gfx1100 and
  **×11–13.4** on gfx942 for binary/TTI. Decompression: ×1.5–1.9 (gfx1100 small batch) and
  ×2.2 / ×1.3–1.5 (gfx942) on binary/TTI from this campaign's wave64 work; short-sequence text and
  long runs lose (gfx1100 words ×0.68–0.72, zeros at saturation ×0.81 — from the curated wave32
  copy vectorisation; gfx942 words ×0.74–0.81 — from this campaign's wave64 one). Both are the same
  documented trade and both are one build flag away (`ARCTO_LZ4_VEC_COPY_DECOMP=0`).
* **Snappy** decompression ×1.10–1.41 (gfx1100) and ×1.03–1.22 (gfx942), compression neutral to
  +2 %.
* **ZFP** is a host-path result: per-call time ×1.35–3.4 faster, most of it on the APU where the
  HIP API calls were the whole cost of a small field.
* **TTI decompression on Cascaded/gfx1100** is the one row at ×0.83–0.98: the input compresses to a
  raw fall-back for most chunks, so the decompressor is a copy loop that the optimizations do not
  touch, and the residual is the byte-determinism fix (CAS-H1, −1…−3 %).
