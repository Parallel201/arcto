# LZ4 decompression on wave64 (CDNA) — experiment log (branch `opt/lz4-decomp-wave64-2026-08`)

Base: `origin/opt/curated` (E17 curated LZ4 lineage: LDS hash table, wave32-only copy
vectorisation / hybrid repeat / warp LSIC gated behind `USE_WARPSIZE_32`) + the benchmark per-rep
CSV and the `-x` duplication fix cherry-picked from the chore/test branches. One commit per change;
IDs are the LZ4-D* items of `docs/AMD_OPTIMIZATION_MAP.md`; categories C1..C8 as in the Snappy logs
(C1 sync/scheduling, C2 wave-level primitives, C3 memory access width/cache policy, C4 LDS layout,
C5 occupancy/registers/launch bounds, C6 launch geometry/compile-time tunables, C7 compiler flags,
C8 host side). Gates: `test_lz4`, `test_random_lz4`, `test_lz4batch_c_api`; exact-bytes ladder;
`benchmark_lz4_chunked` on tti / synth_binary / words at saturation and small batch (both nodes).
Baseline resource usage (gfx942, ROCm 7.0.1, from the Snappy branch's report build):
`lz4DecompressBatchKernel` SGPR 74 / VGPR 65 / LDS 1024 B / est. 7 waves per SIMD, no spills.

### LZ4-D7 — `__launch_bounds__` on the batched decompression kernel                    Category: C5   Status: PENDING
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/lowlevel/LZ4CompressionKernels.hip`
Change: `lz4DecompressBatchKernel` carries `__launch_bounds__(LZ4_DECOMP_THREADS_PER_CHUNK *
LZ4_DECOMP_CHUNKS_PER_BLOCK)` (128 threads on wave64, 64 on wave32), and
`-DARCTO_LZ4_DECOMP_MIN_WAVES_PER_EU=N` adds the occupancy target as the second argument (HIP:
min waves per EU; CUDA: min blocks per SM). Default: bounds only, no occupancy target.
Why (mechanism): without bounds hip-clang budgets registers for a 1024-thread workgroup; at 65
VGPRs the kernel sits at 7 waves/SIMD on gfx942 — one VGPR over the 64-register step for 8 waves.
The bound alone lets the compiler schedule for the real block; the flag variant pins the budget
so the re-test of the wave32-only vectorisation (LZ4-D1/D2) on wave64 is not confounded by a
VGPR step (the earlier wave64 losses were suspected to be exactly that).
Prediction: bounds only 0–3 %; `MIN_WAVES_PER_EU=8` 0–8 % if the 65th VGPR was the limiter, a
loss if the compiler spills to reach it (check `-Rpass-analysis`); bytes identical.
Measured: gfx1100 (RX 7900 XT, wave32, 30 reps, lz4a sweep): decompression tti x0 9.73 → 7.69 GB/s
(×0.79), x512 240.1 → 231.8 (×0.965); binary and words ×0.997–1.002; compression ≈ (tti x512
×1.02, within noise). Bytes identical, tests green. gfx942: pending (lz4a_gfx942 sweep).
Result (gfx1100): NEGATIVE — the bare `__launch_bounds__(64)` on the wave32 build changes the
generated code for the worse on the literal-heavy input (−21 % per-chunk latency). Mechanism to
confirm with the resource report (VGPR count / scheduling with the 1..64 flat work-group size vs
the 1..1024 default; RDNA3 wave32 is not the target of this item).
Re-measured (gfx1100, interleaved A/B base/D7/base/D7, 30 reps each): tti x0 7.71 / 7.69 / 7.70 /
7.70 GB/s, x512 232.9 / 231.8 / 232.9 / 231.9; binary and words ×0.995–1.002 — EXACTLY NEUTRAL; the
first sweep's baseline (9.73 at x0) was the outlier, not the bounded build. The register report
confirms identical resources (91 VGPRs, 44 SGPRs, occupancy 16, no spills) with and without bounds.
gfx942 (lz4a sweep): ×0.996–1.004 binary, ×0.992 x0 / ×1.028 x512 tti (noise band), words ×1.00 —
neutral; `MIN_WAVES_PER_EU=8` on top of the branch head: binary ×0.87 and tti ×0.92 at saturation
(x0 ≈) — NEGATIVE (the forced ≤ 64-VGPR budget costs more than the eighth wave buys).
Verdict: bounds KEPT (neutral, documents the real block; the launcher and the HLIF path agree);
occupancy target stays 0 — do not pin the VGPR budget on this kernel.

### LZ4-D2 — incremental source index in the repeat (overlapped-match) copy             Category: C1   Status: PENDING
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/LZ4Kernels.hiph` (`coopCopyRepeat`)
Change: the strided loop `dest[i] = source[i % dist]` (the wave64/CUDA path for every overlapped
match, and the wave32 path for matches ≤ 2 × blockDim) keeps the lane's source index
incrementally: `r = lane % dist`, `step = blockDim.x % dist` once, then `r += step; if (r >= dist)
r -= dist` per iteration. Exact (r, step < dist, so one conditional subtract suffices).
Why (mechanism): a 32-bit remainder by a runtime divisor is ~30 VALU instructions on GCN/CDNA/
RDNA; the inherited loop paid it per byte. Repetitive inputs (zero runs, TTI's sparse regions) are
decoded almost entirely through this loop on wave64 (the doubling fast path is wave32-only in the
curated lineage). Bytes identical.
Prediction: gfx942 +10–30 % decompression on zeros / repetitive data, ≈ 0 on incompressible; gfx1100
small gain on short matches; bytes identical.
Measured: gfx1100 ×0.997–1.002 (wave32 takes the doubling path for long matches; the modulo loop
only serves ≤ 2·blockDim matches there); gfx942: the 2ea6797 row reads ×0.85 at saturation on
binary/tti while the next commit (baf0fce, identical wave64 code) reads ×1.01 and x0 is ×0.99–1.01
— a node-level transient during that commit's window, not the change. The sweep's inputs contain
no long repeated runs (binary/tti/words), so the loop it optimises is barely exercised; zeros is
added to the LZ4 inputs for the next round.
Result: neutral on the measured inputs; to be re-read with zeros in the lz4b round.
Verdict: KEPT (exact; cheaper per byte where it runs), pending the zeros number.

### LZ4-D1 — vectorised copies and doubling repeat on wave64 (knob, default = curated)      Category: C3   Status: KNOB — variants pending
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/LZ4Kernels.hiph`
Change: the E17 wave32-only pieces — `coopCopyVec` (dword body fed by `unaligned_load32`, byte
head/tail) in `coopCopyNoOverlap`/`copyLiterals`, and the doubling expansion of long overlapped
matches in `coopCopyRepeat` — are selected by `ARCTO_LZ4_VEC_COPY` instead of `USE_WARPSIZE_32`:
default 1 on wave32, 0 on wave64 and CUDA (identical code to the curated lineage). Variants to
measure on gfx942: `-DARCTO_LZ4_VEC_COPY=1`, with and without `-DARCTO_LZ4_DECOMP_MIN_WAVES_PER_EU=8`.
Why (mechanism): on wave64 a byte copy issues one 64-B store per instruction; the dword path
issues 256 B per instruction with 4× fewer `s_waitcnt` round trips; the doubling turns a
period-`dist` match of length L into log2(L/dist) straight copies. The earlier wave64 loss (E17)
was measured without launch bounds; under LZ4-D7 the VGPR budget is explicit.
Prediction: with bounds, VEC_COPY=1 on wave64 +5–20 % on literal-/run-heavy inputs; if it still
loses, the cause is the extra registers (check `-Rpass-analysis` VGPRs vs 64) — then keep off.
Measured (gfx942, 30 reps, vs the branch head with the knob off): `VEC_COPY=1`: decompression
synth_binary x0 4.45 → 9.92 GB/s (×2.23), x512 589 → 764 (×1.31); tti x0 ×2.15, x512 482 → 709
(×1.48); words x0 ×0.80, x32 26.2 → 19.4 (×0.75); compression words x32 11.5 → 6.4 (×0.56, x0
×0.83), binary/tti ×0.98–0.99. `VEC_COPY=1 + MIN_WAVES=8`: same picture, 3–5 % lower at saturation.
Bytes identical, tests green.
Result: on wave64 the dword copy path is a very large win for literal-/long-match-dominated data and
a large loss for short-sequence data, on both sides: the compressor's `copyLiterals` goes through
the same routine and its per-call setup (alignment head, body, tail loops, `unaligned_load32`)
is paid on every short literal run; the `length < 32` byte fast path is too low for 64 lanes.
Verdict: SPLIT (next commit, LZ4-D1s): the decompression side and the compression side become
separate knobs with a tunable short-length cutoff; wave64 defaults to be fixed by the lz4b variants
(decomp-only at MIN 32 / 128 / 256, both at 128).

### LZ4-D1s — split the vectorised-copy knob: decompression / compression / short-length cutoff   Category: C3   Status: KNOBS — variants pending on gfx942
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/LZ4Kernels.hiph`
Change: `ARCTO_LZ4_VEC_COPY_DECOMP` (coopCopyNoOverlap + doubling repeat), `ARCTO_LZ4_VEC_COPY_COMP`
(`copyLiterals`), `ARCTO_LZ4_VEC_COPY_MIN` (byte length below which the plain byte loop runs; 32 =
curated). Defaults unchanged on every target (wave32 both on / 32; wave64 and CUDA off).
Variants (gfx942): DECOMP=1 MIN=32; DECOMP=1 MIN=128; DECOMP=1 MIN=256; DECOMP=1 COMP=1 MIN=128.
gfx1100: the commit is a no-op (regression check only).
Prediction: DECOMP-only at MIN 128 keeps the binary/tti gains and removes most of the words loss
(short copies stay on the byte loop); COMP stays off on wave64 unless MIN=128 turns the words
compression loss around.
Measured: (pending)

### LZ4-D5 — wave-uniform decompressor state through `readfirstlane`                    Category: C2   Status: PENDING
Commit: (this commit)  (branch `opt/lz4-decomp-wave64-2026-08`)
Files: `src/LZ4Kernels.hiph` (`decompressStream`, `BufferControl::readLSIC`)
Change: the token byte, every LSIC byte and the 16-bit match offset — loaded by all lanes from the
same LDS/global address — pass through `__builtin_amdgcn_readfirstlane` (`ARCTO_LZ4_UNIFORM`,
identity on CUDA). Everything derived from them (`num_literals`, `match`, `comp_idx`, `decomp_idx`,
the bounds checks and the branch conditions) becomes scalar.
Why (mechanism): an LDS/global load always lands in VGPRs even when the address is uniform, so the
serial sequence parser ran on the vector unit with exec-mask branches and replicated its loop
state in every lane; with the state in SGPRs the parsing is SALU work (`s_add`, `s_cmp`,
`s_cbranch`) that overlaps with the lanes' copy loops, and fewer VGPRs are live across the copies.
Same bytes: the values are identical on every lane.
Prediction: +3–10 % decompression on inputs with many short sequences (words, binary), small on TTI;
both wave sizes; bytes identical. Check `-Rpass-analysis`: VGPRs should drop.
Measured: (pending)
Result: (pending)
Verdict: (pending)
