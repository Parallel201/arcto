# Snappy compression — experiment log (branch `opt/snappy-comp-2026-08`)

Baseline: `main` @ `e3e1a1f` plus `chore/build-and-cleanup-2026-08` and `test/coverage-2026-08`
(build hygiene and the Snappy coverage gate — no kernel changes). The compressor (`snap_kernel`,
`src/snappy/compression.hiph`) is the hipified nvCOMP 2.2 design: one 2-wave block per chunk,
WARP0 emits the previous literal run and copy token while WARP1 searches the next 4-byte match
(12-bit hash map in LDS) and extends it; two `__syncthreads` per literal/copy pair. gfx1100
resource report (ARCTO_DEVICE_REPORTS, commit 798b0c4): 35 VGPRs, 8.2 KB LDS/block, occupancy
8 waves/SIMD (LDS-bound), 0 spills.

Plan (from `docs/AMD_OPTIMIZATION_MAP.md` §5.3): SNP-C5 → C2 → C1 → C3 (→ C4 if the resource
report on gfx942 shows a register constraint). Measured with `benchmark_snappy_chunked`
(compression column), the exact-bytes ladder and `test_snappy_coverage`, same protocol as the
decompression branch (`docs/experiments/README.md`). Branch is independent of
`opt/snappy-decomp-2026-08` (disjoint files except `snappy/config.h`); the two stack in an
integration branch.

---

### SNP-C5 — clear the whole hash map with 16-byte LDS stores (fixes the wave64 half-clear)   Category: C4   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-comp-2026-08`)
Files: `src/snappy/compression.hiph`, `src/snappy/compression_state.hiph`
Change: `hash_map` (4096 × `uint16_t` = 8 KB in LDS) is `alignas(16)` and cleared with `uint4`
stores, `sizeof/16` words spread over the group's lanes (8 per lane on wave64, 16 on wave32).
The previous loop (`for i = t; i < sizeof/4; i += 2·GROUPSIZE` dword stores) covers the table for
a 32-lane group but only **half of it for a 64-lane group** (stride 128 dwords over 2048): on
wave64 the compressor started every chunk with stale entries left by the previous block on that
CU. Entries are validated against the data before a match is taken, so the stream stayed valid,
but match *choice* — and therefore the compressed bytes — depended on LDS history. Now every
entry is zero on entry on every wave size. On wave32 (gfx1100) the bytes are unchanged by
construction; on wave64 (gfx942) the exact-bytes ladder may move — report it.
Why (mechanism): correctness/determinism first; performance-wise 16 `ds_write_b32` per lane →
8 `ds_write_b128` on wave64 (and 16 instead of 32 on wave32) at kernel start, negligible per chunk.
Prediction: compression throughput unchanged within noise on both archs; bytes identical on
gfx1100; bytes may differ on gfx942 versus the half-cleared baseline (a fix, not a regression).
Measured: (pending) `benchmark_snappy_chunked` compression column, exact-bytes ladder,
`test_snappy_coverage` (determinism check: two compressions identical).
Result: (pending)
Verdict: (pending)

### SNP-C2 — one barrier per literal/copy pair (double-buffered match results)        Category: C1   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-comp-2026-08`)
Files: `src/snappy/compression.hiph`, `src/snappy/compression_state.hiph`
Change: `literal_length`, `copy_length`, `copy_distance` become 2-slot arrays indexed by iteration
parity. In iteration k every thread reads slot `k&1` (WARP1's results from iteration k−1) and
WARP1 (`FindFourByteMatch` + `Match60`) writes slot `(k+1)&1`, so the `__syncthreads` that stood
between "read the three fields" and "WARP1 overwrites them" is gone; the end-of-iteration barrier
(WARP0 needs WARP1's results next iteration) stays. Slot `nxt` was last read in iteration k−1
before that barrier, so there is no read/overwrite hazard. Same symbol sequence; bytes unchanged;
same code on CUDA.
Why (mechanism): the loop runs once per literal/copy pair (thousands of times per 64 KB chunk);
a workgroup barrier for a 2-wave block is cheap but it serialises the two waves' instruction
streams at every iteration — halving the barrier count shortens the per-iteration critical path.
Prediction: ~5–10 % compression throughput; bytes identical (gfx1100) / identical to SNP-C5 (gfx942).
Measured: (pending) same protocol as SNP-C5.
Result: (pending)
Verdict: (pending)

### SNP-C1 — one-round-ahead prefetch of the match-search data word                  Category: C3   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-comp-2026-08`)
Files: `src/snappy/compression.hiph`
Change: in `FindFourByteMatch` each round loaded its lane's 4-byte window (`unaligned_load32`, two
aligned dword loads + funnel shift) at the top of the round, heading the dependent chain
load → hash → 12 ballots → LDS lookup → verify load → ballot. The loop only continues when
`literal_cnt == GROUPSIZE`, so the next round's position is exactly `pos + GROUPSIZE`: its load is
now issued at the start of the current round and consumed next round (the last round's prefetch
is wasted — one cache-resident 4-byte load per call). Same rounds, same matches, bytes unchanged.
Why (mechanism): removes one global-load latency (L1 hit ≈ 100+ cycles on CDNA, more from L2) from
the start of rounds ≥ 1 and overlaps it with the ballot/LDS work; the compressor runs 2 waves per
block at ≈ 8 waves/SIMD (LDS-bound), so little latency is hidden by other waves.
Prediction: single digits to ~15 % on literal-heavy inputs (more rounds per literal: random,
binary, TTI), ≈ 0 on highly compressible data; bytes identical.
Measured: (pending) same protocol as SNP-C5.
Result: (pending)
Verdict: (pending)

### SNP-C3 — dword literal emission in `StoreLiterals` (AMD)                            Category: C3   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-comp-2026-08`)
Files: `src/snappy/compression.hiph`
Change (AMD only; CUDA keeps the byte loop): the emitting wave copied a literal run byte by byte
(`dst[i] = src[i]`, stride GROUPSIZE). It now moves 4 bytes per lane per step with unaligned dword
loads/stores (whole dwords only where the dword's last byte is inside the output buffer, otherwise
byte-wise — same end-of-buffer semantics), then a byte tail. Output bytes identical.
Why (mechanism): 4× fewer VMEM instructions for literal bodies (up to 256 B per run); the
emitting wave is rarely the bottleneck (the match-finding wave is), so the gain is bounded by how
often WARP0 is on the critical path — mostly on incompressible data with long literal runs.
Prediction: small; visible only on random/binary/TTI inputs; bytes identical.
Measured: (pending) same protocol as SNP-C5.
Result: (pending)
Verdict: (pending)
