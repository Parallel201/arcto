# Snappy decompression — experiment log (branch `opt/snappy-decomp-2026-08`)

Baseline: `main` @ `e3e1a1f` plus the `chore/build-and-cleanup-2026-08` and
`test/coverage-2026-08` branches (build hygiene and the Snappy coverage gate — no kernel
changes). The Snappy kernels on this baseline are the hipified nvCOMP 2.2 design: one
3-wave block per chunk (decode / prefetch / process) communicating through a `volatile`
queue in LDS; nothing in them had been tuned for AMD before this branch.

Plan (from `docs/AMD_OPTIMIZATION_MAP.md` §5, on the `docs/amd-optimization-map-2026-08`
branch): SNP-D1 → D3 → D4 → D5 → D2 → D7 → D9 → D6 → D8 → D11 → D12 → D10.

Gates: see `docs/experiments/README.md`. Benchmark: `benchmark_snappy_chunked`.

---

### SNP-D1 — yield the SIMD in the decoder's spin-waits (`s_sleep`)        Category: C1   Status: PENDING (gfx1100 neutral; gfx942 pending)
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/device_functions.hiph`, `src/snappy/config.h`, `src/snappy/decompression_decode.hiph`,
`src/snappy/decompression_prefetch.hiph`, `src/snappy/decompression_process.hiph`
Change: the four polling loops of the three-wave decoder (decoder waiting for prefetched
bytes, decoder waiting for a free symbol slot, prefetcher waiting for ring space,
processor waiting for a batch) back off with `__builtin_amdgcn_s_sleep(n)` on AMD
instead of the inherited `clock()` no-op; `NANOSLEEP` becomes `SPIN_SLEEP` with
platform-dependent units (ns on CUDA sm_70+, s_sleep units on AMD), the literal
`NANOSLEEP(100)` becomes the named `PROCESS_SLEEP`, and the AMD amounts are tunables
`ARCTO_SNAPPY_{PREFETCH,DECODE,PROCESS}_SLEEP` (defaults 2 / 1 / 1). Algorithm, stream
format and emitted bytes unchanged (scheduling only). CUDA path unchanged.
Why (mechanism): `s_sleep n` parks the wave for ~64·n clocks and hands the SIMD's issue
slots to other resident waves. With ≈ 6–9 Snappy blocks per CU (LDS-bound) a large share
of resident waves are pollers at any moment; a tight `volatile`-LDS poll competes for
issue and LDS bandwidth with the productive waves of the other chunks on the same SIMD
(and, for the decoder lane-0 polls, with its own sibling waves). Applies to CDNA and RDNA3.
Prediction: decompression throughput up at saturation (many resident blocks) — tens of
percent plausible, near zero at small batch; compression unchanged; bytes identical;
effect should grow with the number of resident blocks (`-x`) and be visible in
`SQ_WAIT_INST_LDS` / `SQ_INSTS_LDS` going down.
Measured: (pending) lunaris gfx1100 wave32 and sdumont2nd4014 gfx942 wave64, ROCm 7.0.1
container, `benchmark_snappy_chunked` on tti_rsf_64x64x64_t050 / synth_binary / words
inputs, `-p 65536`, `-x 512` and `-x 0`, 30 reps; sweep of the three tunables
(0/0/0 = yield-only, 1/1/2 default, 2/2/4, 4/4/8).
Result (gfx1100, RX 7900 XT, wave32, ROCm 7.0.1 container, commit f1a3e62 vs baseline 0263261,
30 reps, median [Q1–Q3], `-p 65536`): decompression synth_binary x512 89.05 → 89.70 GB/s
(×1.007), tti_rsf x512 89.78 → 89.54 (×0.997), words x32 25.17 → 25.45 (×1.011); small batch
(x0) ×1.001 / ×1.007 / ×1.003; compression unchanged (27.06 / 26.97 / 3.75 GB/s, ×1.000);
exact-bytes ladder identical on all six fixtures; all Snappy tests green. ctest of the
underlying test branch: 1 pre-existing failure, `BitPackGPU_test` (legacy Cascaded unit test,
gfx1100 wave32; unrelated to Snappy — to be investigated on the Cascaded branch).
Verdict: NEUTRAL on gfx1100 (≤ +1 % decompression, within noise). The prediction "tens of %"
did not hold on RDNA3 wave32: the polling waves are not stealing a measurable share of the
SIMD from productive waves there. Side observation: decompression of copy-heavy input (words,
25 GB/s) is 3.5× slower than of all-literal input (TTI/binary, ~89 GB/s) — the decoder is
symbol-rate bound, which is what SNP-D3/D4/D7 target. Kept pending the gfx942 (wave64,
8 waves/SIMD) measurement and the tunable sweep before a final verdict.

### SNP-D4 — compute wave-uniform decoder state on all lanes (no lane-0 + broadcast)   Category: C2   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/decompression_decode_strategies.hiph`, `src/snappy/decompression_decode.hiph`
Change: `len3_mask`, `batch_len` (2-to-3 strategy) and `len5_mask` (2-to-5 strategy) were
computed by lane 0 only and broadcast with a 64-bit shuffle; their inputs are ballot results,
i.e. identical on every lane, so every lane now computes them directly. The batch slot pointer
`b` was set by lane 0 and broadcast as a pointer; the batch index is now advanced by every lane
(after the existing `batch_len` broadcast, where it is uniform) and the slot pointer computed by
every lane. Removes three `SHFL10` (64-bit/pointer `ds_bpermute` pairs) per batch and one per
2-to-5 round. Same values on every lane as before; algorithm and bytes unchanged; CUDA unchanged.
Why (mechanism): a `(t == 0) ? f(ballots) : 0` region forces the `get_len3_mask_64` chain
(16 dependent `k_len3lut` lookups) into a divergent VALU region and the result through the LDS
crossbar; with all lanes computing from SGPR-resident ballots the compiler can keep the chain on
the scalar ALU / scalar cache and `batch_len`/`len3_mask` become SGPRs, which also enables
`readlane` broadcasts with uniform indices (SNP-D3).
Prediction: small per-batch gain (a few hundred cycles per batch), larger on short-symbol-heavy
data (more batches); bytes identical; ISA shows `s_load_dword`/`s_lshr_b64` in the mask chain.
Measured: (pending) same protocol as SNP-D1.
Result: (pending)
Verdict: (pending)

### SNP-D3 — `readfirstlane` / `readlane` instead of shuffles for wave-uniform broadcasts   Category: C2   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/device_functions.hiph`, `src/snappy/decompression_decode_strategies.hiph`,
`src/snappy/decompression_process.hiph`
Change: on AMD, `SHFL10(v)` (broadcast of lane 0) is now `v_readfirstlane_b32` (two for 64-bit
values) instead of `__shfl(v, 0)` (= `ds_bpermute_b32` through the LDS crossbar). A new
`SHFL1_UNIFORM(v, lane)` maps to `v_readlane_b32` and replaces `SHFL1` only where the lane index
is provably wave-uniform: `batch_len-1` / `batch_len` / `batch_add-1` in the two decode strategies
(derived from ballots, computed on all lanes since SNP-D4), `n-1` and the loop counter `i`, `i+1`
in the symbol processor (`n` comes from a ballot and a butterfly sum). Per-lane-indexed shuffles
(`it`, `(n+t)&63`, the compressor's `min(local_match_lane,t)`) stay `ds_bpermute`. CUDA path
unchanged (`SHFL1_UNIFORM` = `__shfl_sync`). `readfirstlane` reads the lowest *active* lane; every
`SHFL10` site is reached by all lanes with lane 0 active (verified site by site), so the value is
lane 0's as before. Algorithm and bytes unchanged.
Why (mechanism): `ds_bpermute` is an LDS-path instruction (issue through the LDS pipeline,
`s_waitcnt lgkmcnt`, ~tens of cycles, two for 64-bit) executed on the decoder/processor critical
path ~10–15 times per batch; `v_readfirstlane`/`v_readlane` read one lane's VGPR into an SGPR in a
few cycles, and the broadcast value living in an SGPR removes VGPRs and lets dependent control
flow be scalar.
Prediction: 5–15 % decompression throughput on the decoder/processor-bound inputs (text-like,
tiny symbols), less on literal-heavy input; ISA `ds_bpermute_b32` count in `unsnap_kernel` drops;
bytes identical.
Measured: (pending) same protocol as SNP-D1.
Result: (pending)
Verdict: (pending)

### SNP-D5 — 8-byte-aligned symbol queue, one 64-bit LDS access per symbol        Category: C4   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/symbol.hiph`
Change: `LZ77Symbol` (len, offset; 8 bytes) is `alignas(8)`, which moves the queue `batch[]` in
`unsnap_queue_s` from byte offset 76 to 80 of the LDS state (and the ring `buf` to 2096), and
`set()`/`get()` move the symbol with one 64-bit `volatile` store/load (len in the low dword,
offset in the high dword — the same layout as before) instead of two 32-bit `volatile` accesses.
Debug builds (`ARCTO_PRINT_DEBUG_INFO`, three fields) keep the field-wise path. Values, order and
bytes unchanged.
Why (mechanism): the decoder lanes write `b[t]` and the processor lanes read `b[t]` at an 8-byte
lane stride; two 32-bit accesses at that stride hit the same bank pair twice (2-way conflict per
32-lane half) and cost two LDS instructions per lane; one aligned `ds_write_b64`/`ds_read_b64` per
lane is conflict-free at full rate and halves the volatile LDS instruction count in `set()` (two
decode strategies, single-thread decoder) and `get()` (processor).
Prediction: small (LDS is not the bottleneck) but free; fewer `ds_*` instructions in the ISA;
`SQ_LDS_BANK_CONFLICT` down; bytes identical.
Measured: (pending) same protocol as SNP-D1.
Result: (pending)
Verdict: (pending)

### SNP-D7a — serial decoder: one dword-window read instead of up to five byte reads   Category: C4   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/decompression_state.hiph`, `src/snappy/decompression_decode.hiph`
Change: `unsnap_queue_s::read_window5(pos)` reads two aligned dwords of the prefetch ring (one
round trip, wrap-safe because the ring size is a multiple of 4) and funnel-shifts them so bytes
`pos..pos+4` (at least) sit in the low bytes of a 64-bit value; the single-thread decoder
(`decode_and_fill_batch_using_single_thread`, lane 0 while 63 lanes idle) takes the tag byte and
the up to four following bytes (1-/2-/4-byte copy offsets, 1–4 long-literal length bytes) from
that window instead of issuing one `volatile` byte load per byte as it discovers it needs it. The
ring is `alignas(16)`. Parsing logic unchanged; bytes unchanged.
Why (mechanism): the serial path runs on every batch that contains a literal > 4 chars or a
copy-4 (text-like and well-compressed data) with one lane active; each symbol cost 1–5 *dependent*
`ds_read_u8` + `s_waitcnt` (the branch on the tag decides which bytes to read next). Two
independent `ds_read_b32` issued back to back cut that to one LDS round trip per symbol.
Prediction: medium on text-like / copy-heavy inputs (words: decode is symbol-rate bound,
25 GB/s vs 89 GB/s for all-literal data), nil on literal-only inputs; bytes identical.
Measured: (pending) same protocol as SNP-D1.
Result: (pending)
Verdict: (pending)

### SNP-D7b — per-lane symbol window in the two parallel decode strategies          Category: C4   Status: PENDING
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/decompression_decode_strategies.hiph`
Change: in both parallel strategies each lane read its tag byte at its symbol position, then —
after the prefix-sum / range-check chain, inside `if (t < batch_len)` — re-read one or two offset
bytes from the ring. Each lane now reads its whole symbol (tag + offset bytes) with one
`read_window5` (two dword LDS loads in one round trip) at the point where the tag was read, and
decodes the offset from that window. The three ballot tag probes at `cur_t + k·GROUPSIZE` in the
2-to-3 strategy and the first probe of the 2-to-5 strategy are unchanged (single bytes at
unrelated positions). Values and bytes unchanged.
Why (mechanism): removes one dependent LDS round trip (the late offset-byte loads) from the
per-batch critical path of the decode wave; LDS instruction count per lane is 2 instead of 1–3
(literal 1 → 2, copy-1 2 → 2, copy-2 3 → 2), i.e. neutral-to-slightly-higher bandwidth but one
less dependent latency — a latency-for-bandwidth trade that should pay on a latency-bound wave.
Prediction: small gain on copy-heavy inputs, possibly neutral; bytes identical. Kept separate
from SNP-D7a so the two can be judged independently.
Measured: (pending) same protocol as SNP-D1.
Result: (pending)
Verdict: (pending)
