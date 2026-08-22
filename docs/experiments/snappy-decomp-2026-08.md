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

### SNP-D1 — yield the SIMD in the decoder's spin-waits (`s_sleep`)        Category: C1   Status: KEPT — neutral on gfx1100 (60-rep A/B), small positive on gfx942 at the branch head (60-rep A/B: removing it costs 4 % TTI / 2 % words at saturation)
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
Result (gfx942): gfx942 (MI300A, wave64, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2; note: saturation medians on this node carry an IQR of ±2–4 %, small-batch (x0) medians are tight): decompression binary x512 ×0.974 (180.4 → 175.7, IQRs overlap partially), tti x512 ×1.019, words x32 ×1.025; small batch ×1.005 / ×0.987 / ×1.009; compression ×1.000; bytes identical. Verdict on gfx942: NEUTRAL (±2 %, inside the node's noise), same as gfx1100. The 'polling waves steal the SIMD' hypothesis is not supported on either architecture at the default amounts.
A/B at branch head (gfx1100, 60 reps): head-without-D1 vs head = ×1.004 / ×0.993 / ×1.000 (binary / tti / words at saturation), ×1.000 / ×1.007 / ×0.999 at small batch; the yield-only variant (sleep amounts 0/0/0) = ×0.992 / ×0.993 / ×0.999. Verdict: NEUTRAL with 60-rep confidence on gfx1100; the amounts do not matter. Kept only because the AMD fallback (`clock()`) was a no-op by accident; it is not a performance contribution.
A/B at branch head (gfx942, 60 reps): head-without-D1 vs head = ×1.001 / ×0.960 / ×0.980 (binary / tti / words at saturation; tti IQRs 280.8–289.1 vs 267.7–277.2 GB/s, disjoint), ×0.995 / ×0.985 / ×1.009 at small batch. On wave64, with the faster decoder of the branch head, the yielding polls are worth ≈ 2–4 % on the decoder-bound inputs — the original hypothesis holds weakly on CDNA3 only.
Verdict: NEUTRAL on gfx1100 (≤ +1 % decompression, within noise). The prediction "tens of %"
did not hold on RDNA3 wave32: the polling waves are not stealing a measurable share of the
SIMD from productive waves there. Side observation: decompression of copy-heavy input (words,
25 GB/s) is 3.5× slower than of all-literal input (TTI/binary, ~89 GB/s) — the decoder is
symbol-rate bound, which is what SNP-D3/D4/D7 target. Kept pending the gfx942 (wave64,
8 waves/SIMD) measurement and the tunable sweep before a final verdict.

### SNP-D4 — compute wave-uniform decoder state on all lanes (no lane-0 + broadcast)   Category: C2   Status: KEPT (enabler; neutral on gfx1100 and gfx942)
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
Result: gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2 — cumulative on the branch, each vs baseline 0263261): decompression binary x512 ×1.007, tti x512 ×0.997, words x32 ×1.005 (small batch ×1.000/×1.006/×0.999); compression ×1.000; bytes identical; tests green.
Result (gfx942): gfx942 (MI300A, wave64, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2; note: saturation medians on this node carry an IQR of ±2–4 %, small-batch (x0) medians are tight): decompression binary x512 ×0.982, tti x512 ×1.042, words x32 ×1.016 (vs baseline); small batch ×1.009 / ×0.988 / ×1.009; bytes identical. Neutral within noise, as on gfx1100.
Verdict: NEUTRAL alone on gfx1100, KEPT as the enabler of SNP-D3 (`batch_len`/masks become wave-uniform SGPR values the `readlane` sites depend on). gfx942 pending.

### SNP-D3 — `readfirstlane` / `readlane` instead of shuffles for wave-uniform broadcasts   Category: C2   Status: KEPT (gfx1100 +3–5 %, gfx942 +2–11 %)
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
Result: gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2 — cumulative on the branch, each vs baseline 0263261): decompression binary x512 89.29 → 92.38 GB/s (×1.042 vs baseline ×1.035 vs previous commit), tti x512 89.43 → 92.62 (×1.033), words x32 25.42 → 26.58 (×1.051); small batch ×1.074 / ×1.051 / ×1.065; compression ×1.000; bytes identical; tests green.
Result (gfx942): gfx942 (MI300A, wave64, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2; note: saturation medians on this node carry an IQR of ±2–4 %, small-batch (x0) medians are tight): decompression tti x512 260.9 → 272.9 GB/s (×1.090 vs baseline, ×1.046 vs SNP-D4), words x32 41.25 → 44.00 (×1.083, ×1.067), binary x512 177.1 → 179.9 (×0.997 vs baseline — the incompressible stream at saturation is not decoder-bound); small batch binary ×1.089, tti ×1.024, words ×1.109; bytes identical. KEPT on both architectures — the largest single gain of the branch so far, larger on wave64 for the decoder-bound inputs.
ISA evidence (gfx1100, `--save-temps` at commit 798b0c4 = through SNP-D7b): the SnappyBatchKernels TU contains 15 `v_readfirstlane_b32` and 10 `v_readlane_b32` (the SNP-D3 sites) and 26 remaining `ds_bpermute_b32` (per-lane shuffles + the WarpReduce trees SNP-D2 replaces); 4 `s_sleep` (SNP-D1); no calls (`s_swappc_b64` = 0, everything inlined); `unsnap_kernel`: 53 VGPRs, 0 spills, 16 waves/SIMD (occupancy-unconstrained on wave32), `snap_kernel`: 35 VGPRs, 8 waves/SIMD (LDS-bound), HLIF decompress 69 VGPRs at 16 waves/SIMD — SNP-D9 (launch bounds) therefore has no lever on gfx1100.
Verdict: KEPT on gfx1100: +3.3…+5.1 % at saturation, +5…+7 % at small batch, across all three inputs — the prediction (5–15 % on decoder-bound input) holds at its lower end. gfx942 pending.

### SNP-D5 — 8-byte-aligned symbol queue, one 64-bit LDS access per symbol        Category: C4   Status: KEPT on both (gfx1100 +1–2 %; gfx942 60-rep A/B: removing it costs 2.5–4 % TTI)
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
Result: gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2 — cumulative on the branch, each vs baseline 0263261): decompression binary x512 92.38 → 93.35 (×1.053 vs baseline, ×1.011 vs SNP-D3), tti x512 92.62 → 94.36 (×1.052, ×1.019), words x32 26.58 → 26.89 (×1.063, ×1.012); small batch ×1.084 / ×1.079 / ×1.072 vs baseline; compression ×1.000; bytes identical; tests green.
Result (gfx942): gfx942 (MI300A, wave64, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2; note: saturation medians on this node carry an IQR of ±2–4 %, small-batch (x0) medians are tight): decompression vs SNP-D3: binary x512 ×0.994, tti x512 272.9 → 257.4 (×0.943), words x32 44.00 → 43.36 (×0.985); small batch binary 2.36 → 2.34 (×0.992), tti 4.44 → 4.60 (×1.036), words 8.47 → 8.10 (×0.956, tight IQRs — a real loss); bytes identical. On wave64 the 64-bit symbol access is not the small free win it is on wave32: the words small-batch case (decoder-bound, many short symbols) loses 4 %. Hypotheses to test with a dedicated A/B (more repetitions, D5 reverted on top of the current branch head): the pack/unpack VALU on the lane-0 serial path, or the changed LDS offsets of `buf` (2124 → 2096) altering ring bank mapping. Verdict deferred.
A/B at branch head (gfx1100, 60 reps): head-without-D5 vs head = ×1.002 / ×0.988 / ×0.993 at saturation, ×0.998 / ×1.007 / ×0.989 at small batch — i.e. D5 is worth ≈ +1 % on gfx1100 on the decoder-bound inputs, consistent with sweep 2/3. gfx942 A/B (60 reps) pending.
A/B at branch head (gfx942, 60 reps): head-without-D5 vs head = ×0.998 / ×0.961 / ×0.997 at saturation, ×0.995 / ×0.975 / ×1.009 at small batch; bytes identical. The sweep-2 'mixed' reading was run-order drift; at the branch head D5 is a consistent +2.5–4 % on TTI on MI300A. KEPT on both architectures.
Verdict: KEPT on gfx1100: a consistent +1.1…+1.9 % on top of SNP-D3 on all inputs, as predicted (small, free). gfx942 pending.

### SNP-D7a — serial decoder: one dword-window read instead of up to five byte reads   Category: C4   Status: NEUTRAL on gfx1100 and gfx942 — kept as the carrier of `read_window5` (used by D7b/D6/D8); judged with them
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
Result: gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2 — cumulative on the branch, each vs baseline 0263261): decompression binary x512 93.35 → 94.44 (×1.065 vs baseline, ×1.012 vs SNP-D5), tti x512 94.36 → 94.59 (×1.055, ×1.002), words x32 26.89 → 26.76 (×1.058, ×0.995); small batch ×1.088 / ×1.075 / ×1.073 vs baseline; compression ×1.000; bytes identical; tests green.
Result (gfx942): gfx942 (MI300A, wave64, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2; note: saturation medians on this node carry an IQR of ±2–4 %, small-batch (x0) medians are tight): decompression vs SNP-D5: binary x512 ×1.010, tti x512 ×0.998, words x32 ×1.012; small batch ×1.002 / ×0.996 / ×1.000; bytes identical. Neutral on both architectures: the serial path is not a measurable share of the words input either. The helper it introduces (`read_window5`) is what D7b, D6 and D8 build on, so the commit stays; its own effect is nil.
Verdict: NEUTRAL on gfx1100 (+1.2 % binary, ±0 tti/words — within noise): the serial path is not where the words input spends its time, contrary to the prediction. Decision deferred to gfx942 and to SNP-D7b (same mechanism on the parallel strategies).

### SNP-D7b — per-lane symbol window in the two parallel decode strategies          Category: C4   Status: NEUTRAL on gfx1100, +0…+3 % on gfx942 — kept (harmless, part of the dword-window set)
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
Result (gfx942): gfx942 (MI300A, wave64, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 2; note: saturation medians on this node carry an IQR of ±2–4 %, small-batch (x0) medians are tight): decompression vs SNP-D7a: binary x512 ×1.005, tti x512 256.9 → 263.6 (×1.026), words x32 43.87 → 43.02 (×0.981); small batch ×1.006 / ×1.000 / ×1.006; bytes identical. Small positive on tti, small negative on words at saturation — inside the noise band; gfx1100 (sweep 3) pending.
Result (gfx1100): gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 3; cumulative on the branch): decompression vs SNP-D5: binary x512 ×0.999, tti x512 ×1.002, words x32 ×0.998; small batch ×1.005 / ×1.004 / ×0.996; bytes identical. Neutral.
Verdict: (pending)

### SNP-D2 — rocPRIM DPP warp scan / all-lanes reduce instead of shuffle trees        Category: C2   Status: KEPT (gfx1100 +1–3 %; gfx942 pending)
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/device_functions.hiph`
Change: on AMD, `WarpReduce<GROUPSIZE,WARPSIZE>::prefix_sum` (inclusive prefix sum used by both
decode strategies and by the symbol processor's combine loop) and `::sum` (all-lanes total, used
for the processor's `start_mask`) are implemented with `rocprim::warp_scan<T,GROUPSIZE>::inclusive_scan`
and `rocprim::warp_reduce<T,GROUPSIZE,UseAllReduce=true>::reduce` instead of the hand-written
`SHFL1`/`SHFL1_XOR` trees (6 dependent `ds_bpermute` steps for 64 lanes, 12 for a 64-bit value).
For a logical warp equal to the hardware wave (64 on CDNA, 32 on the gfx1100 wave32 build) rocPRIM
selects its cross-lane DPP implementation (row operations / swizzles, no LDS traffic, empty
storage type — a `static_assert` guards against a silent shared-memory fallback). Results
identical (inclusive prefix sum, total broadcast to all lanes). CUDA keeps the shuffle trees.
Why (mechanism): every shuffle step is an LDS-path instruction with `lgkmcnt` latency; DPP row
operations execute at VALU rate. The processor's combine loop runs a prefix sum + a 64-bit
all-reduce + ~5 shuffles per 64 output bytes, the decode strategies one prefix sum per batch.
Prediction: 10–25 % on short-symbol data (words, tiny symbols) where the combine loop dominates,
small on literal-heavy data; bytes identical; ISA `ds_bpermute_b32` count drops further.
Measured: (pending) same protocol as SNP-D1.
Result: gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 3; cumulative on the branch): decompression vs SNP-D7b: binary x512 93.57 → 94.62 (×1.011), tti x512 94.30 → 95.76 (×1.015), words x32 26.78 → 27.65 (×1.032); small batch ×1.022 / ×1.031 / ×1.045; vs baseline ×1.063 / ×1.074 / ×1.093 at saturation; compression ×1.000; bytes identical; tests green.
Verdict: KEPT on gfx1100: consistent +1–3 % (largest on the short-symbol words input, as predicted in direction though below the 10–25 % guess — the combine loop is one of several latency chains, not the only one). gfx942 pending (sweep 4).

### SNP-D6 — dword-granular prefetch of the compressed stream                         Category: C3   Status: KEPT (gfx1100 +1…+6 %; gfx942 pending)
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/decompression_prefetch.hiph`
Change: for a full granule (`PREFETCH_SECTORS·GROUPSIZE` = 512 B on wave64, 256 B on wave32) each
prefetcher lane loaded 8 single bytes at `pos + t + i·GROUPSIZE` and stored them byte-wise into
the ring. `base+pos` is GROUPSIZE-aligned by construction and advances by whole granules, so the
granule is now loaded as `PREFETCH_SECTORS/4` aligned dwords per lane (`pos + 4t + 4·GROUPSIZE·i`,
wave-contiguous), and written to the ring as dwords when `pos` is 4-byte aligned (the common
case: inputs in `hipMalloc`'d buffers — the alignment of `pos` is constant for the whole chunk) or
as bytes otherwise. Ring contents identical; bytes unchanged; same code on CUDA.
Why (mechanism): 8 `global_load_ubyte` → 2 `global_load_dword` per lane per granule (4× fewer
VMEM instructions and address computations, same coalescing); on the ring, lane-contiguous dword
stores hit one bank per lane instead of four lanes writing sub-dword bytes of the same bank.
Prediction: small–medium on inputs where the prefetcher is on the critical path (large,
incompressible chunks: binary/TTI), ~0 where the decoder is the bottleneck; bytes identical.
Measured: (pending) same protocol as SNP-D1.
Result: gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 3; cumulative on the branch): decompression vs SNP-D2: binary x512 94.62 → 95.83 (×1.013), tti x512 95.76 → 101.54 (×1.060), words x32 27.65 → 27.84 (×1.007); small batch ×0.997 / ×1.002 / ×1.002; vs baseline ×1.077 / ×1.139 / ×1.100 at saturation; compression ×1.000; bytes identical; tests green.
Verdict: KEPT on gfx1100: +6 % on the incompressible TTI stream at saturation, +1.3 % binary, +0.7 % words, nil at small batch — exactly the predicted profile (the prefetcher is on the critical path only when many chunks of literal-heavy data are resident). gfx942 pending (sweep 4).

### SNP-D10 — size kernel: one chunk per thread                                       Category: C6   Status: KEPT (correct; outside the timed paths)
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/lowlevel/SnappyBatchKernels.hip`
Change: `get_uncompressed_sizes_kernel` (varint header parse, used by
`arctoBatchedSnappyGetDecompressSizeAsync`) launched one `warpsize`-thread workgroup per chunk
with a single active lane; it now parses one chunk per thread in 256-thread blocks
(`grid = ⌈count/256⌉`). Same parse, same outputs.
Why (mechanism): 63 of 64 lanes idle and one workgroup dispatch per chunk → 64× fewer lanes
wasted and ~256× fewer workgroups; the parse is 1–5 dependent byte loads per chunk, so a
thread per chunk is the natural shape.
Prediction: µs-level absolute gain on the size query (not part of the timed compress/decompress
paths of the benchmark); visible only for very large batches; bytes unaffected.
Measured: (pending) correctness via test_snappy_coverage (size query checked per chunk).
Result: gfx1100 (RX 7900 XT, wave32, ROCm 7.0.1 container, 30 reps, median, `-p 65536`, sweep 3; cumulative on the branch): decompression vs SNP-D6: ×1.005 / ×0.999 / ×0.992 at saturation, ×0.998 / ×0.998 / ×0.999 at small batch — noise (the size query is not in the timed paths); test_snappy_coverage checks the query per chunk: green; bytes identical.
Verdict: KEPT (hygiene; no measurable effect on compress/decompress throughput, as expected).

### SNP-D8 — 4 bytes per lane in the processor's literal and non-overlapping copy paths   Category: C3   Status: SPLIT PENDING — gfx1100 +15…+17 %; gfx942 +6…+13 % on TTI but −17…−22 % on synth_binary
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/decompression_process.hiph`
Change (AMD only, `#if __HIP_PLATFORM_AMD__`; CUDA keeps the byte loops): (a) literals — each lane
moves 4 bytes per step (unaligned dword loads from the compressed stream, or two aligned LDS dword
loads + shift via `read_window5` while the literal is still in the ring; unaligned dword stores to
the output), `LITERAL_SECTORS/4` dwords per lane so a step still covers `LITERAL_SECTORS·GROUPSIZE`
bytes; whole-dword tail then 0–3 trailing bytes. (b) copies with `dist ≥ blen` (no overlap between
source and destination; Snappy copies are ≤ 64 B) — one unaligned dword per lane, byte tail; all
loads precede all stores and the source lies entirely before `out`, so there is no intra-copy
hazard. Overlapping copies (`dist < blen`, the repeat-pattern case) keep the original byte loop
with the modulo source index. Output bytes identical. (The first version of this commit indexed the
copy tail as `out[4*ndw + t - dist]` with an unsigned `ndw`, which wraps when `dist > 4·ndw + t` —
an out-of-bounds store caught immediately by the test gate as `hipErrorIllegalAddress` on every
input; fixed to pointer arithmetic before any measurement.)
Why (mechanism): the output side of the processor issued one `global_store_byte` per byte per lane
(64 B per wave-instruction); on RDNA3/CDNA3 literal-heavy streams are bound by store-instruction
issue rather than bandwidth. Unaligned dword global access is supported on AMD (unaligned-access
mode), so a 4× reduction in VMEM store (and load) instructions costs nothing in correctness.
Prediction: medium on literal-heavy / incompressible input (binary, TTI: 89–94 GB/s today),
smaller on copy-heavy input (words); bytes identical; ISA shows `global_store_dword` with
`align 1` replacing most `global_store_byte`.
Measured: (pending) same protocol as SNP-D1; test_snappy_coverage exercises odd sizes and
misaligned output pointers.
Result: gfx1100 (RX 7900 XT, wave32, 30 reps, sweep 5, commit cb7bd08 vs SNP-D10 b19be64): decompression binary x512 95.01 → 108.92 GB/s (×1.146; ×1.221 vs baseline), tti x512 101.32 → 118.57 (×1.170; ×1.315 vs baseline), words x32 27.75 → 27.10 (×0.977; ×1.069 vs baseline); small batch binary ×1.104→×1.221, tti ×1.117→×1.270, words ×1.120→×1.096 vs baseline; compression ×1.000; exact-bytes ladder identical; all Snappy tests green (incl. misaligned outputs). ISA at head (gfx1100): `global_store_b32`/`b64` present, `s_waitcnt` 155 in the TU.
Result (gfx942, MI300A, wave64, 30 reps, sweep 5, cb7bd08 vs SNP-D10 b19be64): decompression tti x512 295.7 → 313.6 GB/s (×1.061), tti x0 4.70 → 5.33 (×1.134); synth_binary x512 191.9 → 153.8 (×0.801!), x0 2.37 → 1.84 (×0.777!); words x32 46.38 → 45.16 (×0.974), x0 ×0.980; bytes identical; tests green. The sign flips with the input: literal-heavy TTI gains as on gfx1100, the copy-heavy `synth_binary` (ratio ≈ 3, many non-overlapping short copies) loses a fifth. Hypothesis: on CDNA3 the unaligned dword global loads/stores of the non-overlapping *copy* path are expensive (split transactions), whereas on RDNA3 they are cheap — the literal path should be fine on both. Action: two compile-time knobs (`ARCTO_SNAPPY_DWORD_LITERALS`, `ARCTO_SNAPPY_DWORD_COPIES`) and a per-knob A/B on gfx942; expected outcome: keep literals on both architectures, gate copies to wave32. (Head 94624af measured in the same sweep: tti x512 280.8 — a −10 % swing against cb7bd08 with identical device code; the MI300A saturation medians carry that much drift run to run, so verdicts lean on the small-batch numbers and paired A/Bs.)
Verdict: KEPT on gfx1100: the largest single gain of the branch on literal-heavy data (+15–17 % at saturation, +10–14 % at small batch); a 2 % loss on the copy-heavy words input, where short literals pay two LDS dword loads for 1–3 bytes. gfx942 pending (sweep 5).

### SNP-D12 — experiment knob: 32-lane decode groups inside a wave64 (+ punning hygiene)   Category: C6   Status: KNOB (default off) — variant pending on gfx942
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/decompression.hiph`, `src/snappy/decompression_decode_strategies.hiph`,
`src/snappy/decompression_decode_warp_scans.hiph`, `src/device_functions.hiph`
Change: `-DARCTO_SNAPPY_DECODE_GROUP=32` on a wave64 build instantiates the two decode strategies
with a 32-bit group mask (`TryDecodeStringOf2To{3,5}ByteSymbols<uint32_t, uint64_t>`): 32 symbols
per batch, the 8-step `len3`/`len5` mask chains of the original 32-lane nvCOMP design, half the
decode lanes; prefetcher and processor stay 64-wide. Default = the whole wave (unchanged code
path). Prerequisites fixed in the same commit: the `uint64_t → uint32_t` mask truncations in
`ballot1<uint32_t,64>` and `get_len3/5_mask<uint32_t,uint64_t>` used pointer punning
(`*reinterpret_cast<uint32_t*>(&v)`), now `static_cast`; and the 2-to-3 strategy's "every lane
decoded a short symbol" default was `warpsize` where `GROUPSIZE` is meant (identical when the
group is the wave, wrong for a 32-lane group). Default build: identical code and bytes.
Why (mechanism): nvCOMP tuned the parallel strategies for 32 symbols per batch; on wave64 the
batch doubled (64) and the serial mask chain doubled (16 lookups). Whether 64 wide symbols per
batch outweigh the longer chain is unknown a priori.
Prediction: unknown (either way); bytes identical. To be measured as a flag variant on gfx942.
Measured: (pending) `94fb297+ -DARCTO_SNAPPY_DECODE_GROUP=32` on gfx942 only.
Result: (pending)
Verdict: (pending)

### SNP-D11 — compile-time tunables: prefetch ring, granule, literal step, sleep amounts   Category: C6   Status: MEASURED (gfx1100); gfx942 pending
Commit: none (flag variants of the branch head 94624af, full rebuild each)
Change: `-DLOG2_PREFETCH_SIZE=13` (8 KB ring), `-DPREFETCH_SECTORS=4` (256-B granules),
`-DLITERAL_SECTORS=8` (two dwords per lane per literal step), `-DARCTO_SNAPPY_*_SLEEP=0`.
Result (gfx1100, 30 reps, vs head 109.06 / 119.11 / 27.12 GB/s decompression at saturation, 4.15 /
5.31 / 12.19 at small batch): 8 KB ring ×0.720 / ×0.771 / ×0.732 (LDS per block 5.2 → 9.3 KB:
fewer resident blocks; the ring depth is not the bottleneck) — REJECT; 4-sector granules
×0.885 / ×0.737 / ×0.990 (x0 ×0.891 / ×0.688) — REJECT; `LITERAL_SECTORS=8` ×1.009 / ×1.039 /
×0.999 (x0 ×1.002 / ×1.099 / ×1.000) — POSITIVE on literal-heavy data (more bytes in flight per
literal step with SNP-D8's dword path); sleeps 0/0/0 ×0.992 / ×0.993 / ×0.999 — neutral (see D1).
Verdict: candidate to make `LITERAL_SECTORS=8` the AMD default once gfx942 confirms; ring and
granule stay as inherited.

### SNP-D8k — knobs to measure D8's two mechanisms separately                          Category: C3   Status: KNOBS (defaults = D8 as measured)
Commit: (this commit)  (branch `opt/snappy-decomp-2026-08`)
Files: `src/snappy/decompression_process.hiph`
Change: `ARCTO_SNAPPY_DWORD_LITERALS` and `ARCTO_SNAPPY_DWORD_COPIES` (both 1 by default, AMD only)
select the dword literal path and the dword non-overlapping-copy path independently. Default
build identical to SNP-D8. Variants to run on gfx942 and gfx1100: `…_COPIES=0` and `…_LITERALS=0`.
Prediction (from the gfx942 sweep-5 pattern): copies=0 recovers the synth_binary loss on gfx942
and keeps the TTI gain; literals=0 loses the TTI gain.
Measured: gfx1100 (30 reps; default = literals+copies 109.3 / 117.8 / 27.08 GB/s decompression at saturation, 4.15 / 5.32 / 12.19 at small batch).
Result (gfx1100): COPIES=0 → binary ×0.920 (x0 ×0.920), tti ×1.005, words ×1.014 (x0 ×1.020); LITERALS=0 → binary ×0.973, tti ×0.858 (x0 ×0.878), words ×1.005. I.e. on gfx1100 the dword literal path is worth +14 % on the literal-heavy TTI and the dword copy path +8 % on the copy-heavy synth_binary at a 1.5–2 % cost on words; bytes identical. gfx942 knob run pending (the sweep-5 pattern there — TTI gain, synth_binary −20 % — points at the copy path).
Verdict: (pending gfx942) expected split: literals on everywhere, copies gated to wave32 (`USE_WARPSIZE_32`).
