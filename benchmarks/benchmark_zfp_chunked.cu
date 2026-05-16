/*
 * ARCTO ZFP — chunked benchmark binary.
 *
 * Phase 0 wires the passthrough stub through the standard chunked benchmark
 * template so the metrics pipeline (throughput, transfer time, stddev, etc.)
 * is exercised end-to-end before the real ZFP kernel lands.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "benchmark_template_chunked.cuh"

#include "arcto/zfp.h"

GENERATE_CHUNKED_BENCHMARK(
    arctoBatchedZFPCompressGetTempSize,
    arctoBatchedZFPCompressGetMaxOutputChunkSize,
    arctoBatchedZFPCompressAsync,
    arctoBatchedZFPDecompressGetTempSize,
    arctoBatchedZFPDecompressAsync,
    inputAlwaysValid,
    arctoBatchedZFPDefaultOpts);
