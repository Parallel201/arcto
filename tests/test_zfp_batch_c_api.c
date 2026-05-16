/*
 * ARCTO ZFP — C API round-trip test.
 *
 * Validates compress -> decompress equality on byte streams using the
 * standard GENERATE_TESTS macro. Phase 0 exercises the passthrough stub;
 * the same test will validate the real ZFP kernel in later phases (lossless
 * mode) without modification.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#define CRASH_SAFE
#define SUPPORT_NULLPTR_APIS

#include "arcto/zfp.h"
#include "test_batch_c_api.h"

GENERATE_TESTS(ZFP);
