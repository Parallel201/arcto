/*
 * ARCTO ZFP — high-level Manager (Phase 2 stub).
 *
 * The ZFPManager constructor currently throws ArctoException with a
 * pointer to the direct single-cube API (arctoZFP3DCompressAsync). A
 * full BatchManager<ZFPFormatSpecHeader> integration is reserved for
 * a future phase; it requires either refactoring the ZFP on-disk
 * format to embed a CommonHeader (so create_manager() can autodetect
 * a ZFP buffer) or implementing dedicated ZFPHlifKernels that mirror
 * the LZ4Manager / SnappyManager pattern.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#pragma once

#include "arcto/zfp.hpp"

#include "Check.h"

namespace arcto {

inline ZFPManager::ZFPManager(
    arctoZFPMode_t  /*mode*/,
    arctoZFPDim_t   /*dim*/,
    double          /*param*/,
    uint32_t        /*nx*/,
    uint32_t        /*ny*/,
    uint32_t        /*nz*/,
    hipStream_t     /*user_stream*/,
    const int       /*device_id*/)
{
  throw ArctoException(
      arctoErrorNotSupported,
      "ZFPManager is not yet wired through the BatchManager factory; use "
      "arctoZFP3DCompressAsync / arctoZFP3DDecompressAsync directly for "
      "single-cube workloads.");
}

inline ZFPManager::~ZFPManager() = default;

} // namespace arcto
