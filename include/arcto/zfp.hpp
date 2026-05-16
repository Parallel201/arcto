#pragma once

/*
 * ARCTO ZFP — C++ high-level types.
 *
 * The ZFP entry point is the direct single-cube API declared in arcto/zfp.h
 * (arctoZFP3DCompressAsync / arctoZFP3DDecompressAsync). Unlike the
 * batched-API formats (LZ4, Snappy, Cascaded, ...), the ZFP on-disk format
 * does not embed a CommonHeader and is therefore not autodetectable by
 * create_manager(). The ZFPManager class below is reserved as a thin
 * convenience wrapper around the direct API; future work may either
 * (a) refactor the ZFP format to start with a CommonHeader so the
 *     factory dispatch works, or
 * (b) build a full BatchManager<ZFPFormatSpecHeader> with its own HLIF
 *     kernels (mirroring LZ4Manager / SnappyManager).
 *
 * For now this header exists so symbol-level integrations (and the
 * forthcoming Manager work) can compile against a stable type.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include <memory>

#include "arcto/zfp.h"
#include "arctoManager.hpp"

namespace arcto {

/**
 * @brief Per-buffer format header for ZFP.
 *
 * Mirrors the on-the-wire fields the direct API stores at bytes 4..23 of
 * its compressed buffer (rate, mode, dim, shape). Exposed here for
 * symmetry with LZ4FormatSpecHeader, SnappyFormatSpecHeader, etc., and
 * for use by any future BatchManager<ZFPFormatSpecHeader> integration.
 */
struct ZFPFormatSpecHeader {
  arctoZFPMode_t mode;
  arctoZFPDim_t  dim;
  arctoType_t    data_type;
  double         param;     /* rate / precision / tolerance */
  uint32_t       shape[4];
};

/**
 * @brief Convenience handle around the single-cube 3D API. Not yet wired
 * to create_manager(); use arctoZFP3DCompressAsync directly today.
 */
struct ZFPManager : PimplManager {
  ZFPManager(arctoZFPMode_t  mode,
             arctoZFPDim_t   dim,
             double          param,
             uint32_t        nx,
             uint32_t        ny,
             uint32_t        nz,
             hipStream_t     user_stream = 0,
             const int       device_id   = 0);

  ~ZFPManager();
};

} // namespace arcto
