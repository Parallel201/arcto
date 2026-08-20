# ARCTO Licensing

ARCTO as a whole is distributed under the MIT license: [LICENSE](LICENSE).

Individual files derived from other projects keep their original licenses,
listed below. This file is generated from the actual per-file license
headers in the tree.

## Vendored zfp (LLNL) — our distribution

`third_party/zfp` is vendored as a git submodule pointing at **our
distribution** of LLNL zfp, `github.com/cristianokunas/zfp`, branch
`amd-hip`, forked from upstream at commit
`cccbb9d5e69a2998f63eb06153ca3b7b63901b56` (the commit this submodule has
always pinned). The fork exists to carry AMD/HIP-focused improvements ahead
of upstream; changes suitable for general use are intended to be offered
back to `LLNL/zfp` as pull requests.

zfp is BSD 3-Clause; its `LICENSE` is preserved in the subtree and
modifications in the fork are marked as such. Updating from upstream is a
deliberate act (merge of an upstream tag into `amd-hip`), never automatic:
the submodule gitlink pins the exact commit either way.

## Files subject to the BSD 3-Clause License (NVIDIA nvCOMP 2.2)

ARCTO is a HIP translation of NVIDIA nvCOMP v2.2 (BSD 3-Clause). The files
below are derived from nvCOMP sources; copyright years differ per file and
every file retains its original NVIDIA header.

* LICENSE FILE: [NVCOMP_2_2_LICENSE](NVCOMP_2_2_LICENSE)
* HOMEPAGE: https://github.com/NVIDIA/nvcomp/tree/branch-2.2
* UPSTREAM COMMIT: `a6e4e64a177e07cd2e5c8c5e07bb66ffefceae84` (tag `v2.2.0`, branch `branch-2.2`, 2022-02-07)

* FILES:
  * ``CMakeLists.txt``
  * ``benchmarks/CMakeLists.txt``
  * ``benchmarks/benchmark_ans_chunked.cu``
  * ``benchmarks/benchmark_bitcomp_chunked.cu``
  * ``benchmarks/benchmark_cascaded_chunked.cu``
  * ``benchmarks/benchmark_common.h``
  * ``benchmarks/benchmark_gdeflate_chunked.cu``
  * ``benchmarks/benchmark_lz4_chunked.cu``
  * ``benchmarks/benchmark_snappy_chunked.cu``
  * ``benchmarks/benchmark_template_chunked.cuh``
  * ``include/arcto.h``
  * ``include/arcto.hpp``
  * ``include/arcto/ans.h``
  * ``include/arcto/ans.hpp``
  * ``include/arcto/arctoManager.hpp``
  * ``include/arcto/arctoManagerFactory.hpp``
  * ``include/arcto/bitcomp.h``
  * ``include/arcto/bitcomp.hpp``
  * ``include/arcto/cascaded.h``
  * ``include/arcto/cascaded.hpp``
  * ``include/arcto/gdeflate.h``
  * ``include/arcto/gdeflate.hpp``
  * ``include/arcto/lz4.h``
  * ``include/arcto/lz4.hpp``
  * ``include/arcto/shared_types.h``
  * ``include/arcto/snappy.h``
  * ``include/arcto/snappy.hpp``
  * ``snappy/len3_mask_64.cpp``
  * ``src/BitPackGPU.h``
  * ``src/BitPackGPU.hip``
  * ``src/CMakeLists.txt``
  * ``src/CascadedKernels.hiph``
  * ``src/Check.cpp``
  * ``src/Check.h``
  * ``src/DeltaGPU.h``
  * ``src/DeltaGPU.hip``
  * ``src/HipUtils.h``
  * ``src/HipUtils.hip``
  * ``src/LZ4Kernels.hiph``
  * ``src/LZ4Types.h``
  * ``src/RunLengthEncodeGPU.h``
  * ``src/RunLengthEncodeGPU.hip``
  * ``src/TempSpaceBroker.cpp``
  * ``src/TempSpaceBroker.h``
  * ``src/arcto_api.cpp``
  * ``src/arcto_common_deps/hlif_shared.hiph``
  * ``src/arcto_common_deps/hlif_shared_types.hpp``
  * ``src/arcto_hipcub.hiph``
  * ``src/common.h``
  * ``src/highlevel/ANSManager.cpp``
  * ``src/highlevel/ANSManager.hpp``
  * ``src/highlevel/BatchManager.hpp``
  * ``src/highlevel/BitcompManager.hip``
  * ``src/highlevel/BitcompManager.hpp``
  * ``src/highlevel/CascadedHlifKernels.h``
  * ``src/highlevel/CascadedHlifKernels.hip``
  * ``src/highlevel/CascadedManager.cpp``
  * ``src/highlevel/CascadedManager.hpp``
  * ``src/highlevel/CompressionConfigs.cpp``
  * ``src/highlevel/CompressionConfigs.hpp``
  * ``src/highlevel/GdeflateBatchManager.hpp``
  * ``src/highlevel/GdeflateManager.cpp``
  * ``src/highlevel/LZ4HlifKernels.h``
  * ``src/highlevel/LZ4HlifKernels.hip``
  * ``src/highlevel/LZ4Manager.cpp``
  * ``src/highlevel/LZ4Manager.hpp``
  * ``src/highlevel/ManagerBase.hpp``
  * ``src/highlevel/PinnedPtrs.hpp``
  * ``src/highlevel/SnappyHlifKernels.h``
  * ``src/highlevel/SnappyHlifKernels.hip``
  * ``src/highlevel/SnappyManager.cpp``
  * ``src/highlevel/SnappyManager.hpp``
  * ``src/highlevel/arctoManagerFactory.cpp``
  * ``src/highlevel/test/PinnedPtrPool_test.cpp``
  * ``src/lowlevel/BitcompBatch.hip``
  * ``src/lowlevel/CascadedBatch.hip``
  * ``src/lowlevel/LZ4Batch.cpp``
  * ``src/lowlevel/LZ4CompressionKernels.h``
  * ``src/lowlevel/LZ4CompressionKernels.hip``
  * ``src/lowlevel/SnappyBatch.cpp``
  * ``src/lowlevel/SnappyBatchKernels.h``
  * ``src/lowlevel/SnappyBatchKernels.hip``
  * ``src/lowlevel/ansBatch.cpp``
  * ``src/lowlevel/gdeflateBatch.cpp``
  * ``src/lowlevel/gdeflateKernels.h``
  * ``src/lowlevel/gdeflateKernels.hip``
  * ``src/snappy/types.h``
  * ``src/test/BitPackGPU_test.cpp``
  * ``src/test/DeltaGPU_test.cpp``
  * ``src/test/HipUtils_test.cpp``
  * ``src/test/RunLengthEncodeGPU_test.cpp``
  * ``src/test/SnappyLargeTokens_test.cpp``
  * ``src/test/TempSpaceBroker_test.cpp``
  * ``src/type_macros.h``
  * ``src/unpack.h``
  * ``tests/CMakeLists.txt``
  * ``tests/test_ans_batch_c_api.c``
  * ``tests/test_batch_c_api.h``
  * ``tests/test_bitcomp.cpp``
  * ``tests/test_bitcomp_batch.cpp``
  * ``tests/test_bitcomp_batch_c_api.c``
  * ``tests/test_cascaded.cpp``
  * ``tests/test_cascaded_batch.cpp``
  * ``tests/test_cascadedbatch_c_api.c``
  * ``tests/test_common.h``
  * ``tests/test_gdeflate.cpp``
  * ``tests/test_gdeflate_batch_c_api.c``
  * ``tests/test_lz4.cpp``
  * ``tests/test_lz4batch_c_api.c``
  * ``tests/test_random_lz4.cpp``
  * ``tests/test_snappy_app.cpp``
  * ``tests/test_snappy_batch_c_api.c``

## Files subject to the Apache License 2.0 (NVIDIA nvCOMP 2.2)

Work derived from NVIDIA nvCOMP v2.2 that is licensed under Apache-2.0
(the Snappy device kernels originate from RAPIDS cuDF). Each file retains
its original header; copyright years differ per file.

* HOMEPAGE: https://github.com/NVIDIA/nvcomp/tree/branch-2.2
* UPSTREAM COMMIT: `a6e4e64a177e07cd2e5c8c5e07bb66ffefceae84` (tag `v2.2.0`, branch `branch-2.2`, 2022-02-07)

* FILES:
  * ``cmake/arcto-config.cmake.amd.in``
  * ``cmake/arcto-config.cmake.nvidia.in``
  * ``src/device_functions.hiph``
  * ``src/snappy/compression.hiph``
  * ``src/snappy/compression_state.hiph``
  * ``src/snappy/config.h``
  * ``src/snappy/decompression.hiph``
  * ``src/snappy/decompression_decode.hiph``
  * ``src/snappy/decompression_decode_strategies.hiph``
  * ``src/snappy/decompression_decode_warp_scans.hiph``
  * ``src/snappy/decompression_prefetch.hiph``
  * ``src/snappy/decompression_process.hiph``
  * ``src/snappy/decompression_state.hiph``
  * ``src/snappy/symbol.hiph``

## Files subject to the Boost Software License 1.0

* ``tests/catch.hpp`` (Catch2, Copyright (c) 2022 Two Blue Cubes Ltd.)

## Vendored third-party projects

* ``third_party/zfp`` (git submodule) — LLNL zfp, BSD 3-Clause.
  See the submodule's own LICENSE file.
  HOMEPAGE: https://github.com/LLNL/zfp

## Other files and modifications: MIT License

All other files — the ZFP integration (``src/lowlevel/ZFPBatch.cpp``,
``include/arcto/zfp.h``), the host staging APIs
(``arctoHostBatch``/``arctoHostBatchAdaptive``), their tests and benchmarks,
and the build system additions — are original to this project and subject
to [LICENSE](LICENSE) (MIT).

Wherever modifications have been applied to a derived file, the
modifications are subject to the MIT license while the original code lines
remain subject to the file's original license.
