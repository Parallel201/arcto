/*
 * ARCTO ZFP wrapper -- delegates to the canonical LLNL/zfp HIP backend
 * (vendored at third_party/zfp).
 *
 * The wrapper is intentionally thin: every encoder/decoder choice that
 * affects the bitstream is delegated to zfp_compress / zfp_decompress, so
 * the output is byte-for-byte identical to what `zfp -h -z` would produce
 * on the same input. Device pointers are passed through unchanged --
 * canonical's HIP backend autodetects them via is_gpu_ptr() and runs the
 * kernel zero-copy.
 *
 * Copyright (C) 2026 Cristiano Künas. Licensed under the MIT license.
 */

#include "arcto/zfp.h"
#include "zfp.h"

#include <climits>
#include <cstdlib>
#include <cstring>
#include <new>
#include <vector>

// Match the vendored zfp's GPU execution policy to our build backend.
#if defined(__HIP_PLATFORM_NVIDIA__) || defined(__HIP_PLATFORM_NVCC__)
  #define ARCTO_ZFP_EXEC zfp_exec_cuda
#else
  #define ARCTO_ZFP_EXEC zfp_exec_hip
#endif

namespace {

zfp_type to_zfp_type(arctoZFPType_t t)
{
  switch (t) {
    case ARCTO_ZFP_TYPE_FLOAT:  return zfp_type_float;
    case ARCTO_ZFP_TYPE_DOUBLE: return zfp_type_double;
    case ARCTO_ZFP_TYPE_INT32:  return zfp_type_int32;
    case ARCTO_ZFP_TYPE_INT64:  return zfp_type_int64;
  }
  return zfp_type_none;
}

zfp_field* make_field(const arctoZFPOpts_t& opts, void* data)
{
  const zfp_type t = to_zfp_type(opts.type);
  if (t == zfp_type_none) return nullptr;
  switch (opts.ndims) {
    case 1: return zfp_field_1d(data, t, opts.shape[0]);
    case 2: return zfp_field_2d(data, t, opts.shape[0], opts.shape[1]);
    case 3: return zfp_field_3d(data, t, opts.shape[0], opts.shape[1], opts.shape[2]);
    case 4: return zfp_field_4d(data, t, opts.shape[0], opts.shape[1], opts.shape[2], opts.shape[3]);
  }
  return nullptr;
}

// Configure the zfp_stream's mode + execution policy. Returns false if any
// step fails.
//
// FIXED_RATE          -> deterministic per-block size; no canonical index
//                        needed on the HIP path.
// FIXED_PRECISION /   -> variable-rate; the HIP decoder requires a block-
// FIXED_ACCURACY         offset index. Phase 4 embeds that index as a
//                        trailer on our compressed buffer so the decoder
//                        can rebuild it without an out-of-band file.
// REVERSIBLE          -> NOT supported on the canonical HIP backend at all
//                        (its compress/decompress switch on
//                        zfp_mode_reversible falls through to the "mode
//                        not supported on GPU" branch).
bool configure_stream(zfp_stream* zfp, const arctoZFPOpts_t& opts)
{
  const zfp_type t = to_zfp_type(opts.type);
  switch (opts.mode) {
    case ARCTO_ZFP_MODE_FIXED_RATE:
      // align=zfp_false matches what the upstream CLI sets for -r.
      zfp_stream_set_rate(zfp, opts.param, t, opts.ndims, zfp_false);
      break;
    case ARCTO_ZFP_MODE_FIXED_PRECISION:
      zfp_stream_set_precision(zfp, static_cast<uint>(opts.param));
      break;
    case ARCTO_ZFP_MODE_FIXED_ACCURACY:
      zfp_stream_set_accuracy(zfp, opts.param);
      break;
    case ARCTO_ZFP_MODE_REVERSIBLE:
    default:
      return false;
  }
  if (!zfp_stream_set_execution(zfp, ARCTO_ZFP_EXEC)) return false;
  return true;
}

// Variable-rate modes require an offset index that the HIP decoder reads
// alongside the bitstream. FIXED_RATE does not -- the block size is
// constant so block N starts at maxbits * N.
bool mode_needs_index(arctoZFPMode_t mode)
{
  switch (mode) {
    case ARCTO_ZFP_MODE_FIXED_PRECISION:
    case ARCTO_ZFP_MODE_FIXED_ACCURACY:
      return true;
    case ARCTO_ZFP_MODE_FIXED_RATE:
    case ARCTO_ZFP_MODE_REVERSIBLE:
    default:
      return false;
  }
}

// Trailer layout (always at the very end of the output buffer when present):
//   bytes [tail-20 .. tail-17]:  trailer payload (1 type + 3 reserved)
//   bytes [tail-16 .. tail-13]:  granularity (uint32_le)
//   bytes [tail-12 .. tail-5]:   index byte size (uint64_le)
//   bytes [tail-4  .. tail-1]:   magic "AZFP" (uint32_le 0x50465A41)
//
// The 16 bytes immediately preceding the trailer hold the canonical's
// own encoded index data (uint64 offsets prefixed by a small 8-byte
// type+granularity header that zfp_decompress also writes itself; we
// keep it verbatim so the round-trip is bit-for-bit symmetric).
constexpr uint32_t kArctoZFPTrailerMagic = 0x50465A41u;  // 'A''Z''F''P' LSB
constexpr size_t   kArctoZFPTrailerBytes = 20u;

void write_u32_le(unsigned char* p, uint32_t v)
{
  p[0] = uint8_t(v); p[1] = uint8_t(v >> 8);
  p[2] = uint8_t(v >> 16); p[3] = uint8_t(v >> 24);
}
void write_u64_le(unsigned char* p, uint64_t v)
{
  for (int i = 0; i < 8; i++) p[i] = uint8_t(v >> (8 * i));
}
uint32_t read_u32_le(const unsigned char* p)
{
  return uint32_t(p[0]) | (uint32_t(p[1]) << 8)
       | (uint32_t(p[2]) << 16) | (uint32_t(p[3]) << 24);
}
uint64_t read_u64_le(const unsigned char* p)
{
  uint64_t v = 0;
  for (int i = 0; i < 8; i++) v |= uint64_t(p[i]) << (8 * i);
  return v;
}

} // namespace

extern "C" {

arctoStatus_t arctoZFPCompressGetMaxOutputSize(
    arctoZFPOpts_t opts, size_t* max_bytes)
{
  if (!max_bytes) return arctoErrorInvalidValue;

  zfp_field* field = make_field(opts, nullptr);
  if (!field) return arctoErrorInvalidValue;

  zfp_stream* zfp = zfp_stream_open(nullptr);
  if (!zfp) { zfp_field_free(field); return arctoErrorInternal; }

  if (!configure_stream(zfp, opts)) {
    zfp_field_free(field);
    zfp_stream_close(zfp);
    return arctoErrorInvalidValue;
  }

  // Payload upper bound (bits). The header is metadata-only and capped at
  // a few bytes; we add ZFP_HEADER_MAX_BITS conservatively.
  const size_t payload_bytes = zfp_stream_maximum_size(zfp, field);
  const size_t header_bytes  = (ZFP_HEADER_MAX_BITS + CHAR_BIT - 1) / CHAR_BIT;

  size_t index_overhead = 0;
  if (mode_needs_index(opts.mode)) {
    // encode_index_offset writes (blocks/granularity + 1) uint64s. We use
    // granularity = 1 for maximum decoder flexibility.
    size_t blocks = 1;
    for (uint32_t d = 0; d < opts.ndims; d++) {
      blocks *= (opts.shape[d] + 3u) / 4u;
    }
    index_overhead = (blocks + 1) * sizeof(uint64_t) + kArctoZFPTrailerBytes;
  }

  *max_bytes = payload_bytes + header_bytes + index_overhead;

  zfp_field_free(field);
  zfp_stream_close(zfp);
  return arctoSuccess;
}

// Build the canonical ZFP_HEADER_FULL into a small host buffer. Returns the
// number of bytes consumed (always <= sizeof(out)) or 0 on failure.
//
// Used so the compress / decompress paths can keep the header and the
// payload in physically separate buffers. The canonical's HIP encode
// kernel writes to bit 0 of its device-side stream buffer and cleanup_device
// later overwrites whatever the CPU bitstream wrote there -- so we can't
// share a bitstream between zfp_write_header and zfp_compress on the HIP
// execution path.
size_t serialize_header(const arctoZFPOpts_t& opts,
                        const zfp_field* field,
                        unsigned char out[/*at least ZFP_HEADER_MAX_BYTES*/])
{
  constexpr size_t kMaxHeader = (ZFP_HEADER_MAX_BITS + CHAR_BIT - 1) / CHAR_BIT;
  std::memset(out, 0, kMaxHeader);

  bitstream* bs = stream_open(out, kMaxHeader);
  if (!bs) return 0;
  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); return 0; }

  if (!configure_stream(zfp, opts)) {
    zfp_stream_close(zfp);
    stream_close(bs);
    return 0;
  }

  const size_t header_bits = zfp_write_header(zfp, field, ZFP_HEADER_FULL);
  zfp_stream_flush(zfp);
  zfp_stream_close(zfp);
  stream_close(bs);

  if (header_bits == 0) return 0;
  return (header_bits + CHAR_BIT - 1) / CHAR_BIT;
}

arctoStatus_t arctoZFPCompress(
    const void* d_input,
    arctoZFPOpts_t opts,
    void* h_output, size_t h_output_capacity,
    size_t* actual_size_out)
{
  if (!d_input || !h_output || !actual_size_out) {
    return arctoErrorInvalidValue;
  }

  // Field describing the source. The canonical accepts a device pointer
  // here -- is_gpu_ptr() in src/hip/device.h detects it and uses the
  // buffer directly. We const_cast because zfp_field_*d() takes a non-const
  // void*; the compress path never writes through this pointer.
  zfp_field* field = make_field(opts, const_cast<void*>(d_input));
  if (!field) return arctoErrorInvalidValue;

  // 1. Serialize the full header into a tiny host buffer.
  constexpr size_t kMaxHeader = (ZFP_HEADER_MAX_BITS + CHAR_BIT - 1) / CHAR_BIT;
  unsigned char header_buf[kMaxHeader];
  const size_t header_bytes = serialize_header(opts, field, header_buf);
  if (header_bytes == 0 || header_bytes > h_output_capacity) {
    zfp_field_free(field);
    return arctoErrorInternal;
  }

  // 2. Encode the payload AFTER the reserved header region. The HIP encode
  //    kernel writes from bit 0 of the device-side stream buffer, so
  //    pointing the bitstream at h_output + header_bytes keeps the payload
  //    cleanly past the header with no memmove.
  bitstream* bs = stream_open(static_cast<unsigned char*>(h_output) + header_bytes,
                              h_output_capacity - header_bytes);
  if (!bs) { zfp_field_free(field); return arctoErrorInternal; }
  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) {
    zfp_field_free(field);
    stream_close(bs);
    return arctoErrorInternal;
  }
  if (!configure_stream(zfp, opts)) {
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    return arctoErrorInvalidValue;
  }

  // Variable-rate modes need a block-offset index alongside the bitstream
  // so the HIP decoder can seek into each block. Attach a fresh zfp_index
  // (type = offset, granularity = 1); zfp_compress fills it.
  zfp_index* idx = nullptr;
  if (mode_needs_index(opts.mode)) {
    idx = zfp_index_create();
    if (!idx) {
      zfp_field_free(field);
      zfp_stream_close(zfp);
      stream_close(bs);
      return arctoErrorInternal;
    }
    zfp_index_set_type(idx, zfp_index_offset, 1u);
    zfp_stream_set_index(zfp, idx);
  }

  const size_t payload_bytes = zfp_compress(zfp, field);
  if (payload_bytes == 0) {
    if (idx) { if (idx->data) std::free(idx->data); zfp_index_free(idx); }
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    return arctoErrorInternal;
  }

  // 3. Append the encoded index (if any) and a fixed-size trailer.
  size_t total = header_bytes + payload_bytes;
  if (idx && idx->data && idx->size > 0) {
    const size_t need = total + idx->size + kArctoZFPTrailerBytes;
    if (need > h_output_capacity) {
      std::free(idx->data);
      zfp_index_free(idx);
      zfp_field_free(field);
      zfp_stream_close(zfp);
      stream_close(bs);
      return arctoErrorInvalidValue;
    }
    auto* out = static_cast<unsigned char*>(h_output);
    std::memcpy(out + total, idx->data, idx->size);
    total += idx->size;

    // Trailer (20 bytes), positioned at out[total .. total+19]:
    out[total + 0] = static_cast<uint8_t>(idx->type);
    out[total + 1] = 0;
    out[total + 2] = 0;
    out[total + 3] = 0;
    write_u32_le(out + total + 4,  idx->granularity);
    write_u64_le(out + total + 8,  static_cast<uint64_t>(idx->size));
    write_u32_le(out + total + 16, kArctoZFPTrailerMagic);
    total += kArctoZFPTrailerBytes;

    std::free(idx->data);
  }
  if (idx) zfp_index_free(idx);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);

  // 4. Prepend the serialized header.
  std::memcpy(h_output, header_buf, header_bytes);
  *actual_size_out = total;
  return arctoSuccess;
}

// Read the canonical ZFP_HEADER_FULL from the front of h_input into `field`,
// and return the number of header BYTES consumed via *header_bytes_out,
// along with the compression mode the header advertises. The caller owns
// and must free `field`.
bool parse_header(const void* h_input, size_t h_input_size,
                  zfp_field*& field, size_t* header_bytes_out,
                  zfp_mode* mode_out)
{
  bitstream* bs = stream_open(const_cast<void*>(h_input), h_input_size);
  if (!bs) return false;
  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); return false; }

  field = zfp_field_alloc();
  if (!field) { zfp_stream_close(zfp); stream_close(bs); return false; }

  const size_t header_bits = zfp_read_header(zfp, field, ZFP_HEADER_FULL);
  if (mode_out) *mode_out = zfp_stream_compression_mode(zfp);
  zfp_stream_close(zfp);
  stream_close(bs);

  if (header_bits == 0) {
    zfp_field_free(field);
    field = nullptr;
    return false;
  }
  *header_bytes_out = (header_bits + CHAR_BIT - 1) / CHAR_BIT;
  return true;
}

// Variable-rate canonical stream with no ARCTO index trailer (e.g.,
// produced by the upstream CLI). The HIP decoder cannot process those
// directly; we fall back to canonical's serial decoder into a temporary
// host buffer and then copy the result to the caller's device buffer.
arctoStatus_t decompress_serial_fallback(
    const void* h_input, size_t h_input_size,
    void* d_output, size_t d_output_capacity,
    size_t* actual_size_out)
{
  bitstream* bs = stream_open(const_cast<void*>(h_input), h_input_size);
  if (!bs) return arctoErrorInternal;
  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); return arctoErrorInternal; }

  zfp_field* field = zfp_field_alloc();
  if (!field) { zfp_stream_close(zfp); stream_close(bs); return arctoErrorInternal; }

  if (zfp_read_header(zfp, field, ZFP_HEADER_FULL) == 0) {
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    return arctoErrorCannotDecompress;
  }

  const size_t need = zfp_field_size_bytes(field);
  if (need > d_output_capacity) {
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    return arctoErrorInvalidValue;
  }

  std::vector<unsigned char> h_temp(need);
  zfp_field_set_pointer(field, h_temp.data());
  zfp_stream_set_execution(zfp, zfp_exec_serial);

  const size_t bytes_read = zfp_decompress(zfp, field);
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);

  if (bytes_read == 0) return arctoErrorCannotDecompress;

  hipError_t herr = hipMemcpy(d_output, h_temp.data(), need, hipMemcpyHostToDevice);
  if (herr != hipSuccess) return arctoErrorInternal;

  *actual_size_out = need;
  return arctoSuccess;
}

arctoStatus_t arctoZFPDecompress(
    const void* h_input, size_t h_input_size,
    void* d_output, size_t d_output_capacity,
    size_t* actual_size_out)
{
  if (!h_input || !d_output || !actual_size_out) {
    return arctoErrorInvalidValue;
  }

  // 1. Parse the header on host to learn shape/type/mode and how many
  //    bytes the header occupied.
  zfp_field* field = nullptr;
  size_t header_bytes = 0;
  zfp_mode header_mode = zfp_mode_null;
  if (!parse_header(h_input, h_input_size, field, &header_bytes, &header_mode)) {
    return arctoErrorCannotDecompress;
  }

  const size_t need = zfp_field_size_bytes(field);
  if (need > d_output_capacity) {
    zfp_field_free(field);
    return arctoErrorInvalidValue;
  }
  if (header_bytes > h_input_size) {
    zfp_field_free(field);
    return arctoErrorCannotDecompress;
  }

  // 2. Sniff a trailing index, if present. A FIXED_RATE stream has none,
  //    and that includes anything produced by an upstream zfp CLI run;
  //    the magic check makes the trailer self-identifying.
  size_t  index_bytes        = 0;     // bytes occupied by canonical index data
  uint8_t trailer_index_type = 0;
  uint32_t trailer_granularity = 1;
  const auto* in = static_cast<const unsigned char*>(h_input);
  if (h_input_size >= header_bytes + kArctoZFPTrailerBytes) {
    const unsigned char* tail = in + h_input_size - kArctoZFPTrailerBytes;
    if (read_u32_le(tail + 16) == kArctoZFPTrailerMagic) {
      trailer_index_type  = tail[0];
      trailer_granularity = read_u32_le(tail + 4);
      const uint64_t idx_size_u64 = read_u64_le(tail + 8);
      // Sanity: index data must fit between the payload region and trailer.
      if (idx_size_u64 + kArctoZFPTrailerBytes
            <= h_input_size - header_bytes) {
        index_bytes = static_cast<size_t>(idx_size_u64);
      } else {
        zfp_field_free(field);
        return arctoErrorCannotDecompress;
      }
    }
  }

  // 2a. Variable-rate stream without our index trailer (e.g., produced by
  //     the upstream `zfp` CLI). The HIP decoder cannot navigate the
  //     variable-length blocks without an index, so fall back to the
  //     canonical serial decoder into a host buffer + hipMemcpy. The
  //     resulting d_output content is the same; only the data path differs.
  const bool needs_index =
      (header_mode == zfp_mode_fixed_precision)
   || (header_mode == zfp_mode_fixed_accuracy)
   || (header_mode == zfp_mode_expert);
  if (needs_index && index_bytes == 0) {
    zfp_field_free(field);
    return decompress_serial_fallback(h_input, h_input_size,
                                      d_output, d_output_capacity,
                                      actual_size_out);
  }

  const size_t payload_bytes =
      h_input_size - header_bytes - index_bytes
      - (index_bytes ? kArctoZFPTrailerBytes : 0);

  // 3. Set up a NEW stream that wraps ONLY the payload portion of h_input.
  //    The HIP decode kernel reads from bit 0 of its device-side stream
  //    buffer (mirror of the compress path), so feeding it the payload
  //    pointer directly is what gets the right bytes into the kernel.
  bitstream* bs = stream_open(const_cast<unsigned char*>(in) + header_bytes,
                              payload_bytes);
  if (!bs) { zfp_field_free(field); return arctoErrorInternal; }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) {
    zfp_field_free(field);
    stream_close(bs);
    return arctoErrorInternal;
  }

  // Replay header parsing on THIS stream so its mode / minbits / maxbits /
  // maxprec / minexp match what the encoder used. parse_header() above
  // populated a throwaway zfp_stream that is now gone; we copy its
  // settings here via the (smaller) zfp_stream_set_params API.
  bitstream* hs = stream_open(const_cast<void*>(h_input), h_input_size);
  zfp_stream* hz = zfp_stream_open(hs);
  zfp_field* throwaway = zfp_field_alloc();
  zfp_read_header(hz, throwaway, ZFP_HEADER_FULL);
  zfp_stream_set_params(zfp, hz->minbits, hz->maxbits, hz->maxprec, hz->minexp);
  zfp_field_free(throwaway);
  zfp_stream_close(hz);
  stream_close(hs);

  // Reconstruct and attach the index, if the stream had one.
  zfp_index* idx = nullptr;
  if (index_bytes > 0) {
    idx = zfp_index_create();
    if (!idx) {
      zfp_field_free(field);
      zfp_stream_close(zfp);
      stream_close(bs);
      return arctoErrorInternal;
    }
    zfp_index_set_type(idx,
        static_cast<zfp_index_type>(trailer_index_type),
        trailer_granularity);
    // The HIP decoder reads from idx->data via setup_device_index_decompress
    // which copies host->device if not already on the GPU. Point at the
    // index bytes in our input buffer; no copy on our side.
    zfp_index_set_data(idx,
        const_cast<unsigned char*>(in)
            + header_bytes + payload_bytes,
        index_bytes);
    zfp_stream_set_index(zfp, idx);
  }

  zfp_field_set_pointer(field, d_output);

  if (!zfp_stream_set_execution(zfp, ARCTO_ZFP_EXEC)) {
    if (idx) zfp_index_free(idx);
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    return arctoErrorInternal;
  }

  const size_t bytes_read = zfp_decompress(zfp, field);

  if (idx) zfp_index_free(idx);
  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);

  if (bytes_read == 0) return arctoErrorCannotDecompress;
  *actual_size_out = need;
  return arctoSuccess;
}

arctoStatus_t arctoZFPGetDecompressSize(
    const void* h_input, size_t h_input_size, size_t* uncomp_bytes)
{
  if (!h_input || !uncomp_bytes) return arctoErrorInvalidValue;

  zfp_field* field = nullptr;
  size_t header_bytes = 0;
  if (!parse_header(h_input, h_input_size, field, &header_bytes, nullptr)) {
    return arctoErrorCannotDecompress;
  }
  *uncomp_bytes = zfp_field_size_bytes(field);
  zfp_field_free(field);
  return arctoSuccess;
}

} // extern "C"
