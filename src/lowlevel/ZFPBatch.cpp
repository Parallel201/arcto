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
#include <cstring>
#include <new>

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
// step fails (e.g. HIP backend not compiled in).
//
// Phase 3 limitation: only ARCTO_ZFP_MODE_FIXED_RATE is supported on the
// HIP path. Variable-rate modes (FIXED_PRECISION, FIXED_ACCURACY,
// REVERSIBLE) produce blocks of varying size, and the canonical HIP
// decompressor requires a precomputed block-offset index that we don't yet
// serialize alongside the stream. A follow-up commit will embed that index
// in the output to enable the variable-rate modes on GPU.
bool configure_stream(zfp_stream* zfp, const arctoZFPOpts_t& opts)
{
  const zfp_type t = to_zfp_type(opts.type);
  switch (opts.mode) {
    case ARCTO_ZFP_MODE_FIXED_RATE:
      // align=zfp_false matches what the upstream CLI sets for -r.
      zfp_stream_set_rate(zfp, opts.param, t, opts.ndims, zfp_false);
      break;
    case ARCTO_ZFP_MODE_FIXED_PRECISION:
    case ARCTO_ZFP_MODE_FIXED_ACCURACY:
    case ARCTO_ZFP_MODE_REVERSIBLE:
      // Documented above -- not yet plumbed through.
      return false;
    default:
      return false;
  }
  // We always target HIP. The canonical also has zfp_exec_serial and
  // zfp_exec_omp; expose those later if the use case appears.
  if (!zfp_stream_set_execution(zfp, zfp_exec_hip)) return false;
  return true;
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
  *max_bytes = payload_bytes + header_bytes;

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

  const size_t payload_bytes = zfp_compress(zfp, field);

  zfp_field_free(field);
  zfp_stream_close(zfp);
  stream_close(bs);

  if (payload_bytes == 0) return arctoErrorInternal;

  // 3. Prepend the serialized header.
  std::memcpy(h_output, header_buf, header_bytes);
  *actual_size_out = header_bytes + payload_bytes;
  return arctoSuccess;
}

// Read the canonical ZFP_HEADER_FULL from the front of h_input into `field`,
// and return the number of header BYTES consumed via *header_bytes_out.
// `zfp_out` (the configured stream) and `field` are owned by the caller and
// freed by it. Returns true on success.
bool parse_header(const void* h_input, size_t h_input_size,
                  zfp_field*& field, size_t* header_bytes_out)
{
  bitstream* bs = stream_open(const_cast<void*>(h_input), h_input_size);
  if (!bs) return false;
  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) { stream_close(bs); return false; }

  field = zfp_field_alloc();
  if (!field) { zfp_stream_close(zfp); stream_close(bs); return false; }

  const size_t header_bits = zfp_read_header(zfp, field, ZFP_HEADER_FULL);
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
  if (!parse_header(h_input, h_input_size, field, &header_bytes)) {
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

  // 2. Set up a NEW stream that wraps ONLY the payload portion of h_input.
  //    The HIP decode kernel reads from bit 0 of its device-side stream
  //    buffer (mirror of the compress path), so feeding it the payload
  //    pointer directly is what gets the right bytes into the kernel.
  bitstream* bs = stream_open(
      static_cast<unsigned char*>(const_cast<void*>(h_input)) + header_bytes,
      h_input_size - header_bytes);
  if (!bs) { zfp_field_free(field); return arctoErrorInternal; }

  zfp_stream* zfp = zfp_stream_open(bs);
  if (!zfp) {
    zfp_field_free(field);
    stream_close(bs);
    return arctoErrorInternal;
  }

  // The stream's mode/rate/etc. were populated when parse_header() opened
  // its temporary zfp_stream; that stream is gone now, so we have to apply
  // the same configuration here from the field metadata. The arcto opts
  // were derived from the same header that parse_header just read, but to
  // stay self-contained, recover the original ZFP mode params directly.
  // Simplest correct path: serialize the header again into a temp buffer
  // backed by this stream so zfp_read_header populates THIS zfp_stream.
  // That is wasteful but unambiguous.
  bitstream* hs = stream_open(const_cast<void*>(h_input), h_input_size);
  zfp_stream* hz = zfp_stream_open(hs);
  zfp_field* throwaway = zfp_field_alloc();
  zfp_read_header(hz, throwaway, ZFP_HEADER_FULL);
  // Copy stream config (mode/precision/rate/etc.) to our payload stream.
  // The canonical exposes this via zfp_stream_compression_mode + the
  // matching set_* functions, but the lowest-friction route is to use
  // zfp_stream_set_params which takes raw min/max/prec/exp.
  zfp_stream_set_params(zfp, hz->minbits, hz->maxbits, hz->maxprec, hz->minexp);
  zfp_field_free(throwaway);
  zfp_stream_close(hz);
  stream_close(hs);

  zfp_field_set_pointer(field, d_output);

  if (!zfp_stream_set_execution(zfp, zfp_exec_hip)) {
    zfp_field_free(field);
    zfp_stream_close(zfp);
    stream_close(bs);
    return arctoErrorInternal;
  }

  const size_t bytes_read = zfp_decompress(zfp, field);

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
  if (!parse_header(h_input, h_input_size, field, &header_bytes)) {
    return arctoErrorCannotDecompress;
  }
  *uncomp_bytes = zfp_field_size_bytes(field);
  zfp_field_free(field);
  return arctoSuccess;
}

} // extern "C"
