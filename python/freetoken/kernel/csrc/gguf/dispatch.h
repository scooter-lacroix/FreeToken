// Minimal AT_DISPATCH helper for the vendored GGUF kernels (borrowed from
// sgl-kernel csrc/quantization/gguf, which are ports of llama.cpp). The donor
// pulls these macros from its large include/utils.h; we only need the float
// dispatch, so vendor just that to keep the JIT compile self-contained.
#pragma once

#include <ATen/Dispatch.h>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif

// Warp-shuffle wrappers the donor pulls from sgl-kernel's utils.h (CUDA variants).
// RDNA GPUs run 64-wide waves while these kernels are written for 32-lane warps.
// HIP's masked __shfl_*_sync overloads require a 64-bit mask naming every
// participating lane -- widening the kernels' u32 mask silently EXCLUDES the
// upper 32-lane segment of each wave, failing those lanes' shuffles and
// corrupting the warp reductions (the Q8_1 activation quantizer reduces this
// way, which zeroed every MMVQ result on gfx1100). HIP's mask-less __shfl_xor
// segments by the explicit width=32 exactly as the CUDA originals intend, with
// no mask semantics to break.
#if defined(__HIP_PLATFORM_AMD__)
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) \
  __shfl_xor((var), (lane_mask), 32)
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor((var), (lane_mask), (width))
#endif
#else
#ifndef SGLANG_SHFL_XOR_SYNC
#define SGLANG_SHFL_XOR_SYNC(mask, var, lane_mask) __shfl_xor_sync((mask), (var), (lane_mask))
#endif
#ifndef SGLANG_SHFL_XOR_SYNC_WIDTH
#define SGLANG_SHFL_XOR_SYNC_WIDTH(mask, var, lane_mask, width) \
  __shfl_xor_sync((mask), (var), (lane_mask), (width))
#endif
#endif

#define DISPATCH_CASE_FLOAT_TYPES(...)                 \
  AT_DISPATCH_CASE(at::ScalarType::Float, __VA_ARGS__) \
  AT_DISPATCH_CASE(at::ScalarType::Half, __VA_ARGS__)  \
  AT_DISPATCH_CASE(at::ScalarType::BFloat16, __VA_ARGS__)

#define DISPATCH_FLOAT_TYPES(TYPE, NAME, ...) \
  AT_DISPATCH_SWITCH(TYPE, NAME, DISPATCH_CASE_FLOAT_TYPES(__VA_ARGS__))
