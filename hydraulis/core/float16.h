#pragma once

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <cuda_fp16.h>
#include <cuda_fp16.hpp>
#include "hydraulis/common/logging.h"
#include "hydraulis/common/except.h"

#if defined(__CUDACC__)
#define HYDRAULIS_HOSTDEVICE __host__ __device__ __inline__
#define HYDRAULIS_HOST __host__  __inline__
#define HYDRAULIS_DEVICE __device__  __inline__
#else
#define HYDRAULIS_HOSTDEVICE inline
#define HYDRAULIS_DEVICE inline
#define HYDRAULIS_HOST inline
#endif /* defined(__CUDACC__) */

namespace hydraulis {
HYDRAULIS_HOSTDEVICE float fp32_from_bits(uint32_t bits) {
  union {
    uint32_t bits;
    float f;
  } fp32 = {bits};
  return fp32.f;
}

HYDRAULIS_HOSTDEVICE uint32_t fp32_to_bits(float f) {
  union {
    float f32;
    uint32_t bits;
  } fp32 = {f};
  return fp32.bits;
}

HYDRAULIS_HOSTDEVICE uint32_t fp16_ieee_to_fp32_bits(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t nonsign = w & UINT32_C(0x7FFFFFFF);
  uint32_t renorm_shift = __builtin_clz(nonsign);
  renorm_shift = renorm_shift > 5 ? renorm_shift - 5 : 0;
  const int32_t inf_nan_mask =
      ((int32_t)(nonsign + 0x04000000) >> 8) & INT32_C(0x7F800000);
  const int32_t zero_mask = (int32_t)(nonsign - 1) >> 31;
  return sign |
      ((((nonsign << renorm_shift >> 3) + ((0x70 - renorm_shift) << 23)) |
        inf_nan_mask) &
       ~zero_mask);
}

HYDRAULIS_HOSTDEVICE float fp16_to_fp32(uint16_t h) {
  const uint32_t w = (uint32_t)h << 16;
  const uint32_t sign = w & UINT32_C(0x80000000);
  const uint32_t two_w = w + w;
  constexpr uint32_t exp_offset = UINT32_C(0xE0) << 23;
  constexpr uint32_t scale_bits = (uint32_t)15 << 23;
  float exp_scale_val;
  std::memcpy(&exp_scale_val, &scale_bits, sizeof(exp_scale_val));
  const float exp_scale = exp_scale_val;
  const float normalized_value =
      fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;
  constexpr uint32_t magic_mask = UINT32_C(126) << 23;
  constexpr float magic_bias = 0.5f;
  const float denormalized_value =
      fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;
  constexpr uint32_t denormalized_cutoff = UINT32_C(1) << 27;
  const uint32_t result = sign |
      (two_w < denormalized_cutoff ? fp32_to_bits(denormalized_value)
                                   : fp32_to_bits(normalized_value));
  return fp32_from_bits(result);
}

HYDRAULIS_HOSTDEVICE uint16_t fp32_to_fp16(float f) {
  constexpr uint32_t scale_to_inf_bits = (uint32_t)239 << 23;
  constexpr uint32_t scale_to_zero_bits = (uint32_t)17 << 23;
  float scale_to_inf_val, scale_to_zero_val;
  std::memcpy(&scale_to_inf_val, &scale_to_inf_bits, sizeof(scale_to_inf_val));
  std::memcpy(
      &scale_to_zero_val, &scale_to_zero_bits, sizeof(scale_to_zero_val));
  const float scale_to_inf = scale_to_inf_val;
  const float scale_to_zero = scale_to_zero_val;

  float base = (fabsf(f) * scale_to_inf) * scale_to_zero;
  const uint32_t w = fp32_to_bits(f);
  const uint32_t shl1_w = w + w;
  const uint32_t sign = w & UINT32_C(0x80000000);
  uint32_t bias = shl1_w & UINT32_C(0xFF000000);
  if (bias < UINT32_C(0x71000000)) {
    bias = UINT32_C(0x71000000);
  }

  base = fp32_from_bits((bias >> 1) + UINT32_C(0x07800000)) + base;
  const uint32_t bits = fp32_to_bits(base);
  const uint32_t exp_bits = (bits >> 13) & UINT32_C(0x00007C00);
  const uint32_t mantissa_bits = bits & UINT32_C(0x00000FFF);
  const uint32_t nonsign = exp_bits + mantissa_bits;
  return static_cast<uint16_t>(
      (sign >> 16) |
      (shl1_w > UINT32_C(0xFF000000) ? UINT16_C(0x7E00) : nonsign));
}

struct alignas(2) float16 {
  unsigned short val;
  struct from_bits_t {};
  HYDRAULIS_HOSTDEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }
  float16() = default;
  HYDRAULIS_HOSTDEVICE constexpr float16(unsigned short bits, from_bits_t) : val(bits){};
  // HYDRAULIS_HOSTDEVICE float16(int value);
  // HYDRAULIS_HOSTDEVICE operator int() const;
  HYDRAULIS_HOSTDEVICE float16(float value);
  HYDRAULIS_HOSTDEVICE operator float() const;
  HYDRAULIS_HOSTDEVICE float16(double value);
  HYDRAULIS_HOSTDEVICE explicit operator double() const;
  HYDRAULIS_HOSTDEVICE explicit operator int() const;
  HYDRAULIS_HOSTDEVICE explicit operator int64_t() const;
  HYDRAULIS_HOSTDEVICE explicit operator size_t() const;
  HYDRAULIS_HOSTDEVICE float16(int value);
  HYDRAULIS_HOSTDEVICE float16(int64_t value);
  HYDRAULIS_HOSTDEVICE float16(size_t value);
  #if defined(__CUDACC__)
  HYDRAULIS_HOSTDEVICE float16(const __half& value);
  HYDRAULIS_HOSTDEVICE operator __half() const;
  HYDRAULIS_HOSTDEVICE __half to_half() const;
  HYDRAULIS_HOSTDEVICE float16 &operator=(const __half& value) { val = *reinterpret_cast<const unsigned short*>(&value); return *this; }
  #endif
  HYDRAULIS_HOSTDEVICE float16 &operator=(const float f) { val = float16(f).val; return *this; }
  HYDRAULIS_HOSTDEVICE float16 &operator=(const float16 h) { val = h.val; return *this; }
};

HYDRAULIS_HOSTDEVICE float16::float16(float value) {
  val = hydraulis::fp32_to_fp16(value);
}

HYDRAULIS_HOSTDEVICE float16::operator float() const {
  return hydraulis::fp16_to_fp32(val);
}

HYDRAULIS_HOSTDEVICE float16::float16(double value) {
  val = hydraulis::fp32_to_fp16(float(value));
}

HYDRAULIS_HOSTDEVICE float16::operator double() const {
  return static_cast<double>(hydraulis::fp16_to_fp32(val));
}

HYDRAULIS_HOSTDEVICE float16::operator int() const {
  return static_cast<int>(hydraulis::fp16_to_fp32(val));
}

HYDRAULIS_HOSTDEVICE float16::operator int64_t() const {
  return static_cast<int64_t>(hydraulis::fp16_to_fp32(val));
}

HYDRAULIS_HOSTDEVICE float16::operator size_t() const {
  return static_cast<size_t>(hydraulis::fp16_to_fp32(val));
}

HYDRAULIS_HOSTDEVICE float16::float16(int value) {
  val = hydraulis::fp32_to_fp16(float(value));
}

HYDRAULIS_HOSTDEVICE float16::float16(int64_t value) {
  val = hydraulis::fp32_to_fp16(float(value));
}

HYDRAULIS_HOSTDEVICE float16::float16(size_t value) {
  val = hydraulis::fp32_to_fp16(float(value));
}

#if defined(__CUDACC__)
HYDRAULIS_HOSTDEVICE float16::float16(const __half& value) {
  val = *reinterpret_cast<const unsigned short*>(&value);
}
HYDRAULIS_HOSTDEVICE float16::operator __half() const {
  return *reinterpret_cast<const __half*>(&val);
}
HYDRAULIS_HOSTDEVICE __half float16::to_half() const {
  return *reinterpret_cast<const __half*>(&val);
}
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 350)) || \
    (defined(__clang__) && defined(__CUDA__))
HYDRAULIS_HOSTDEVICE __device__ float16 __ldg(const float16* ptr) {
  return __ldg(reinterpret_cast<const __half*>(ptr));
}
#endif

/// Arithmetic
HYDRAULIS_DEVICE float16 operator+(const float16& a, const float16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

HYDRAULIS_DEVICE float16 operator-(const float16& a, const float16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

HYDRAULIS_DEVICE float16 operator*(const float16& a, const float16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

HYDRAULIS_DEVICE float16 operator/(const float16& a, const float16& b) {
  // HT_ASSERT(static_cast<float>(b) != 0)
  // << "Divided by zero.";
  return static_cast<float>(a) / static_cast<float>(b);
}

HYDRAULIS_DEVICE float16 operator-(const float16& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 530)
  return __hneg(a);
#else
  return -static_cast<float>(a);
#endif
}

HYDRAULIS_DEVICE float16& operator+=(float16& a, const float16& b) {
  a = a + b;
  return a;
}

HYDRAULIS_DEVICE float16& operator-=(float16& a, const float16& b) {
  a = a - b;
  return a;
}

HYDRAULIS_DEVICE float16& operator*=(float16& a, const float16& b) {
  a = a * b;
  return a;
}

HYDRAULIS_DEVICE float16& operator/=(float16& a, const float16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

HYDRAULIS_DEVICE float operator+(float16 a, float b) {
  return static_cast<float>(a) + b;
}
HYDRAULIS_DEVICE float operator-(float16 a, float b) {
  return static_cast<float>(a) - b;
}
HYDRAULIS_DEVICE float operator*(float16 a, float b) {
  return static_cast<float>(a) * b;
}
HYDRAULIS_DEVICE float operator/(float16 a, float b) {
  return static_cast<float>(a) / b;
}

HYDRAULIS_DEVICE float operator+(float a, float16 b) {
  return a + static_cast<float>(b);
}
HYDRAULIS_DEVICE float operator-(float a, float16 b) {
  return a - static_cast<float>(b);
}
HYDRAULIS_DEVICE float operator*(float a, float16 b) {
  return a * static_cast<float>(b);
}
HYDRAULIS_DEVICE float operator/(float a, float16 b) {
  return a / static_cast<float>(b);
}

HYDRAULIS_DEVICE float& operator+=(float& a, const float16& b) {
  return a += static_cast<float>(b);
}
HYDRAULIS_DEVICE float& operator-=(float& a, const float16& b) {
  return a -= static_cast<float>(b);
}
HYDRAULIS_DEVICE float& operator*=(float& a, const float16& b) {
  return a *= static_cast<float>(b);
}
HYDRAULIS_DEVICE float& operator/=(float& a, const float16& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

HYDRAULIS_DEVICE double operator+(float16 a, double b) {
  return static_cast<double>(a) + b;
}
HYDRAULIS_DEVICE double operator-(float16 a, double b) {
  return static_cast<double>(a) - b;
}
HYDRAULIS_DEVICE double operator*(float16 a, double b) {
  return static_cast<double>(a) * b;
}
HYDRAULIS_DEVICE double operator/(float16 a, double b) {
  return static_cast<double>(a) / b;
}

HYDRAULIS_DEVICE double operator+(double a, float16 b) {
  return a + static_cast<double>(b);
}
HYDRAULIS_DEVICE double operator-(double a, float16 b) {
  return a - static_cast<double>(b);
}
HYDRAULIS_DEVICE double operator*(double a, float16 b) {
  return a * static_cast<double>(b);
}
HYDRAULIS_DEVICE double operator/(double a, float16 b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

HYDRAULIS_DEVICE float16 operator+(float16 a, int b) {
  return a + static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator-(float16 a, int b) {
  return a - static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator*(float16 a, int b) {
  return a * static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator/(float16 a, int b) {
  return a / static_cast<float16>(b);
}

HYDRAULIS_DEVICE float16 operator+(int a, float16 b) {
  return static_cast<float16>(a) + b;
}
HYDRAULIS_DEVICE float16 operator-(int a, float16 b) {
  return static_cast<float16>(a) - b;
}
HYDRAULIS_DEVICE float16 operator*(int a, float16 b) {
  return static_cast<float16>(a) * b;
}
HYDRAULIS_DEVICE float16 operator/(int a, float16 b) {
  return static_cast<float16>(a) / b;
}

//// Arithmetic with int64_t

HYDRAULIS_DEVICE float16 operator+(float16 a, int64_t b) {
  return a + static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator-(float16 a, int64_t b) {
  return a - static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator*(float16 a, int64_t b) {
  return a * static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator/(float16 a, int64_t b) {
  return a / static_cast<float16>(b);
}

HYDRAULIS_DEVICE float16 operator+(int64_t a, float16 b) {
  return static_cast<float16>(a) + b;
}
HYDRAULIS_DEVICE float16 operator-(int64_t a, float16 b) {
  return static_cast<float16>(a) - b;
}
HYDRAULIS_DEVICE float16 operator*(int64_t a, float16 b) {
  return static_cast<float16>(a) * b;
}
HYDRAULIS_DEVICE float16 operator/(int64_t a, float16 b) {
  return static_cast<float16>(a) / b;
}

//// Arithmetic with size_t

HYDRAULIS_DEVICE float16 operator+(float16 a, size_t b) {
  return a + static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator-(float16 a, size_t b) {
  return a - static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator*(float16 a, size_t b) {
  return a * static_cast<float16>(b);
}
HYDRAULIS_DEVICE float16 operator/(float16 a, size_t b) {
  return a / static_cast<float16>(b);
}

HYDRAULIS_DEVICE float16 operator+(size_t a, float16 b) {
  return static_cast<float16>(a) + b;
}
HYDRAULIS_DEVICE float16 operator-(size_t a, float16 b) {
  return static_cast<float16>(a) - b;
}
HYDRAULIS_DEVICE float16 operator*(size_t a, float16 b) {
  return static_cast<float16>(a) * b;
}
HYDRAULIS_DEVICE float16 operator/(size_t a, float16 b) {
  return static_cast<float16>(a) / b;
}

std::ostream& operator<<(std::ostream& out, const float16& value);
} //namespace hydraulis

namespace std {

template <>
class numeric_limits<hydraulis::float16> {
 public:
  static constexpr bool is_specialized = true;
  static constexpr bool is_signed = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = true;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 11;
  static constexpr int digits10 = 3;
  static constexpr int max_digits10 = 5;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -13;
  static constexpr int min_exponent10 = -4;
  static constexpr int max_exponent = 16;
  static constexpr int max_exponent10 = 4;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr hydraulis::float16 min() {
    return hydraulis::float16(0x0400, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 lowest() {
    return hydraulis::float16(0xFBFF, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 max() {
    return hydraulis::float16(0x7BFF, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 epsilon() {
    return hydraulis::float16(0x1400, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 round_error() {
    return hydraulis::float16(0x3800, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 infinity() {
    return hydraulis::float16(0x7C00, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 quiet_NaN() {
    return hydraulis::float16(0x7E00, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 signaling_NaN() {
    return hydraulis::float16(0x7D00, hydraulis::float16::from_bits());
  }
  static constexpr hydraulis::float16 denorm_min() {
    return hydraulis::float16(0x0001, hydraulis::float16::from_bits());
  }
};

} //namespace std