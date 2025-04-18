#pragma once

#include <stdio.h>
#include <math.h>
#include <stdint.h>
#include <cstdint>
#include <cstring>
#include <iosfwd>
#include <limits>
#include <cuda_bf16.h>
#include <cuda_bf16.hpp>
#include "hydraulis/core/float16.h"
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
HYDRAULIS_HOSTDEVICE float fp32_from_bits16(uint16_t bits) {
  float res = 0;
  uint32_t tmp = bits;
  tmp <<= 16;
  std::memcpy(&res, &tmp, sizeof(tmp));
  return res;
}

HYDRAULIS_HOSTDEVICE uint16_t fp32_to_bits16(float f) {
  uint32_t res = 0;
  std::memcpy(&res, &f, sizeof(res));
  return res >> 16;
}

HYDRAULIS_HOSTDEVICE uint16_t fp32_to_bf16(float f) {
  if (std::isnan(f)) {
    return UINT16_C(0x7FC0);
  } else {
    union {
      uint32_t U32;
      float F32;
    };
    F32 = f;
    uint32_t rounding_bias = ((U32 >> 16) & 1) + UINT32_C(0x7FFF);
    return static_cast<uint16_t>((U32 + rounding_bias) >> 16);
  }
}

struct alignas(2) bfloat16 {
  unsigned short val;
  struct from_bits_t {};
  HYDRAULIS_HOSTDEVICE static constexpr from_bits_t from_bits() {
    return from_bits_t();
  }
  bfloat16() = default;
  HYDRAULIS_HOSTDEVICE constexpr bfloat16(unsigned short bits, from_bits_t) : val(bits){};
  // HYDRAULIS_HOSTDEVICE bfloat16(int value);
  // HYDRAULIS_HOSTDEVICE operator int() const;
  HYDRAULIS_HOSTDEVICE bfloat16(float value);
  HYDRAULIS_HOSTDEVICE operator float() const;
  HYDRAULIS_HOSTDEVICE bfloat16(double value);
  HYDRAULIS_HOSTDEVICE explicit operator float16() const;
  HYDRAULIS_HOSTDEVICE explicit operator double() const;
  HYDRAULIS_HOSTDEVICE explicit operator int() const;
  HYDRAULIS_HOSTDEVICE explicit operator int64_t() const;
  HYDRAULIS_HOSTDEVICE explicit operator size_t() const;
  HYDRAULIS_HOSTDEVICE bfloat16(int value);
  HYDRAULIS_HOSTDEVICE bfloat16(float16 value);
  HYDRAULIS_HOSTDEVICE bfloat16(int64_t value);
  HYDRAULIS_HOSTDEVICE bfloat16(size_t value);
  #if defined(__CUDACC__) 
  HYDRAULIS_HOSTDEVICE bfloat16(const __nv_bfloat16& value);
  HYDRAULIS_HOSTDEVICE operator __nv_bfloat16() const;
  HYDRAULIS_HOSTDEVICE __nv_bfloat16 to_bf16() const;
  #endif
  HYDRAULIS_HOSTDEVICE bfloat16 &operator=(const __nv_bfloat16& value) { val = *reinterpret_cast<const unsigned short*>(&value); return *this; }
  HYDRAULIS_HOSTDEVICE bfloat16 &operator=(const float f) { val = bfloat16(f).val; return *this; }
  HYDRAULIS_HOSTDEVICE bfloat16 &operator=(const float16 f) { val = bfloat16(float(f)).val; return *this; }
  HYDRAULIS_HOSTDEVICE bfloat16 &operator=(const bfloat16 h) { val = h.val; return *this; }
};

HYDRAULIS_HOSTDEVICE bfloat16::bfloat16(float value) {
  val = hydraulis::fp32_to_bf16(value);
}

HYDRAULIS_HOSTDEVICE bfloat16::bfloat16(float16 value) {
  val = hydraulis::fp32_to_bf16(static_cast<float>(value));
}

HYDRAULIS_HOSTDEVICE bfloat16::operator float() const {
  return hydraulis::fp32_from_bits16(val);
}

HYDRAULIS_HOSTDEVICE bfloat16::bfloat16(double value) {
  val = hydraulis::fp32_to_bf16(float(value));
}

HYDRAULIS_HOSTDEVICE bfloat16::operator float16() const {
  return static_cast<float16>(hydraulis::fp32_from_bits16(val));
}

HYDRAULIS_HOSTDEVICE bfloat16::operator double() const {
  return static_cast<double>(hydraulis::fp32_from_bits16(val));
}

HYDRAULIS_HOSTDEVICE bfloat16::operator int() const {
  return static_cast<int>(hydraulis::fp32_from_bits16(val));
}

HYDRAULIS_HOSTDEVICE bfloat16::operator int64_t() const {
  return static_cast<int64_t>(hydraulis::fp32_from_bits16(val));
}

HYDRAULIS_HOSTDEVICE bfloat16::operator size_t() const {
  return static_cast<size_t>(hydraulis::fp32_from_bits16(val));
}

HYDRAULIS_HOSTDEVICE bfloat16::bfloat16(int value) {
  val = hydraulis::fp32_to_bf16(float(value));
}

HYDRAULIS_HOSTDEVICE bfloat16::bfloat16(int64_t value) {
  val = hydraulis::fp32_to_bf16(float(value));
}

HYDRAULIS_HOSTDEVICE bfloat16::bfloat16(size_t value) {
  val = hydraulis::fp32_to_bf16(float(value));
}

#if defined(__CUDACC__) 
HYDRAULIS_HOSTDEVICE bfloat16::bfloat16(const __nv_bfloat16& value) {
  val = *reinterpret_cast<const unsigned short*>(&value);
}
HYDRAULIS_HOSTDEVICE bfloat16::operator __nv_bfloat16() const {
  return *reinterpret_cast<const __nv_bfloat16*>(&val);
}
HYDRAULIS_HOSTDEVICE __nv_bfloat16 bfloat16::to_bf16() const {
  return *reinterpret_cast<const __nv_bfloat16*>(&val);
}
#endif

/// Arithmetic
HYDRAULIS_DEVICE bfloat16 operator+(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) + static_cast<float>(b);
}

HYDRAULIS_DEVICE bfloat16 operator-(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) - static_cast<float>(b);
}

HYDRAULIS_DEVICE bfloat16 operator*(const bfloat16& a, const bfloat16& b) {
  return static_cast<float>(a) * static_cast<float>(b);
}

HYDRAULIS_DEVICE bfloat16 operator/(const bfloat16& a, const bfloat16& b) {
  // HT_ASSERT(static_cast<float>(b) != 0)
  // << "Divided by zero.";
  return static_cast<float>(a) / static_cast<float>(b);
}

HYDRAULIS_DEVICE bfloat16 operator-(const bfloat16& a) {
#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800)
  return __hneg(a);
#else
  return -static_cast<float>(a);
#endif
}

HYDRAULIS_DEVICE bfloat16& operator+=(bfloat16& a, const bfloat16& b) {
  a = a + b;
  return a;
}

HYDRAULIS_DEVICE bfloat16& operator-=(bfloat16& a, const bfloat16& b) {
  a = a - b;
  return a;
}

HYDRAULIS_DEVICE bfloat16& operator*=(bfloat16& a, const bfloat16& b) {
  a = a * b;
  return a;
}

HYDRAULIS_DEVICE bfloat16& operator/=(bfloat16& a, const bfloat16& b) {
  a = a / b;
  return a;
}

/// Arithmetic with floats

HYDRAULIS_DEVICE float operator+(bfloat16 a, float b) {
  return static_cast<float>(a) + b;
}
HYDRAULIS_DEVICE float operator-(bfloat16 a, float b) {
  return static_cast<float>(a) - b;
}
HYDRAULIS_DEVICE float operator*(bfloat16 a, float b) {
  return static_cast<float>(a) * b;
}
HYDRAULIS_DEVICE float operator/(bfloat16 a, float b) {
  return static_cast<float>(a) / b;
}

HYDRAULIS_DEVICE float operator+(float a, bfloat16 b) {
  return a + static_cast<float>(b);
}
HYDRAULIS_DEVICE float operator-(float a, bfloat16 b) {
  return a - static_cast<float>(b);
}
HYDRAULIS_DEVICE float operator*(float a, bfloat16 b) {
  return a * static_cast<float>(b);
}
HYDRAULIS_DEVICE float operator/(float a, bfloat16 b) {
  return a / static_cast<float>(b);
}

HYDRAULIS_DEVICE float& operator+=(float& a, const bfloat16& b) {
  return a += static_cast<float>(b);
}
HYDRAULIS_DEVICE float& operator-=(float& a, const bfloat16& b) {
  return a -= static_cast<float>(b);
}
HYDRAULIS_DEVICE float& operator*=(float& a, const bfloat16& b) {
  return a *= static_cast<float>(b);
}
HYDRAULIS_DEVICE float& operator/=(float& a, const bfloat16& b) {
  return a /= static_cast<float>(b);
}

/// Arithmetic with doubles

HYDRAULIS_DEVICE double operator+(bfloat16 a, double b) {
  return static_cast<double>(a) + b;
}
HYDRAULIS_DEVICE double operator-(bfloat16 a, double b) {
  return static_cast<double>(a) - b;
}
HYDRAULIS_DEVICE double operator*(bfloat16 a, double b) {
  return static_cast<double>(a) * b;
}
HYDRAULIS_DEVICE double operator/(bfloat16 a, double b) {
  return static_cast<double>(a) / b;
}

HYDRAULIS_DEVICE double operator+(double a, bfloat16 b) {
  return a + static_cast<double>(b);
}
HYDRAULIS_DEVICE double operator-(double a, bfloat16 b) {
  return a - static_cast<double>(b);
}
HYDRAULIS_DEVICE double operator*(double a, bfloat16 b) {
  return a * static_cast<double>(b);
}
HYDRAULIS_DEVICE double operator/(double a, bfloat16 b) {
  return a / static_cast<double>(b);
}

/// Arithmetic with ints

HYDRAULIS_DEVICE bfloat16 operator+(bfloat16 a, int b) {
  return a + static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator-(bfloat16 a, int b) {
  return a - static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator*(bfloat16 a, int b) {
  return a * static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator/(bfloat16 a, int b) {
  return a / static_cast<bfloat16>(b);
}

HYDRAULIS_DEVICE bfloat16 operator+(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
HYDRAULIS_DEVICE bfloat16 operator-(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
HYDRAULIS_DEVICE bfloat16 operator*(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
HYDRAULIS_DEVICE bfloat16 operator/(int a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

//// Arithmetic with int64_t

HYDRAULIS_DEVICE bfloat16 operator+(bfloat16 a, int64_t b) {
  return a + static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator-(bfloat16 a, int64_t b) {
  return a - static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator*(bfloat16 a, int64_t b) {
  return a * static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator/(bfloat16 a, int64_t b) {
  return a / static_cast<bfloat16>(b);
}

HYDRAULIS_DEVICE bfloat16 operator+(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
HYDRAULIS_DEVICE bfloat16 operator-(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
HYDRAULIS_DEVICE bfloat16 operator*(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
HYDRAULIS_DEVICE bfloat16 operator/(int64_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

//// Arithmetic with size_t

HYDRAULIS_DEVICE bfloat16 operator+(bfloat16 a, size_t b) {
  return a + static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator-(bfloat16 a, size_t b) {
  return a - static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator*(bfloat16 a, size_t b) {
  return a * static_cast<bfloat16>(b);
}
HYDRAULIS_DEVICE bfloat16 operator/(bfloat16 a, size_t b) {
  return a / static_cast<bfloat16>(b);
}

HYDRAULIS_DEVICE bfloat16 operator+(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) + b;
}
HYDRAULIS_DEVICE bfloat16 operator-(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) - b;
}
HYDRAULIS_DEVICE bfloat16 operator*(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) * b;
}
HYDRAULIS_DEVICE bfloat16 operator/(size_t a, bfloat16 b) {
  return static_cast<bfloat16>(a) / b;
}

std::ostream& operator<<(std::ostream& out, const bfloat16& value);
} //namespace hydraulis

namespace std {

template <>
class numeric_limits<hydraulis::bfloat16> {
 public:
  static constexpr bool is_signed = true;
  static constexpr bool is_specialized = true;
  static constexpr bool is_integer = false;
  static constexpr bool is_exact = false;
  static constexpr bool has_infinity = true;
  static constexpr bool has_quiet_NaN = true;
  static constexpr bool has_signaling_NaN = true;
  static constexpr auto has_denorm = numeric_limits<float>::has_denorm;
  static constexpr auto has_denorm_loss =
      numeric_limits<float>::has_denorm_loss;
  static constexpr auto round_style = numeric_limits<float>::round_style;
  static constexpr bool is_iec559 = false;
  static constexpr bool is_bounded = true;
  static constexpr bool is_modulo = false;
  static constexpr int digits = 8;
  static constexpr int digits10 = 2;
  static constexpr int max_digits10 = 4;
  static constexpr int radix = 2;
  static constexpr int min_exponent = -125;
  static constexpr int min_exponent10 = -37;
  static constexpr int max_exponent = 128;
  static constexpr int max_exponent10 = 38;
  static constexpr auto traps = numeric_limits<float>::traps;
  static constexpr auto tinyness_before =
      numeric_limits<float>::tinyness_before;
  static constexpr hydraulis::bfloat16 min() {
    return hydraulis::bfloat16(0x0080, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 lowest() {
    return hydraulis::bfloat16(0xFF7F, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 max() {
    return hydraulis::bfloat16(0x7F7F, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 epsilon() {
    return hydraulis::bfloat16(0x3C00, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 round_error() {
    return hydraulis::bfloat16(0x3F00, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 infinity() {
    return hydraulis::bfloat16(0x7F80, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 quiet_NaN() {
    return hydraulis::bfloat16(0x7FC0, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 signaling_NaN() {
    return hydraulis::bfloat16(0x7F80, hydraulis::bfloat16::from_bits());
  }
  static constexpr hydraulis::bfloat16 denorm_min() {
    return hydraulis::bfloat16(0x0001, hydraulis::bfloat16::from_bits());
  }
};

} //namespace std