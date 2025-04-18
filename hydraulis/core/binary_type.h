#pragma once

#include "hydraulis/common/macros.h"

namespace hydraulis {

enum class BinaryType : int8_t {
  ADD = 0,
  SUB,
  MUL,
  DIV,
  MOD,
  NUM_BINARY_OPS
};

constexpr BinaryType ADD = BinaryType::ADD;
constexpr BinaryType SUB = BinaryType::SUB;
constexpr BinaryType MUL = BinaryType::MUL;
constexpr BinaryType DIV = BinaryType::DIV;
constexpr BinaryType MOD = BinaryType::MOD;
constexpr int16_t NUM_BINARY_OPS =
  static_cast<int16_t>(BinaryType::NUM_BINARY_OPS);

std::string BinaryType2Str(const BinaryType&);
std::ostream& operator<<(std::ostream&, const BinaryType&);

} // namespace hydraulis

namespace std {
template <>
struct hash<hydraulis::BinaryType> {
  std::size_t operator()(hydraulis::BinaryType binary_type) const noexcept {
    return std::hash<int>()(static_cast<int>(binary_type));
  }
};
} // namespace std
