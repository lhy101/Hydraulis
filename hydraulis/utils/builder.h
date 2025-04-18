#pragma once

namespace hydraulis {

template <typename T>
class Builder {
 public:
  static T builder() {
    return {};
  }
  T& build() {
    return static_cast<T&>(*this);
  }
};

} // namespace hydraulis
