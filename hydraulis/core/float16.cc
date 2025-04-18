#include "hydraulis/core/float16.h"
#include <iostream>
namespace hydraulis {
std::ostream& operator<<(std::ostream& out, const float16& value) {
  out << (float)value;
  return out;
}
} //namespace hydraulis
