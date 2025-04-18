#include "hydraulis/core/bfloat16.h"
#include <iostream>
namespace hydraulis {
std::ostream& operator<<(std::ostream& out, const bfloat16& value) {
  out << (float)value;
  return out;
}
} //namespace hydraulis
