#pragma once

#include <sstream>
#include <cstring>
#include <execinfo.h>
#include "hydraulis/common/logging.h"

namespace hydraulis {

class hydraulis_exception : public std::exception {
 public:
  explicit hydraulis_exception(const char* msg, const char* file, int line) {
    std::ostringstream ss;
    ss << " (" << file << ":" << line << ") " << msg;
    _msg = ss.str();
  }
  explicit hydraulis_exception(const std::string& msg, const char* file, int line)
  : hydraulis_exception(msg.c_str(), file, line) {}
  explicit hydraulis_exception(const char* msg) : _msg(msg) {}
  explicit hydraulis_exception(const std::string& msg) : _msg(msg) {}
  virtual ~hydraulis_exception() = default;
  virtual const char* what() const noexcept override {
    return _msg.c_str();
  }

 protected:
  std::string _msg;
};

#define DECLARE_HT_EXCEPTION(Exception)                                        \
  class Exception : public hydraulis_exception {                                    \
   public:                                                                     \
    using hydraulis_exception::hydraulis_exception;                                      \
    ~Exception() = default;                                                    \
  }

DECLARE_HT_EXCEPTION(assertion_error);
DECLARE_HT_EXCEPTION(bad_alloc);
DECLARE_HT_EXCEPTION(not_implemented);
DECLARE_HT_EXCEPTION(value_error);
DECLARE_HT_EXCEPTION(timeout_error);
DECLARE_HT_EXCEPTION(type_error);
DECLARE_HT_EXCEPTION(runtime_error);

} // namespace hydraulis

/******************************************************
 * Utils for Exception Throwing
 ******************************************************/

#define HT_ASSERT(cond)                                                        \
  if (!(cond))                                                                 \
  __HT_FATAL_SILENT(hydraulis::assertion_error)                                     \
    << "Assertion " << #cond << " failed: "
#define HT_ASSERT_LT(x, y) HT_ASSERT((x) < (y))
#define HT_ASSERT_GT(x, y) HT_ASSERT((x) > (y))
#define HT_ASSERT_LE(x, y) HT_ASSERT((x) <= (y))
#define HT_ASSERT_GE(x, y) HT_ASSERT((x) >= (y))
#define HT_ASSERT_EQ(x, y) HT_ASSERT((x) == (y))
#define HT_ASSERT_NE(x, y) HT_ASSERT((x) != (y))
#define HT_ASSERT_FUZZY_EQ(x, y, atol, rtol)                                   \
  HT_ASSERT(std::abs((x) - (y)) <= (atol) + (rtol) *std::abs(y))
#define ASSERT_FUZZY_NE(x, y, atol, rtol)                                      \
  HT_ASSERT(std::abs((x) - (y)) > (atol) + (rtol) *std::abs(y))

#define HT_BAD_ALLOC __HT_FATAL_SILENT(hydraulis::bad_alloc) << "Bad alloc: "
#define HT_BAD_ALLOC_IF(cond)                                                  \
  if (cond)                                                                    \
  HT_BAD_ALLOC
#define HT_NOT_IMPLEMENTED                                                     \
  __HT_FATAL_SILENT(hydraulis::not_implemented) << "Not implemented: "
#define HT_NOT_IMPLEMENTED_IF(cond)                                            \
  if (cond)                                                                    \
  HT_NOT_IMPLEMENTED
#define HT_VALUE_ERROR __HT_FATAL_SILENT(hydraulis::value_error) << "Value error: "
#define HT_VALUE_ERROR_IF(cond)                                                \
  if (cond)                                                                    \
  HT_VALUE_ERROR
#define HT_TYPE_ERROR __HT_FATAL_SILENT(hydraulis::type_error) << "Type error: "
#define HT_TYPE_ERROR_IF(cond)                                                 \
  if (cond)                                                                    \
  HT_TYPE_ERROR
#define HT_TIMEOUT_ERROR                                                       \
  __HT_FATAL_SILENT(hydraulis::timeout_error) \<< "Timeout error: "
#define HT_TIMEOUT_ERROR_IF(cond)                                              \
  if (cond)                                                                    \
  HT_TIMEOUT_ERROR
#define HT_RUNTIME_ERROR                                                       \
  __HT_FATAL_SILENT(hydraulis::runtime_error) << "Runtime error: "
#define HT_RUNTIME_ERROR_IF(cond)                                              \
  if (cond)                                                                    \
  HT_RUNTIME_ERROR
