#pragma once

#include "hydraulis/common/except.h"
#include <Python.h>

namespace hydraulis {

#define HT_PY_FUNC_BEGIN try {
#define HT_PY_FUNC_RETSMT(retsmt)                                             \
  } catch (const hydraulis::assertion_error& e) {                                  \
    PyErr_SetString(PyExc_AssertionError, e.what());                          \
    retsmt;                                                                   \
  } catch (const hydraulis::bad_alloc& e) {                                        \
    PyErr_SetString(PyExc_MemoryError, e.what());                             \
    retsmt;                                                                   \
  } catch (const hydraulis::not_implemented& e) {                                  \
    PyErr_SetString(PyExc_NotImplementedError, e.what());                     \
    retsmt;                                                                   \
  } catch (const hydraulis::type_error& e) {                                       \
    PyErr_SetString(PyExc_TypeError, e.what());                               \
    retsmt;                                                                   \
  } catch (const hydraulis::value_error& e) {                                      \
    PyErr_SetString(PyExc_ValueError, e.what());                              \
    retsmt;                                                                   \
  } catch (const hydraulis::timeout_error& e) {                                    \
    PyErr_SetString(PyExc_TimeoutError, e.what());                            \
    retsmt;                                                                   \
  } catch (const hydraulis::runtime_error& e) {                                    \
    PyErr_SetString(PyExc_RuntimeError, e.what());                            \
    retsmt;                                                                   \
  } catch (const std::exception& e) {                                         \
    PyErr_SetString(PyExc_RuntimeError, e.what());                            \
    retsmt;                                                                   \
  }
#define HT_PY_FUNC_RETURN(retval) HT_PY_FUNC_RETSMT(return retval)
#define HT_PY_FUNC_END HT_PY_FUNC_RETURN(nullptr)

#define HT_PY_PARSER_INCORRECT_SIGNATURE(parsed_args)                         \
  HT_VALUE_ERROR << "Error occurred in parser: incorrect signature index "    \
    << parsed_args.signature_index() << " for "                               \
    << parsed_args.signature_name()

} // namespace hydraulis
