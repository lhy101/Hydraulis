#pragma once

#include <Python.h>
#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/optim/optimizer.h"
#include "hydraulis/graph/tensor.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {
namespace graph {

struct PyAdamOptimizer {
  PyObject_HEAD;
  AdamOptimizer optimizer;
};

extern PyTypeObject* PyAdamOptimizer_Type;

inline bool PyAdamOptimizer_Check(PyObject* obj) {
  return PyAdamOptimizer_Type && PyObject_TypeCheck(obj, PyAdamOptimizer_Type);
}

inline bool PyAdamOptimizer_CheckExact(PyObject* obj) {
  return PyAdamOptimizer_Type && obj->ob_type == PyAdamOptimizer_Type;
}

PyObject* PyAdamOptimizer_New(AdamOptimizer&& tensor, bool return_none_if_undefined = true);

void AddPyAdamOptimizerTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyAdamOptimizer(PyObject* obj) {
  return PyAdamOptimizer_Check(obj);
}

inline AdamOptimizer AdamOptimizer_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyAdamOptimizer*>(obj)->optimizer;
}

} // namespace graph
} // namespace hydraulis
