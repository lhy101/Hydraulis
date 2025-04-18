#pragma once

#include <Python.h>
#include "hydraulis/graph/operator.h"
#include "hydraulis/graph/optim/optimizer.h"
#include "hydraulis/graph/tensor.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {
namespace graph {

struct PySGDOptimizer {
  PyObject_HEAD;
  SGDOptimizer optimizer;
};

extern PyTypeObject* PySGDOptimizer_Type;

inline bool PySGDOptimizer_Check(PyObject* obj) {
  return PySGDOptimizer_Type && PyObject_TypeCheck(obj, PySGDOptimizer_Type);
}

inline bool PySGDOptimizer_CheckExact(PyObject* obj) {
  return PySGDOptimizer_Type && obj->ob_type == PySGDOptimizer_Type;
}

PyObject* PySGDOptimizer_New(SGDOptimizer&& tensor, bool return_none_if_undefined = true);

void AddPySGDOptimizerTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPySGDOptimizer(PyObject* obj) {
  return PySGDOptimizer_Check(obj);
}

inline SGDOptimizer SGDOptimizer_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PySGDOptimizer*>(obj)->optimizer;
}

} // namespace graph
} // namespace hydraulis
