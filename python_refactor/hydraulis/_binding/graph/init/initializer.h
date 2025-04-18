#pragma once

#include <Python.h>
#include "hydraulis/graph/init/initializer.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {
namespace graph {

struct PyInitializer {
  PyObject_HEAD;
  std::shared_ptr<Initializer> init;
};

extern PyTypeObject* PyInitializer_Type;

inline bool PyInitializer_Check(PyObject* obj) {
  return PyInitializer_Type && PyObject_TypeCheck(obj, PyInitializer_Type);
}

inline bool PyInitializer_CheckExact(PyObject* obj) {
  return PyInitializer_Type && obj->ob_type == PyInitializer_Type;
}

PyObject* PyInitializer_New(const Initializer& ds);

void AddPyInitializerTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyInitializer(PyObject* obj) {
  return PyInitializer_Check(obj);
}

inline std::shared_ptr<Initializer> Initializer_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyInitializer*>(obj)->init;
}

} // namespace graph
} // namespace hydraulis
