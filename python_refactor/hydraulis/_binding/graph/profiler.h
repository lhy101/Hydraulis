#pragma once

#include <Python.h>
#include "hydraulis/_binding/core/ndarray.h"
#include "hydraulis/_binding/graph/tensor.h"
#include "hydraulis/_binding/utils/numpy.h"
#include "hydraulis/_binding/utils/pybind_common.h"
#include "hydraulis/impl/profiler/profiler.h"

namespace hydraulis {
namespace impl {

struct PyProfile {
  PyObject_HEAD;
  ProfileId profile_id;
};

extern PyTypeObject* PyProfile_Type;

PyObject* PyProfile_New(ProfileId profile_id);

void AddPyProfileTypeToModule(py::module_& module);
void AddProfileContextManagingFunctionsToModule(py::module_& m);

} // namespace impl
} // namespace hydraulis
