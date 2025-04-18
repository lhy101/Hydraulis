#pragma once

#include <Python.h>
#include "hydraulis/impl/communication/comm_group.h"
#include "hydraulis/impl/communication/nccl_comm_group.h"
#include "hydraulis/core/stream.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {

struct PyCommGroup {
  PyObject_HEAD;
};

extern PyTypeObject* PyCommGroup_Type;

void AddPyCommGroupTypeToModule(py::module_& module);

} // namespace hydraulis
