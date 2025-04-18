#pragma once

#include <Python.h>
#include "hydraulis/graph/offload/activation_cpu_offload.h"
#include "hydraulis/_binding/core/ndarray.h"
#include "hydraulis/_binding/graph/tensor.h"
#include "hydraulis/_binding/utils/numpy.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {
namespace graph {

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddCPUOffloadContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hydraulis
