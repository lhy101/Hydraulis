#pragma once

#include <Python.h>
#include "hydraulis/graph/recompute/recompute.h"
#include "hydraulis/_binding/core/ndarray.h"
#include "hydraulis/_binding/graph/tensor.h"
#include "hydraulis/_binding/utils/numpy.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {
namespace graph {

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddRecomputeContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hydraulis
