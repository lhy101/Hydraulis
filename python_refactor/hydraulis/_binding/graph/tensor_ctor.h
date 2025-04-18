#pragma once

#include <Python.h>
#include "hydraulis/_binding/graph/tensor.h"
#include "hydraulis/graph/init/initializer.h"

namespace hydraulis {
namespace graph {

PyObject* TensorCopyCtor(PyTypeObject* type, PyObject* args, PyObject* kwargs);

} // namespace graph
} // namespace hydraulis
