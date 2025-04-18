#pragma once

#include <Python.h>
#include "hydraulis/graph/operator.h"
#include "hydraulis/_binding/core/device.h"
#include "hydraulis/_binding/core/stream.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {
namespace graph {

struct PyOperator {
  PyObject_HEAD;
  Operator op;
};

extern PyTypeObject* PyOperator_Type;

inline bool PyOperator_Check(PyObject* obj) {
  return PyOperator_Type && PyObject_TypeCheck(obj, PyOperator_Type);
}

inline bool PyOperator_CheckExact(PyObject* obj) {
  return PyOperator_Type && obj->ob_type == PyOperator_Type;
}

PyObject* PyOperator_New(const Operator& op,
                         bool return_none_if_undefined = true);

PyObject* PyOperatorList_New(const OpList& ops,
                             bool return_none_if_undefined = true);

void AddPyOperatorTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyOperator(PyObject* obj) {
  return PyOperator_Check(obj);
}

inline Operator Operator_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyOperator*>(obj)->op;
}

inline bool CheckPyOperatorList(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  if (is_tuple || PyList_Check(obj)) {
    size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
    if (size > 0) {
      // only check for the first item for efficiency
      auto* item = is_tuple ? PyTuple_GET_ITEM(obj, 0) \
                            : PyList_GET_ITEM(obj, 0);
      if (!CheckPyOperator(item))
        return false;
    }
    return true;
  }
  return false;
}

inline OpList OperatorList_FromPyObject(PyObject* obj) {
  bool is_tuple = PyTuple_Check(obj);
  size_t size = is_tuple ? PyTuple_GET_SIZE(obj) : PyList_GET_SIZE(obj);
  OpList ret(size);
  for (size_t i = 0; i < size; i++) {
    auto* item = is_tuple ? PyTuple_GET_ITEM(obj, i) : PyList_GET_ITEM(obj, i);
    ret[i] = Operator_FromPyObject(item);
  }
  return ret;
}

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddOpContextManagingFunctionsToModule(py::module_&);

ContextManager<TensorList>& get_extra_deps_ctx();

inline OpMeta CurrentOpMetaCtx() {
  OpMeta ret;
  auto stream_index_opt = get_stream_index_ctx().peek();
  if (stream_index_opt != nullopt)
    ret.set_stream_index(*stream_index_opt);
  auto eager_device_opt = get_eager_device_ctx().peek();
  if (eager_device_opt != nullopt)
    ret.set_eager_device(*eager_device_opt);
  auto device_group_hierarchy_opt = get_dg_hierarchy_ctx().peek();
  if (device_group_hierarchy_opt != nullopt)
    ret.set_device_group_hierarchy(*device_group_hierarchy_opt);
  // TODO: name & extra_deps
  return ret;
}

} // namespace graph
} // namespace hydraulis
