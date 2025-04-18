#pragma once

#include <Python.h>
#include "hydraulis/graph/graph.h"
#include "hydraulis/_binding/core/ndarray.h"
#include "hydraulis/_binding/graph/tensor.h"
#include "hydraulis/_binding/utils/numpy.h"
#include "hydraulis/_binding/utils/python_primitives.h"
#include "hydraulis/_binding/utils/pybind_common.h"

namespace hydraulis {
namespace graph {

struct PyGraph {
  PyObject_HEAD;
  GraphId graph_id;
};

extern PyTypeObject* PyGraph_Type;

inline bool PyGraph_Check(PyObject* obj) {
  return PyGraph_Type && PyObject_TypeCheck(obj, PyGraph_Type);
}

inline bool PyGraph_CheckExact(PyObject* obj) {
  return PyGraph_Type && obj->ob_type == PyGraph_Type;
}

PyObject* PyGraph_New(GraphId graph_id);

void AddPyGraphTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyGraph(PyObject* obj) {
  return PyGraph_Check(obj);
}

inline GraphId GraphId_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyGraph*>(obj)->graph_id;
}

inline bool CheckPyFeedDict(PyObject* obj) {
  if (PyDict_Check(obj)) {
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
      if (!CheckPyTensor(key))
        return false;
      if (!CheckPyNDArray(value) && !CheckNumpyArray(value) &&
          !CheckPyNDArrayList(value) && !CheckNumpyArrayList(value))
        return false;
    }
    return true;
  }
  return false;
}

inline bool CheckPyParameterDict(PyObject* obj) {
  if (PyDict_Check(obj)) {
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
      if (!CheckPyString(key))
        return false;
      if (!CheckPyLong(value))
        return false;
    }
    return true;
  }
  return false;
}

inline bool CheckPyStateDict(PyObject* obj) {
  if (PyDict_Check(obj)) {
    PyObject* key;
    PyObject* value;
    Py_ssize_t pos = 0;
    while (PyDict_Next(obj, &pos, &key, &value)) {
      if (!CheckPyString(key))
        return false;
      if (!CheckPyTensor(value))
        return false;
    }
    return true;
  }
  return false;
}

inline FeedDict FeedDict_FromPyObject(PyObject* obj) {
  FeedDict feed_dict;
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int64_t cnt = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    HT_LOG_TRACE << hydraulis::impl::comm::GetLocalDevice() << ": processing element " << cnt++ << " in FeedDict...";
    TensorId k = Tensor_FromPyObject(key)->id();
    NDArrayList v;
    if (PyList_Check(value)) {
      v = CheckPyNDArrayList(value) ? NDArrayList_FromPyObject(value)
                                    : NDArrayListFromNumpyList(value, {}, Tensor_FromPyObject(key)->dtype());
    } else {
      v = {CheckPyNDArray(value) ? NDArray_FromPyObject(value)
                                 : NDArrayFromNumpy(value, {}, Tensor_FromPyObject(key)->dtype())};
    }
    feed_dict.insert({k, v});
  }
  return feed_dict;
}

inline ParameterDict ParameterDict_FromPyObject(PyObject* obj) {
  ParameterDict parameter_dict;
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int64_t cnt = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    HT_LOG_TRACE << hydraulis::impl::comm::GetLocalDevice() << ": processing element " << cnt++ << " in ParamDict...";
    std::string k = String_FromPyUnicode(key);
    int64_t v = Int64_FromPyLong(value);
    parameter_dict.insert({k, v});
  }
  return parameter_dict;
}

inline StateDict StateDict_FromPyObject(PyObject* obj) {
  StateDict StateDict;
  PyObject* key;
  PyObject* value;
  Py_ssize_t pos = 0;
  int64_t cnt = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    HT_LOG_TRACE << hydraulis::impl::comm::GetLocalDevice() << ": processing element " << cnt++ << " in ParamDict...";
    std::string k = String_FromPyUnicode(key);
    Tensor v = Tensor_FromPyObject(value);
    StateDict.insert({k, v});
  }
  return StateDict;
}

/******************************************************
 * For contextlib usage
 ******************************************************/

void AddGraphContextManagingFunctionsToModule(py::module_&);

} // namespace graph
} // namespace hydraulis
