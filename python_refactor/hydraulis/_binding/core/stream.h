#pragma once

#include <Python.h>
#include "hydraulis/core/stream.h"
#include "hydraulis/_binding/utils/pybind_common.h"
#include "hydraulis/_binding/utils/context_manager.h"

namespace hydraulis {

struct PyStream {
  PyObject_HEAD;
  Stream stream;
};

extern PyTypeObject* PyStream_Type;

inline bool PyStream_Check(PyObject* obj) {
  return PyStream_Type && PyObject_TypeCheck(obj, PyStream_Type);
}

inline bool PyStream_CheckExact(PyObject* obj) {
  return PyStream_Type && obj->ob_type == PyStream_Type;
}

PyObject* PyStream_New(const Stream& stream);

void AddPyStreamTypeToModule(py::module_& module);

/******************************************************
 * ArgParser Utils
 ******************************************************/

inline bool CheckPyStream(PyObject* obj) {
  return PyStream_Check(obj);
}

inline Stream Stream_FromPyObject(PyObject* obj) {
  return reinterpret_cast<PyStream*>(obj)->stream;
}

/******************************************************
 * For contextlib usage
 ******************************************************/

ContextManager<StreamIndex>& get_stream_index_ctx();
ContextManager<Stream>& get_stream_ctx();

} // namespace hydraulis
