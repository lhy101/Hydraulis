#include "hydraulis/_binding/distributed/comm_group.h"
#include "hydraulis/_binding/utils/except.h"
#include "hydraulis/_binding/utils/arg_parser.h"
#include "hydraulis/_binding/utils/decl_utils.h"
#include "hydraulis/_binding/utils/function_registry.h"

namespace hydraulis {

PyTypeObject PyCommGroup_Type_obj = {
  PyVarObject_HEAD_INIT(nullptr, 0) 
  "hydraulis.CommGroup", /* tp_name */
  sizeof(PyCommGroup), /* tp_basicsize */
  0, /* tp_itemsize */
  nullptr, /* tp_dealloc */
  0, /* tp_vectorcall_offset */
  nullptr, /* tp_getattr */
  nullptr, /* tp_setattr */
  nullptr, /* tp_reserved */
  nullptr, /* tp_repr */
  nullptr, /* tp_as_number */
  nullptr, /* tp_as_sequence */
  nullptr, /* tp_as_mapping */
  nullptr, /* tp_hash  */
  nullptr, /* tp_call */
  nullptr, /* tp_str */
  nullptr, /* tp_getattro */
  nullptr, /* tp_setattro */
  nullptr, /* tp_as_buffer */
  Py_TPFLAGS_DEFAULT, /* tp_flags */
  nullptr, /* tp_doc */
  nullptr, /* tp_traverse */
  nullptr, /* tp_clear */
  nullptr, /* tp_richcompare */
  0, /* tp_weaklistoffset */
  nullptr, /* tp_iter */
  nullptr, /* tp_iternext */
  nullptr, /* tp_methods */
  nullptr, /* tp_members */
  nullptr, /* tp_getset */
  nullptr, /* tp_base */
  nullptr, /* tp_dict */
  nullptr, /* tp_descr_get */
  nullptr, /* tp_descr_set */
  0, /* tp_dictoffset */
  nullptr, /* tp_init */
  nullptr, /* tp_alloc */
  nullptr, /* tp_new */
};
PyTypeObject* PyCommGroup_Type = &PyCommGroup_Type_obj;

// TODO: update init params
PyObject* CommGroup_Init(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  static PyArgParser parser({"init_comm_group(int device_num=8, List[int] device_idxs=[], std::string server_address=\"127.0.0.1:23457\")"});
  auto parsed_args = parser.parse(args, kwargs);
  int device_num = parsed_args.get_int64_or_default(0);
  std::vector<int64_t> device_idxs = parsed_args.get_int64_list_or_default(1);
  std::string server_address = parsed_args.get_string_or_default(2);
  return PyDevice_New(hydraulis::impl::comm::SetUpDeviceMappingAndAssignLocalDeviceOnce({{kCUDA, device_num}}, device_idxs, server_address));
  HT_PY_FUNC_END
}

PyObject* CommGroup_GetLocalDevice(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  return PyDevice_New(hydraulis::impl::comm::GetLocalDevice());
  HT_PY_FUNC_END
}

PyObject* CommGroup_GetGlobalDeviceGroup(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  return PyDeviceGroup_New(hydraulis::impl::comm::GetGlobalDeviceGroup());
  HT_PY_FUNC_END
}

// workaround
PyObject* CommGroup_GlobalCommBarrier(PyObject*, PyObject* args, PyObject* kwargs) {
  HT_PY_FUNC_BEGIN
  std::vector<int> all_ranks(hydraulis::impl::comm::GetWorldSize());
  std::iota(all_ranks.begin(), all_ranks.end(), 0);
  hydraulis::impl::comm::GetLocalClient()->Barrier(0, all_ranks);
  Py_RETURN_NONE;
  HT_PY_FUNC_END
}

std::vector<PyMethodDef> InitCommGroupPyClassMethodDefs() {
  std::vector<PyMethodDef> ret = {{nullptr}};
  AddPyMethodDefs(ret, {
    {"init_comm_group", (PyCFunction) CommGroup_Init, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"local_device", (PyCFunction) CommGroup_GetLocalDevice, METH_VARARGS | METH_KEYWORDS, nullptr }, 
    {"global_device_group", (PyCFunction) CommGroup_GetGlobalDeviceGroup, METH_VARARGS | METH_KEYWORDS, nullptr },    
    {"global_comm_barrier", (PyCFunction) CommGroup_GlobalCommBarrier, METH_VARARGS | METH_KEYWORDS, nullptr },     
    {nullptr}
  });
  
  AddPyMethodDefs(ret, hydraulis::graph::get_registered_tensor_class_methods());
  return ret;
}

void AddPyCommGroupTypeToModule(py::module_& module) {
  static auto comm_group_class_methods = InitCommGroupPyClassMethodDefs();
  HT_RUNTIME_ERROR_IF(0 != PyModule_AddFunctions(
      module.ptr(), comm_group_class_methods.data()))
    << "Failed to add CommGroup class methods";  
}

} // namespace hydraulis
  