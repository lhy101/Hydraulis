#include "hydraulis/_binding/module.h"
#include "hydraulis/_binding/constants.h"
#include "hydraulis/_binding/utils/pybind_common.h"
#include "hydraulis/_binding/core/device.h"
#include "hydraulis/_binding/core/dtype.h"
#include "hydraulis/_binding/core/stream.h"
#include "hydraulis/_binding/core/ndarray.h"
#include "hydraulis/_binding/core/symbol.h"
#include "hydraulis/_binding/graph/operator.h"
#include "hydraulis/_binding/graph/tensor.h"
#include "hydraulis/_binding/graph/distributed_states.h"
#include "hydraulis/_binding/graph/graph.h"
#include "hydraulis/_binding/graph/autocast.h"
#include "hydraulis/_binding/graph/recompute.h"
#include "hydraulis/_binding/graph/cpu_offload.h"
#include "hydraulis/_binding/graph/gradscaler.h"
#include "hydraulis/_binding/graph/sgdoptimizer.h"
#include "hydraulis/_binding/graph/subgraph.h"
#include "hydraulis/_binding/graph/adamoptimizer.h"
#include "hydraulis/_binding/graph/dataloader.h"
#include "hydraulis/_binding/graph/init/initializer.h"
#include "hydraulis/_binding/distributed/comm_group.h"
#include "hydraulis/_binding/graph/profiler.h"

PYBIND11_MODULE(HT_CORE_PY_MODULE, m) {
  hydraulis::AddPyDeviceTypeToModule(m);
  hydraulis::AddPyDeviceGroupTypeToModule(m);
  hydraulis::AddPyDataTypeTypeToModule(m);
  hydraulis::AddPyStreamTypeToModule(m);
  hydraulis::AddPyNDArrayTypeToModule(m);
  hydraulis::AddPyIntSymbolTypeToModule(m);
  hydraulis::AddPyCommGroupTypeToModule(m);
  hydraulis::graph::AddPyOperatorTypeToModule(m);
  hydraulis::graph::AddPyTensorTypeToModule(m);
  hydraulis::graph::AddPyDistributedStatesTypeToModule(m);
  hydraulis::graph::AddPyDistributedStatesUnionTypeToModule(m);
  hydraulis::graph::AddPyGraphTypeToModule(m);
  hydraulis::graph::AddPyAutoCastTypeToModule(m);
  hydraulis::graph::AddPyGradScalerTypeToModule(m);
  hydraulis::graph::AddPySGDOptimizerTypeToModule(m);
  hydraulis::graph::AddPySubGraphTypeToModule(m);
  hydraulis::graph::AddPyAdamOptimizerTypeToModule(m);
  hydraulis::graph::AddPyDataloaderTypeToModule(m);
  hydraulis::graph::AddPyInitializerTypeToModule(m);
  auto internal_sub_module = m.def_submodule("_internal_context");
  hydraulis::graph::AddOpContextManagingFunctionsToModule(internal_sub_module);
  hydraulis::graph::AddGraphContextManagingFunctionsToModule(internal_sub_module);
  hydraulis::graph::AddAutoCastContextManagingFunctionsToModule(internal_sub_module);
  hydraulis::graph::AddSubGraphContextManagingFunctionsToModule(internal_sub_module);
  hydraulis::graph::AddRecomputeContextManagingFunctionsToModule(internal_sub_module);
  hydraulis::graph::AddCPUOffloadContextManagingFunctionsToModule(internal_sub_module);
  hydraulis::impl::AddPyProfileTypeToModule(m);
  hydraulis::impl::AddProfileContextManagingFunctionsToModule(internal_sub_module);
}
