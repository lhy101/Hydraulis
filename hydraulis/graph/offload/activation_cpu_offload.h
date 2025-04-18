#pragma once

#include "hydraulis/graph/common.h"
#include "hydraulis/graph/executable_graph.h"
#include "hydraulis/graph/graph.h"
#include <functional>

namespace hydraulis {
namespace graph {

class ActivationCPUOffload {
 public:
  static bool enabled() {
    return _enabled;
  }

  static void set_cpu_offload_enabled() {
    _enabled = true;
  }

  static void reset_cpu_offload_enabled() {
    _enabled = false;
  }

  static void OffloadToCPU(const OpRefList& topo_order);

 protected:

  static bool IsNoOffloadOp(Operator& op);

  static void OffloadTensorToCPU(const OpRefList& topo_order, const Tensor& tensor);

  static bool _enabled;
};

} // namespace graph
} // namespace hydraulis