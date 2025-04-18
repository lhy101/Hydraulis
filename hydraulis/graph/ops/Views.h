#pragma once

#include "hydraulis/graph/operator.h"

namespace hydraulis {
namespace graph {

class ViewsOpImpl;

class ViewsOpImpl : public OpInterface {
 protected:
  ViewsOpImpl(OpType&& op_type)
  : OpInterface(std::move(op_type)) {}
 
 public:
  inline bool require_contig_inputs() const override {
    return false;
  }
};

} // namespace graph
} // namespace hydraulis