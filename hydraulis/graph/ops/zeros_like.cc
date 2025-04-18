#include "hydraulis/graph/ops/zeros_like.h"
#include "hydraulis/graph/headers.h"

namespace hydraulis {
namespace graph {

Tensor MakeZerosLikeOp(Tensor input, OpMeta op_meta) {
  return Graph::MakeOp(std::make_shared<ZerosLikeOpImpl>(), {std::move(input)},
                       std::move(op_meta))
    ->output(0);
}

} // namespace graph
} // namespace hydraulis
