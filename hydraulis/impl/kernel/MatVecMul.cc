#include "hydraulis/core/ndarray.h"
#include "hydraulis/core/stream.h"
#include "hydraulis/impl/utils/common_utils.h"
#include "hydraulis/impl/utils/omp_utils.h"
#include "hydraulis/impl/stream/CPUStream.h"

namespace hydraulis {
namespace impl {

template <typename spec_t>
void matvecmul_cpu(const spec_t* a, const spec_t* x, bool trans,
                   size_t m, size_t n, spec_t* output) {
#ifdef _OPENMP
#pragma omp parallel
#endif
{
  size_t i, j;
  size_t row = trans ? n : m;
  size_t col = trans ? m : n;
  spec_t out[row] = {};
  for (i = 0; i < row; i++)
#ifdef _OPENMP
#pragma omp for
#endif
    for (j = 0; j < col; j++)
      out[i] += a[i * n + j] * x[j];
#ifdef _OPENMP
#pragma omp critical
#endif
  for (i = 0; i < row; i++)
    output[i] = out[i];
}
}

void MatVecMulCpu(const NDArray& a, bool trans, const NDArray& x,
               NDArray& output, const Stream& stream) {
  HT_ASSERT_CPU_DEVICE(a);
  HT_ASSERT_SAME_DEVICE(a, x);
  HT_ASSERT_SAME_DEVICE(a, output);
  HT_ASSERT_NDIM(a, 2);
  HT_ASSERT_NDIM(x, 1);
  HT_ASSERT_NDIM(output, 1);
  HT_ASSERT_SAME_DTYPE(a, x);
  HT_ASSERT_SAME_DTYPE(a, output);

  int32_t m = a->shape(0);
  int32_t n = a->shape(1);

  CPUStream cpu_stream(stream);
  HT_DISPATCH_FLOATING_TYPES(output->dtype(), spec_t, "MatVecMul", [&]() {
    auto _future = cpu_stream.EnqueueTask(
    [a, x, trans, output, m, n]() {
      matvecmul_cpu<spec_t>(a->data_ptr<spec_t>(), x->data_ptr<spec_t>(),
                            trans, m, n, output->data_ptr<spec_t>());
    },"MatVecmul");
  });
  NDArray::MarkUsedBy({a, x, output}, stream);
}

} // namespace impl
} // namespace hydraulis
