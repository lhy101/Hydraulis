#include "hetu/graph/ops/Abs.h"
#include "hetu/graph/ops/Arange.h"
#include "hetu/graph/ops/Arithmetics.h"
#include "hetu/graph/ops/AsStrided.h"
#include "hetu/graph/ops/Attention.h"
#include "hetu/graph/ops/ParallelAttention.h"
#include "hetu/graph/ops/AvgPool.h"
#include "hetu/graph/ops/BatchMatMul.h"
#include "hetu/graph/ops/BatchNorm.h"
#include "hetu/graph/ops/binary_cross_entropy.h"
#include "hetu/graph/ops/Broadcast.h"
#include "hetu/graph/ops/Ceil.h"
#include "hetu/graph/ops/CheckFinite.h"
#include "hetu/graph/ops/Communication.h"
#include "hetu/graph/ops/Concat.h"
#include "hetu/graph/ops/Concatenate.h"
#include "hetu/graph/ops/Contiguous.h"
#include "hetu/graph/ops/dynamic_concatenate.h"
#include "hetu/graph/ops/Conv2d.h"
#include "hetu/graph/ops/data_transfer.h"
#include "hetu/graph/ops/Diagonal.h"
#include "hetu/graph/ops/Dropout.h"
#include "hetu/graph/ops/Dropout2d.h"
#include "hetu/graph/ops/Einsum.h"
#include "hetu/graph/ops/Exp.h"
#include "hetu/graph/ops/EmbeddingLookup.h"
#include "hetu/graph/ops/Floor.h"
#include "hetu/graph/ops/Gather.h"
#include "hetu/graph/ops/Gelu.h"
#include "hetu/graph/ops/group.h"
#include "hetu/graph/ops/InstanceNorm.h"
#include "hetu/graph/ops/Interpolate.h"
#include "hetu/graph/ops/KLDivLoss.h"
#include "hetu/graph/ops/LayerNorm.h"
#include "hetu/graph/ops/LeakyRelu.h"
#include "hetu/graph/ops/Linear.h"
#include "hetu/graph/ops/Log.h"
#include "hetu/graph/ops/Loss.h"
#include "hetu/graph/ops/Maskedfill.h"
#include "hetu/graph/ops/MatDot.h"
#include "hetu/graph/ops/matmul.h"
#include "hetu/graph/ops/MaxPool.h"
#include "hetu/graph/ops/MSELoss.h"
#include "hetu/graph/ops/NLLLoss.h"
#include "hetu/graph/ops/Norm.h"
#include "hetu/graph/ops/Onehot.h"
#include "hetu/graph/ops/ones_like.h"
#include "hetu/graph/ops/optimizer_update.h"
#include "hetu/graph/ops/placeholder.h"
#include "hetu/graph/ops/Pad.h"
#include "hetu/graph/ops/Pow.h"
#include "hetu/graph/ops/Quantization.h"
#include "hetu/graph/ops/RangeMask.h"
#include "hetu/graph/ops/Reduce.h"
#include "hetu/graph/ops/Relu.h"
#include "hetu/graph/ops/Repeat.h"
#include "hetu/graph/ops/Reshape.h"
#include "hetu/graph/ops/RMSNorm.h"
#include "hetu/graph/ops/Roll.h"
#include "hetu/graph/ops/Rotary.h"
#include "hetu/graph/ops/Round.h"
#include "hetu/graph/ops/scalars_like.h"
#include "hetu/graph/ops/Sigmoid.h"
#include "hetu/graph/ops/Sin.h"
#include "hetu/graph/ops/Slice.h"
#include "hetu/graph/ops/Softmax.h"
#include "hetu/graph/ops/SoftmaxCrossEntropy.h"
#include "hetu/graph/ops/SoftmaxCrossEntropySparse.h"
#include "hetu/graph/ops/Split.h"
#include "hetu/graph/ops/Sqrt.h"
#include "hetu/graph/ops/sum.h"
#include "hetu/graph/ops/SwiGLU.h"
#include "hetu/graph/ops/Tanh.h"
#include "hetu/graph/ops/Transpose.h"
#include "hetu/graph/ops/Triu.h"
#include "hetu/graph/ops/update_scale.h"
#include "hetu/graph/ops/variable.h"
#include "hetu/graph/ops/VocabParallelCrossEntropyLoss.h"
#include "hetu/graph/ops/Where.h"
#include "hetu/graph/ops/zeros_like.h"
#include "hetu/graph/ops/Elu.h"
#include "hetu/graph/ops/Hardshrink.h"
#include "hetu/graph/ops/Hardsigmoid.h"
#include "hetu/graph/ops/Hardtanh.h"
#include "hetu/graph/ops/Hardswish.h"
#include "hetu/graph/ops/Logsigmoid.h"
#include "hetu/graph/ops/Silu.h"
#include "hetu/graph/ops/Mish.h"
#include "hetu/graph/ops/Softplus.h"
#include "hetu/graph/ops/Softshrink.h"