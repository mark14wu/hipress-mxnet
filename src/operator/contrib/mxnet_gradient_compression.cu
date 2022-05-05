#include "./mxnet_gradient_compression-inl.h"

namespace mxnet{
namespace op{
    NNVM_REGISTER_OP(_contrib_mgc)
        .set_attr<FCompute>("FCompute<gpu>", MxnetGradientCompressionImpl<gpu>)
        ;
    NNVM_REGISTER_OP(_contrib_mgcr)
        .set_attr<FCompute>("FCompute<gpu>", MxnetGradientCompressionRImpl<gpu>)
        ;
}
}