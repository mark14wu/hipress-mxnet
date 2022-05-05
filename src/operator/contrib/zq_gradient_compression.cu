#include "./zq_gradient_compression-inl.h"
#include <thrust/execution_policy.h>  //thrust::device

namespace mxnet{
namespace op{
    NNVM_REGISTER_OP(_contrib_zgc)
        .set_attr<FCompute>("FCompute<gpu>", ZqGradientCompressionImpl<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
        ;
    NNVM_REGISTER_OP(_contrib_zgcr)
        .set_attr<FCompute>("FCompute<gpu>", ZqGradientCompressionRImpl<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
        ;
}
}