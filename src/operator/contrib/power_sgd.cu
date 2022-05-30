#include "./power_sgd-inl.h"


namespace mxnet {
namespace op {

NNVM_REGISTER_OP(_contrib_power_sgd_encode1)
.set_attr<FCompute>("FCompute<gpu>", power_sgd_encode1<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

NNVM_REGISTER_OP(_contrib_power_sgd_encode2)
.set_attr<FCompute>("FCompute<gpu>", power_sgd_encode2<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

NNVM_REGISTER_OP(_contrib_power_sgd_decode)
.set_attr<FCompute>("FCompute<gpu>", power_sgd_decode<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

}  // namespace op
}  // namespace mxnet