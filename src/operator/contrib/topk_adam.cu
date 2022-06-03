#include "./topk_adam-inl.h"
#include <thrust/execution_policy.h>  //thrust::device

namespace mxnet {
namespace op {


NNVM_REGISTER_OP(_contrib_topk_adam_encode)
.set_attr<FCompute>("FCompute<gpu>", topk_adam_encode<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;
NNVM_REGISTER_OP(_contrib_topk_adam_server_encode)
.set_attr<FCompute>("FCompute<gpu>", topk_adam_server_encode<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;
NNVM_REGISTER_OP(_contrib_topk_adam_decode)
.set_attr<FCompute>("FCompute<gpu>", topk_adam_decode<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

}  // namespace op
}  // namespace mxnet
