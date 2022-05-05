#include "./ecq_sgd-inl.h"
#include <thrust/execution_policy.h>  //thrust::device

namespace mxnet {
namespace op {


NNVM_REGISTER_OP(_contrib_ecq_sgd)
.set_attr<FCompute>("FCompute<gpu>", ecq_sgd_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;
NNVM_REGISTER_OP(_contrib_ecq_sgd_r)
.set_attr<FCompute>("FCompute<gpu>", ecq_sgd_r_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

}  // namespace op
}  // namespace mxnet
