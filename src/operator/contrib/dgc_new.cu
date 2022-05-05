#include "./dgc_new-inl.h"
#include <thrust/execution_policy.h>  //thrust::device

namespace mxnet {
namespace op {


NNVM_REGISTER_OP(_contrib_dgc_new)
.set_attr<FCompute>("FCompute<gpu>", dgc_new_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;
NNVM_REGISTER_OP(_contrib_dgc_new_r)
.set_attr<FCompute>("FCompute<gpu>", dgc_new_r_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

}  // namespace op
}  // namespace mxnet
