#include "./scdgd-inl.h"
#include <thrust/execution_policy.h>  //thrust::device

namespace mxnet {
namespace op {


NNVM_REGISTER_OP(_contrib_scdgd)
.set_attr<FCompute>("FCompute<gpu>", scdgd_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;
NNVM_REGISTER_OP(_contrib_scdgd_r)
.set_attr<FCompute>("FCompute<gpu>", scdgd_r_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

}  // namespace op
}  // namespace mxnet
