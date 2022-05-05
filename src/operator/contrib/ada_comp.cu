#include "./ada_comp-inl.h"
#include <thrust/execution_policy.h>  //thrust::device

namespace mxnet {
namespace op {


NNVM_REGISTER_OP(_contrib_ada_comp)
.set_attr<FCompute>("FCompute<gpu>", ada_comp_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;
NNVM_REGISTER_OP(_contrib_ada_comp_r)
.set_attr<FCompute>("FCompute<gpu>", ada_comp_r_func<gpu, thrust::cuda_cub::par_t::stream_attachment_type>)
;

}  // namespace op
}  // namespace mxnet
