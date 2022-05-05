#include "./terngrad_op-inl.h"
#include <thrust/execution_policy.h>  //thrust::device


namespace mxnet {
namespace op {



NNVM_REGISTER_OP(_contrib_terngrad)
.set_attr<FCompute>("FCompute<gpu>", TernGradOpForward_gpu<gpu,thrust::cuda_cub::par_t::stream_attachment_type>);

NNVM_REGISTER_OP(_contrib_terngradr)
.set_attr<FCompute>("FCompute<gpu>", TernGradROpForward_gpu<gpu,thrust::cuda_cub::par_t::stream_attachment_type>);


}  // namespace op
}  // namespace mxnet
