#include "./power_sgd-inl.h"
#include <thrust/execution_policy.h>
#include <thrust/copy.h>


namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(power_sgd_param);

// encode 1

NNVM_REGISTER_OP(_contrib_power_sgd_encode1)
.set_attr_parser(ParamParser<power_sgd_param>)
.set_num_inputs(4)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"grad", "q", "residual", "m"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", power_sgd_shape)
.set_attr<nnvm::FInferType>("FInferType", power_sgd_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("grad", "NDArray-or-Symbol", "Input Grad")
.add_argument("q", "NDArray-or-Symbol", "Input Q")
.add_argument("residual", "NDArray-or-Symbol", "Input Residual")
.add_argument("m", "NDArray-or-Symbol", "Input Matrix")
.add_arguments(power_sgd_param::__FIELDS__())
;

NNVM_REGISTER_OP(_contrib_power_sgd_encode1)
.set_attr<FCompute>("FCompute<cpu>", power_sgd_encode1<cpu, thrust::detail::host_t>)
;

// encode 2

NNVM_REGISTER_OP(_contrib_power_sgd_encode2)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<FResourceRequest>("FResourceRequest",
  [](const NodeAttrs& attrs) {
    return std::vector<ResourceRequest>{ResourceRequest::kTempSpace};
  })
.add_argument("p", "NDArray", "Input P")
.add_argument("m", "NDArray", "Input Matrix")
;

NNVM_REGISTER_OP(_contrib_power_sgd_encode2)
.set_attr<FCompute>("FCompute<cpu>", power_sgd_encode2<cpu, thrust::detail::host_t>)
;

// decode

NNVM_REGISTER_OP(_contrib_power_sgd_decode)
.set_num_inputs(5)
.set_num_outputs(0)
.add_argument("grad", "NDArray-or-Symbol", "Grad")
.add_argument("q", "NDArray-or-Symbol", "Q")
.add_argument("residual", "NDArray-or-Symbol", "Residual")
.add_argument("m", "NDArray-or-Symbol", "Matrix")
.add_argument("p", "NDArray-or-Symbol", "P")
;

NNVM_REGISTER_OP(_contrib_power_sgd_decode)
.set_attr<FCompute>("FCompute<cpu>", power_sgd_decode<cpu, thrust::detail::host_t>)
;

}
}