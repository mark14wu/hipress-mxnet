#include "./topk_adam-inl.h"
#include <thrust/execution_policy.h>  //thrust::device


namespace mxnet {
namespace op {

// ======================= encode ==============================
DMLC_REGISTER_PARAMETER(topk_adam_param);

NNVM_REGISTER_OP(_contrib_topk_adam_encode)
.set_attr_parser(ParamParser<topk_adam_param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", topk_adam_shape)
.set_attr<nnvm::FInferType>("FInferType", topk_adam_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input Data(Adam)")
.add_arguments(topk_adam_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_topk_adam_encode)
.set_attr<FCompute>("FCompute<cpu>", topk_adam_encode<cpu, thrust::detail::host_t>)
;

// ======================= server ==============================
DMLC_REGISTER_PARAMETER(topk_adam_server_param);

NNVM_REGISTER_OP(_contrib_topk_adam_server_encode)
.set_attr_parser(ParamParser<topk_adam_server_param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", topk_adam_server_shape)
.set_attr<nnvm::FInferType>("FInferType", topk_adam_server_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
;


NNVM_REGISTER_OP(_contrib_topk_adam_server_encode)
.set_attr<FCompute>("FCompute<cpu>", topk_adam_server_encode<cpu, thrust::detail::host_t>)
;

// ======================= decode ==============================
DMLC_REGISTER_PARAMETER(topk_adam_r_param);

NNVM_REGISTER_OP(_contrib_topk_adam_decode)
.set_attr_parser(ParamParser<topk_adam_r_param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", topk_adam_r_shape)
.set_attr<nnvm::FInferType>("FInferType", topk_adam_r_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(topk_adam_r_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_topk_adam_decode)
.set_attr<FCompute>("FCompute<cpu>", topk_adam_decode<cpu, thrust::detail::host_t>)
;

}  // namespace op
}  // namespace mxnet