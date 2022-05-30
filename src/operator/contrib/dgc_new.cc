#include "./dgc_new-inl.h"
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/omp/execution_policy.h>


namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(dgc_new_param);

NNVM_REGISTER_OP(_contrib_dgc_new)
.set_attr_parser(ParamParser<dgc_new_param>)
.set_num_inputs(3)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"grad", "u", "v"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", dgc_new_shape)
.set_attr<nnvm::FInferType>("FInferType", dgc_new_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("grad", "NDArray-or-Symbol", "Input Grad")
.add_argument("u", "NDArray-or-Symbol", "Input U")
.add_argument("v", "NDArray-or-Symbol", "Input V")
.add_arguments(dgc_new_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_dgc_new)
.set_attr<FCompute>("FCompute<cpu>", dgc_new_func<cpu, thrust::detail::host_t>)
;

// ======================= server ==============================
DMLC_REGISTER_PARAMETER(dgc_new_server_param);

NNVM_REGISTER_OP(_contrib_dgc_new_server)
.set_attr_parser(ParamParser<dgc_new_server_param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", dgc_new_shape)
.set_attr<nnvm::FInferType>("FInferType", dgc_new_server_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
;


NNVM_REGISTER_OP(_contrib_dgc_new_server)
.set_attr<FCompute>("FCompute<cpu>", dgc_new_server_func<cpu, thrust::detail::host_t>)
;

// ======================= r ===================================
DMLC_REGISTER_PARAMETER(dgc_new_r_param);

NNVM_REGISTER_OP(_contrib_dgc_new_r)
.set_attr_parser(ParamParser<dgc_new_r_param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", dgc_new_r_shape)
.set_attr<nnvm::FInferType>("FInferType", dgc_new_r_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(dgc_new_r_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_dgc_new_r)
.set_attr<FCompute>("FCompute<cpu>", dgc_new_r_func<cpu, thrust::detail::host_t>)
;

}  // namespace op
}  // namespace mxnet


// ======================= omp ==============================
// namespace mxnet {
// namespace op {

// NNVM_REGISTER_OP(_contrib_dgc_new_omp)
// .set_attr_parser(ParamParser<dgc_new_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", dgc_new_shape)
// .set_attr<nnvm::FInferType>("FInferType", dgc_new_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(dgc_new_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_dgc_new_omp)
// .set_attr<FCompute>("FCompute<cpu>", dgc_new_func<cpu, thrust::system::omp::detail::par_t>)
// ;



// NNVM_REGISTER_OP(_contrib_dgc_new_omp_r)
// .set_attr_parser(ParamParser<dgc_new_r_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", dgc_new_r_shape)
// .set_attr<nnvm::FInferType>("FInferType", dgc_new_r_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(dgc_new_r_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_dgc_new_omp_r)
// .set_attr<FCompute>("FCompute<cpu>", dgc_new_r_func<cpu, thrust::system::omp::detail::par_t>)
// ;

// }  // namespace op
// }  // namespace mxnet