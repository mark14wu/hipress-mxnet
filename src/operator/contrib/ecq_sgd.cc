#include "./ecq_sgd-inl.h"
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/omp/execution_policy.h>


namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ecq_sgd_param);

NNVM_REGISTER_OP(_contrib_ecq_sgd)
.set_attr_parser(ParamParser<ecq_sgd_param>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data","residual"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ecq_sgd_shape)
.set_attr<nnvm::FInferType>("FInferType", ecq_sgd_type)
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_argument("residual", "NDArray-or-Symbol", "residual array")
.add_arguments(ecq_sgd_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_ecq_sgd)
.set_attr<FCompute>("FCompute<cpu>", ecq_sgd_func<cpu, thrust::detail::host_t>)
;

//======================= r ===================================
DMLC_REGISTER_PARAMETER(ecq_sgd_r_param);

NNVM_REGISTER_OP(_contrib_ecq_sgd_r)
.set_attr_parser(ParamParser<ecq_sgd_r_param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ecq_sgd_r_shape)
.set_attr<nnvm::FInferType>("FInferType", ecq_sgd_r_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(ecq_sgd_r_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_ecq_sgd_r)
.set_attr<FCompute>("FCompute<cpu>", ecq_sgd_r_func<cpu, thrust::detail::host_t>)
;

}  // namespace op
}  // namespace mxnet


// ======================= omp ==============================
// namespace mxnet {
// namespace op {

// NNVM_REGISTER_OP(_contrib_ecq_sgd_omp)
// .set_attr_parser(ParamParser<ecq_sgd_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", ecq_sgd_shape)
// .set_attr<nnvm::FInferType>("FInferType", ecq_sgd_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(ecq_sgd_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_ecq_sgd_omp)
// .set_attr<FCompute>("FCompute<cpu>", ecq_sgd_func<cpu, thrust::system::omp::detail::par_t>)
// ;



// NNVM_REGISTER_OP(_contrib_ecq_sgd_omp_r)
// .set_attr_parser(ParamParser<ecq_sgd_r_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", ecq_sgd_r_shape)
// .set_attr<nnvm::FInferType>("FInferType", ecq_sgd_r_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(ecq_sgd_r_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_ecq_sgd_omp_r)
// .set_attr<FCompute>("FCompute<cpu>", ecq_sgd_r_func<cpu, thrust::system::omp::detail::par_t>)
// ;

// }  // namespace op
// }  // namespace mxnet