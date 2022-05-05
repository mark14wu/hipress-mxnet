#include "./scdgd-inl.h"
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/omp/execution_policy.h>


namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(scdgd_param);

NNVM_REGISTER_OP(_contrib_scdgd)
.set_attr_parser(ParamParser<scdgd_param>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<mxnet::FInferShape>("FInferShape", scdgd_shape)
.set_attr<nnvm::FInferType>("FInferType", scdgd_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_argument("residual", "NDArray-or-Symbol", "residual")
.add_arguments(scdgd_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_scdgd)
.set_attr<FCompute>("FCompute<cpu>", scdgd_func<cpu, thrust::detail::host_t>)
;

// ======================= r ===================================
DMLC_REGISTER_PARAMETER(scdgd_r_param);

NNVM_REGISTER_OP(_contrib_scdgd_r)
.set_attr_parser(ParamParser<scdgd_r_param>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", scdgd_r_shape)
.set_attr<nnvm::FInferType>("FInferType", scdgd_r_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(scdgd_r_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_scdgd_r)
.set_attr<FCompute>("FCompute<cpu>", scdgd_r_func<cpu, thrust::detail::host_t>)
;

}  // namespace op
}  // namespace mxnet


// ======================= omp ==============================
// namespace mxnet {
// namespace op {

// NNVM_REGISTER_OP(_contrib_scdgd_omp)
// .set_attr_parser(ParamParser<scdgd_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", scdgd_shape)
// .set_attr<nnvm::FInferType>("FInferType", scdgd_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(scdgd_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_scdgd_omp)
// .set_attr<FCompute>("FCompute<cpu>", scdgd_func<cpu, thrust::system::omp::detail::par_t>)
// ;



// NNVM_REGISTER_OP(_contrib_scdgd_omp_r)
// .set_attr_parser(ParamParser<scdgd_r_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", scdgd_r_shape)
// .set_attr<nnvm::FInferType>("FInferType", scdgd_r_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(scdgd_r_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_scdgd_omp_r)
// .set_attr<FCompute>("FCompute<cpu>", scdgd_r_func<cpu, thrust::system::omp::detail::par_t>)
// ;

// }  // namespace op
// }  // namespace mxnet