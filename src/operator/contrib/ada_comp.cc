#include "./ada_comp-inl.h"
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/omp/execution_policy.h>


namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(ada_comp_param);

NNVM_REGISTER_OP(_contrib_ada_comp)
.set_attr_parser(ParamParser<ada_comp_param>)
.set_num_inputs(2)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data","residual"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ada_comp_shape)
.set_attr<nnvm::FInferType>("FInferType", ada_comp_type)
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_argument("residual", "NDArray-or-Symbol", "residual")
.add_arguments(ada_comp_param::__FIELDS__())
;


NNVM_REGISTER_OP(_contrib_ada_comp)
.set_attr<FCompute>("FCompute<cpu>", ada_comp_func<cpu, thrust::detail::host_t>)
;

//======================= r ===================================

NNVM_REGISTER_OP(_contrib_ada_comp_r)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", ada_comp_r_shape)
.set_attr<nnvm::FInferType>("FInferType", ada_comp_r_type)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
;


NNVM_REGISTER_OP(_contrib_ada_comp_r)
.set_attr<FCompute>("FCompute<cpu>", ada_comp_r_func<cpu, thrust::detail::host_t>)
;

}  // namespace op
}  // namespace mxnet


// ======================= omp ==============================
// namespace mxnet {
// namespace op {

// NNVM_REGISTER_OP(_contrib_ada_comp_omp)
// .set_attr_parser(ParamParser<ada_comp_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", ada_comp_shape)
// .set_attr<nnvm::FInferType>("FInferType", ada_comp_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(ada_comp_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_ada_comp_omp)
// .set_attr<FCompute>("FCompute<cpu>", ada_comp_func<cpu, thrust::system::omp::detail::par_t>)
// ;



// NNVM_REGISTER_OP(_contrib_ada_comp_omp_r)
// .set_attr_parser(ParamParser<ada_comp_r_param>)
// .set_num_inputs(1)
// .set_num_outputs(1)
// .set_attr<nnvm::FListInputNames>("FListInputNames",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::string>{"data"};
//   })
// .set_attr<mxnet::FInferShape>("FInferShape", ada_comp_r_shape)
// .set_attr<nnvm::FInferType>("FInferType", ada_comp_r_type)
// .set_attr<nnvm::FInplaceOption>("FInplaceOption",
//   [](const NodeAttrs& attrs) {
//     return std::vector<std::pair<int, int> >{{0, 0}};
//   })
// .add_argument("data", "NDArray-or-Symbol", "Input ndarray")
// .add_arguments(ada_comp_r_param::__FIELDS__())
// ;


// NNVM_REGISTER_OP(_contrib_ada_comp_omp_r)
// .set_attr<FCompute>("FCompute<cpu>", ada_comp_r_func<cpu, thrust::system::omp::detail::par_t>)
// ;

// }  // namespace op
// }  // namespace mxnet