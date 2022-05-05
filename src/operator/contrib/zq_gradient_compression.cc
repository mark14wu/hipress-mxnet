#include "./zq_gradient_compression-inl.h"
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/omp/execution_policy.h>

namespace mxnet{
namespace op{
  DMLC_REGISTER_PARAMETER(ZqGradientCompressionParam);
  DMLC_REGISTER_PARAMETER(ZqGradientCompressionRParam);

  NNVM_REGISTER_OP(_contrib_zgc)
    .set_attr_parser(ParamParser<ZqGradientCompressionParam>)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs){
        return std::vector<std::string>{"data", "residual"};
      }
    )
    .set_attr<mxnet::FInferShape>("FInferShape", ZqGradientCompressionShape)
    .set_attr<nnvm::FInferType>("FInferType", ZqGradientCompressionType)
    //.set_attr<nnvm::FInplaceOption>
    .add_argument("data", "NDArray-or-Symbol","data")
    .add_argument("residual", "NDArray-or-Symbol", "residual")
    .add_arguments(ZqGradientCompressionParam::__FIELDS__())
    ;

  NNVM_REGISTER_OP(_contrib_zgcr)
    .set_attr_parser(ParamParser<ZqGradientCompressionRParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs){
        return std::vector<std::string>{"data"};
      }
    )
    .set_attr<mxnet::FInferShape>("FInferShape", ZqGradientCompressionRShape)
    .set_attr<nnvm::FInferType>("FInferType", ZqGradientCompressionRType)
    //.set_attr<nnvm::FInplaceOption>
    .add_argument("data", "NDArray-or-Symbol","data")
    .add_arguments(ZqGradientCompressionRParam::__FIELDS__())
    ;
  
  NNVM_REGISTER_OP(_contrib_zgc)
    .set_attr<FCompute>("FCompute<cpu>", ZqGradientCompressionImpl<cpu, thrust::detail::host_t>)
    ;
  NNVM_REGISTER_OP(_contrib_zgcr)
    .set_attr<FCompute>("FCompute<cpu>", ZqGradientCompressionRImpl<cpu, thrust::detail::host_t>)
    ;

  
  NNVM_REGISTER_OP(_contrib_zgc_omp)
    .set_attr_parser(ParamParser<ZqGradientCompressionParam>)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs){
        return std::vector<std::string>{"data", "residual"};
      }
    )
    .set_attr<mxnet::FInferShape>("FInferShape", ZqGradientCompressionShape)
    .set_attr<nnvm::FInferType>("FInferType", ZqGradientCompressionType)
    //.set_attr<nnvm::FInplaceOption>
    .add_argument("data", "NDArray-or-Symbol","data")
    .add_argument("residual", "NDArray-or-Symbol", "residual")
    .add_arguments(ZqGradientCompressionParam::__FIELDS__())
    ;

  NNVM_REGISTER_OP(_contrib_zgcr_omp)
    .set_attr_parser(ParamParser<ZqGradientCompressionRParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs){
        return std::vector<std::string>{"data"};
      }
    )
    .set_attr<mxnet::FInferShape>("FInferShape", ZqGradientCompressionRShape)
    .set_attr<nnvm::FInferType>("FInferType", ZqGradientCompressionRType)
    //.set_attr<nnvm::FInplaceOption>
    .add_argument("data", "NDArray-or-Symbol","data")
    .add_arguments(ZqGradientCompressionRParam::__FIELDS__())
    ;
  
  NNVM_REGISTER_OP(_contrib_zgc_omp)
    .set_attr<FCompute>("FCompute<cpu>", ZqGradientCompressionImpl<cpu, thrust::system::omp::detail::par_t>)
    ;
  NNVM_REGISTER_OP(_contrib_zgcr_omp)
    .set_attr<FCompute>("FCompute<cpu>", ZqGradientCompressionRImpl<cpu, thrust::system::omp::detail::par_t>)
    ;
}

}