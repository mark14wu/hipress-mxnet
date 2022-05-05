#include "./mxnet_gradient_compression-inl.h"
namespace mxnet{
namespace op{
  DMLC_REGISTER_PARAMETER(MxnetGradientCompressionParam);
  DMLC_REGISTER_PARAMETER(MxnetGradientCompressionRParam);

  NNVM_REGISTER_OP(_contrib_mgc)
    .set_attr_parser(ParamParser<MxnetGradientCompressionParam>)
    .set_num_inputs(2)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs){
        return std::vector<std::string>{"data", "residual"};
      }
    )
    .set_attr<mxnet::FInferShape>("FInferShape", MxnetGradientCompressionShape)
    .set_attr<nnvm::FInferType>("FInferType", MxnetGradientCompressionType)
    .set_attr<FCompute>("FCompute<cpu>", MxnetGradientCompressionImpl<cpu>)
    //.set_attr<nnvm::FInplaceOption>
    .add_argument("data", "NDArray-or-Symbol","data")
    .add_argument("residual", "NDArray-or-Symbol", "residual")
    .add_arguments(MxnetGradientCompressionParam::__FIELDS__())
    ;

  NNVM_REGISTER_OP(_contrib_mgcr)
    .set_attr_parser(ParamParser<MxnetGradientCompressionRParam>)
    .set_num_inputs(1)
    .set_num_outputs(1)
    .set_attr<nnvm::FListInputNames>("FListInputNames",
      [](const NodeAttrs& attrs){
        return std::vector<std::string>{"data"};
      }
    )
    .set_attr<mxnet::FInferShape>("FInferShape", MxnetGradientCompressionRShape)
    .set_attr<nnvm::FInferType>("FInferType", MxnetGradientCompressionRType)
    .set_attr<FCompute>("FCompute<cpu>", MxnetGradientCompressionRImpl<cpu>)
    //.set_attr<nnvm::FInplaceOption>
    .add_argument("data", "NDArray-or-Symbol","data")
    .add_arguments(MxnetGradientCompressionRParam::__FIELDS__())
    ;
}

}