#ifndef CONTRIB_zq_GRADIENT_COMPRESSION
#define CONTRIB_zq_GRADIENT_COMPRESSION

#include <dmlc/parameter.h>
#include <mshadow/base.h>
#include <mxnet/op_attr_types.h>
#include <mxnet/operator.h>
#include <mxnet/operator_util.h>
#include <nnvm/op.h>
#include <nnvm/op_attr_types.h>
#include <vector>
#include "../elemwise_op_common.h"
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../tensor/init_op.h"
#include "../tensor/util/tensor_util-inl.h"

#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/functional.h>  //greater<float>
#include <thrust/iterator/counting_iterator.h>  // counting_iterator
#include <thrust/transform.h> //trnasform

#include "get_policy-inl.h"

namespace mxnet{
namespace op{
  struct ZqGradientCompressionRParam: public dmlc::Parameter<ZqGradientCompressionRParam>{
    float threshold;
    int is_add_to;
    DMLC_DECLARE_PARAMETER(ZqGradientCompressionRParam){
      DMLC_DECLARE_FIELD(threshold)
        .set_default(1.0f)
        .describe("Greater than 0.");
      DMLC_DECLARE_FIELD(is_add_to)
        .set_default(0)
        .describe("0: write to; others: add to");
    }
  };
  struct ZqGradientCompressionParam: public dmlc::Parameter<ZqGradientCompressionParam>{
    float threshold;

    DMLC_DECLARE_PARAMETER(ZqGradientCompressionParam){
      DMLC_DECLARE_FIELD(threshold)
        .set_default(1.0f)
        .describe("Greater than 0.");
    }
  };
  inline bool ZqGradientCompressionRType(
    const nnvm::NodeAttrs &attrs,
    std::vector<int>* in_attrs,
    std::vector<int>* out_attrs
  ){
    CHECK_EQ(in_attrs->size(),1U) << "Input: data";
    CHECK_EQ(out_attrs->size(),1U);
    CHECK_EQ(in_attrs->at(0),3) 
      << "Only support decompression data with type == uint8.";
    CHECK_EQ(out_attrs->at(0),0) 
      << "Output data type should be float32.";
    return true;
  };
  inline bool ZqGradientCompressionType(
    const nnvm::NodeAttrs &attrs,
    std::vector<int>* in_attrs,
    std::vector<int>* out_attrs
  ){
    CHECK_EQ(in_attrs->size(),2U) << "Input: data, residual.";
    CHECK_EQ(out_attrs->size(),1U);
    CHECK_EQ(in_attrs->at(0),0) 
      << "Only support compressiong float32 data.";
    CHECK_EQ(in_attrs->at(1),0) 
      << "Residual data type should be same type(float32) with data data.";
    CHECK_EQ(out_attrs->at(0),3) 
      << "Output data type should be uint8.";
    return true;
  };

  inline bool ZqGradientCompressionRShape(
    const nnvm::NodeAttrs& attrs,
    mxnet::ShapeVector* in_attrs,
    mxnet::ShapeVector* out_attrs
  ){
    auto to_decompress_size = in_attrs->at(0)[0];
    auto original_size = out_attrs->at(0)[0];
    auto min_compressed_size = (original_size + 4 - 1 ) >> 2;
    CHECK_GE(to_decompress_size, min_compressed_size)
      << "to decompressed data size should be greater than or equal to ceil(out_data/16).";
    return true;
  };
  inline bool ZqGradientCompressionShape(
    const nnvm::NodeAttrs& attrs,
    mxnet::ShapeVector* in_attrs,
    mxnet::ShapeVector* out_attrs
  ){
    CHECK_EQ(out_attrs->at(0).ndim(),1U) 
      << "please provide an output vector with ndim == 1";
    auto to_compress_size = in_attrs->at(0)[0];
    auto residual_size = in_attrs->at(1)[0];
    auto out_size = out_attrs->at(0)[0];
    CHECK_EQ(to_compress_size, residual_size)
      << "data size should equal to residual size.";
    auto min_out_size = (to_compress_size + 16 - 1 ) >> 4;
    CHECK_GE(out_size, min_out_size)
      << "out size should be greater than or equal to ceil(to_compress_size/16).";
    return true;
  };
  #define to_array(obj,to_type,from_type) ((to_type*)(obj.dptr<from_type>()))

  struct compressr_write_to{
    uint8_t* in;
    float* out;
    float threshold;
    compressr_write_to(
      uint8_t* _in,
      float* _out,
      float _threshold
    ):in(_in),out(_out),threshold(_threshold){}
    __host__ __device__
    float operator()(const int32_t&i) const{
      int32_t input_index = i>>2;
      int32_t input_offset = i&3;
      uint8_t qval = (in[input_index] >> (input_offset<<1)) & 3;
      return (static_cast<float>(qval)-1)*threshold;
    }
  };
  struct compressr_add_to{
    uint8_t* in;
    float* out;
    float threshold;
    compressr_add_to(
      uint8_t* _in,
      float* _out,
      float _threshold
    ):in(_in),out(_out),threshold(_threshold){}
    __host__ __device__
    float operator()(const int32_t&i) const{
      int32_t input_index = i>>2;
      int32_t input_offset = i&3;
      uint8_t qval = (in[input_index] >> (input_offset<<1)) & 3;
      return (static_cast<float>(qval)-1)*threshold + out[i];
    }
  };
  
  struct compress{
    float *grad, *residual;
    float threshold;
    compress(
      float *b,
      float *c,
      float d
    ){
      grad = b;
      residual = c;
      threshold = d;
    }
    __host__ __device__
    uint8_t operator()(const int32_t&i) const{
      int32_t start = i<<2;
      uint8_t qval = 85;
      int32_t k,j;
      for (k = 0; k < 4; k++){
        j = k + start;
        residual[j] += grad[j];
        if (residual[j] >= threshold){
          qval ^= (3<<(k<<1));
          residual[j] -= threshold;
        }
        else if (residual[j] <= -threshold){
          qval ^= (1<<(k<<1));
          residual[j] += threshold;
        }
      }
      return qval;
    }
  };


  
  template<typename xpu, typename policy_t>
  void ZqGradientCompressionImpl(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>&inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs
  ){
    using namespace mxnet_op;
    const ZqGradientCompressionParam& param
      = nnvm::get<ZqGradientCompressionParam>(attrs.parsed);
    const TBlob& data = inputs[0];
    const TBlob& residual = inputs[1];
    const TBlob& out_data = outputs[0];
    float* to_compress_float = to_array(data,float,float);
    float* residual_float = to_array(residual, float, float);
    uint8_t* out_uint8_t = to_array(out_data,uint8_t, uint8_t);
    thrust::counting_iterator<int32_t> index_sequence_begin(0);
    thrust::transform(
      get_policy<policy_t>::get(ctx),
      index_sequence_begin,
      index_sequence_begin + (data.Size() >> 2), //  (x+3) >> 2 ; take ceil 
      out_uint8_t,
      compress(
        to_compress_float,
        residual_float,
        param.threshold
      )
    );
    uint8_t left = data.Size() & 3; // data.Size() % 4
    if (left){
      float left_grad[4];
      float left_resi[4];
      get_policy<policy_t>::memcpyOut(
        left_grad,
        to_compress_float+(data.Size()-left),
        sizeof(float)*left,
        ctx
      );
      get_policy<policy_t>::memcpyOut(
        left_resi,
        residual_float+(data.Size()-left),
        sizeof(float)*left,
        ctx
      );
      get_policy<policy_t>::streamSynchronize(ctx);
      uint8_t qval = 85;
      for (auto j = 0; j < left; j++){
        left_resi[j] += left_grad[j];
        if (left_resi[j] >= param.threshold){
          qval ^= (3<<(j<<1));
          left_resi[j] -= param.threshold;
        }
        else if (left_resi[j] <= -param.threshold){
          qval ^= (1<<(j<<1));
          left_resi[j] += param.threshold;
        }
      }
      get_policy<policy_t>::memcpyIn(
        residual_float + (data.Size() - left),
        left_resi,
        sizeof(float)*left,
        ctx
      );
      get_policy<policy_t>::memcpyIn(
        out_uint8_t + (data.Size() >> 2),
        &qval,
        sizeof(uint8_t),
        ctx
      );
    }
  }
  
  template<typename xpu, typename policy_t>
  void ZqGradientCompressionRImpl(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>&inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs
  ){
    using namespace mxnet_op;
    const ZqGradientCompressionRParam& param
      = nnvm::get<ZqGradientCompressionRParam>(attrs.parsed);
    const TBlob& in_data = inputs[0];
    const TBlob& out_data = outputs[0];
    float* out_float = to_array(out_data, float, float);
    uint8_t* in_uint8_t = to_array(in_data, uint8_t, uint8_t);
    thrust::counting_iterator<int32_t> index_sequence_begin(0);
    if (param.is_add_to){
      thrust::transform(
        get_policy<policy_t>::get(ctx),
        index_sequence_begin,
        index_sequence_begin + (out_data.Size()),
        out_float,
        compressr_add_to(
          in_uint8_t,
          out_float,
          param.threshold
        )
      );
    }
    else{
      thrust::transform(
        get_policy<policy_t>::get(ctx),
        index_sequence_begin,
        index_sequence_begin + (out_data.Size()),
        out_float,
        compressr_write_to(
          in_uint8_t,
          out_float,
          param.threshold
        )
      );
    }
  }

}
}

#endif