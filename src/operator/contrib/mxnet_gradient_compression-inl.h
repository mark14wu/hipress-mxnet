#ifndef CONTRIB_MXNET_GRADIENT_COMPRESSION
#define CONTRIB_MXNET_GRADIENT_COMPRESSION

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

#include "ZQ_CPP_LIB/time_cost.hpp"
namespace mxnet{
namespace op{
  struct MxnetGradientCompressionRParam: public dmlc::Parameter<MxnetGradientCompressionRParam>{
    float threshold;
    int is_add_to;
    DMLC_DECLARE_PARAMETER(MxnetGradientCompressionRParam){
      DMLC_DECLARE_FIELD(threshold)
        .set_default(1.0f)
        .describe("Greater than 0.");
      DMLC_DECLARE_FIELD(is_add_to)
        .set_default(0)
        .describe("0: write to; others: add to");
    }
  };
  struct MxnetGradientCompressionParam: public dmlc::Parameter<MxnetGradientCompressionParam>{
    float threshold;
    DMLC_DECLARE_PARAMETER(MxnetGradientCompressionParam){
      DMLC_DECLARE_FIELD(threshold)
        .set_default(1.0f)
        .describe("Greater than 0.");
    }
  };
  inline bool MxnetGradientCompressionRType(
    const nnvm::NodeAttrs &attrs,
    std::vector<int>* in_attrs,
    std::vector<int>* out_attrs
  ){
    CHECK_EQ(in_attrs->size(),1U) << "Input: to_decompress";
    CHECK_EQ(out_attrs->size(),1U);
    CHECK_EQ(in_attrs->at(0),3) 
      << "Only support decompressiong data with type == uint8.";
    CHECK_EQ(out_attrs->at(0),0) 
      << "Output data type should be float32.";
    return true;
  };
  inline bool MxnetGradientCompressionType(
    const nnvm::NodeAttrs &attrs,
    std::vector<int>* in_attrs,
    std::vector<int>* out_attrs
  ){
    CHECK_EQ(in_attrs->size(),2U) << "Input: to_compress, residual.";
    CHECK_EQ(out_attrs->size(),1U);
    CHECK_EQ(in_attrs->at(0),0) 
      << "Only support compressiong float32 data.";
    CHECK_EQ(in_attrs->at(1),0) 
      << "Residual data type should be same type(float32) with to_compress data.";
    CHECK_EQ(out_attrs->at(0),3) 
      << "Output data type should be uint8.";
    return true;
  };

  inline bool MxnetGradientCompressionRShape(
    const nnvm::NodeAttrs& attrs,
    mxnet::ShapeVector* in_attrs,
    mxnet::ShapeVector* out_attrs
  ){
    auto to_decompress_size = in_attrs->at(0)[0];
    auto original_size = out_attrs->at(0)[0];
    auto min_compressed_size = (original_size + 16 - 1 ) >> 4; // size in float
    min_compressed_size *= (sizeof(float)/sizeof(uint8_t)); // size in uint8
    CHECK_GE(to_decompress_size, min_compressed_size)
      << "to decompressed data size should be greater than or equal to ceil(out_data/16).";
    return true;
  };
  inline bool MxnetGradientCompressionShape(
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
      << "to_compress size should equal to residual size.";
    auto min_out_size = (to_compress_size + 16 - 1 ) >> 4; // size in float
    min_out_size *= (sizeof(float)/sizeof(uint8_t)); // size in uint8
    CHECK_GE(out_size, min_out_size)
      << "out size should be greater than or equal to ceil(to_compress_size/16).";
    return true;
  };
  #define to_array(obj,to_type,from_type) ((to_type*)(obj.dptr<from_type>()))

  template <typename xpu>
  struct mxnet_gradient_compressionr_kernel {
    MSHADOW_XINLINE static void Map(int i,
                                    float *out,
                                    float *in,
                                    const float neg_threshold,
                                    const float pos_threshold) {
      // get position of dequantized value to fill
      float *outval = out + i;
      // gets byte which holds quantized value for this position
      char *ch_ptr = reinterpret_cast<char *>(in + (i >> 4));
      ch_ptr += ((i & 15) >> 2);
      // masks used to quantize data
      const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
      const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
      // col denotes which two bits of a byte are set for this value
      // col=0 implies first two bits, col=3 implies last two bits,...
      const int col = i & 3;
      const uint8_t mask = posbits[col];
      const uint8_t negmask = negbits[col];
      const uint8_t masked = *ch_ptr & mask;
      if (masked == mask) {
        *outval = pos_threshold;
      } else if (masked == negmask) {
        // use posbits for mask as posbits are both 1s
        // then compare masked with negbits to see if only negbits were set
        *outval = neg_threshold;
      } else {
        *outval = 0;
      }
    }
  };

  template <typename xpu>
  struct mxnet_gradient_compression_kernel {
    MSHADOW_XINLINE static void Map(int out_block_id,
                                    int original_size,
                                    float *out,
                                    float *grad,
                                    float *residual,
                                    const float neg_threshold,
                                    const float pos_threshold) {
      // this block contains the compressed representation of
      // upto 16 values starting from out_block_id*16
      float *compr_block = out + out_block_id;
      // init to 0
      *compr_block = 0;
      // start and end are indices in original grad array
      const int start = out_block_id << 4;
      const int end = (start + 16 <= original_size) ? start + 16 : original_size;
      // cast as char* to manipulate bits of float addresses
      char *block_ptr = reinterpret_cast < char * > (compr_block);
      // masks to set bits when value meets pos_threshold
      // 0xc0 is mask when value is to be represented by the first two bits in a char*
      // 0xc0 means first two bits are set to 11
      const uint8_t posbits[] = {0xc0, 0x30, 0x0c, 0x03};
      // masks to set bits when value meets neg_threshold
      const uint8_t negbits[] = {0x80, 0x20, 0x08, 0x02};
      
      for (int i = start; i < end; i++) {
        // adds offset to reach appropriate byte
        char *curr_byte = block_ptr + ((i - start) >> 2);
        // adds gradient to existing residual to get updated grad
        residual[i] += grad[i];
        if (residual[i] >= pos_threshold) {
          // set data to 11
          *curr_byte |= posbits[(i & 3)];
          // reduce residual by pos_threshold
          residual[i] -= pos_threshold;
        } else if (residual[i] <= neg_threshold) {
          // set data to 10
          *curr_byte |= negbits[(i & 3)];
          residual[i] -= neg_threshold;
        }
      }
    }
  };

  template<typename xpu>
  void MxnetGradientCompressionImpl(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>&inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs
  ){
    // zq_cpp_lib::time_cost zq_t;
    // zq_t.start();
    using namespace mxnet_op;
    const MxnetGradientCompressionParam& param
      = nnvm::get<MxnetGradientCompressionParam>(attrs.parsed);
    const TBlob& to_compress = inputs[0];
    const TBlob& residual = inputs[1];
    const TBlob& out_data = outputs[0];
    mshadow::Stream<xpu>*s = ctx.get_stream<xpu>();
    // zq_t.record("initialize");
    Kernel<mxnet_gradient_compression_kernel<xpu>,xpu>::Launch(
      s,
      // out_data.Size(),
      out_data.Size()/(sizeof(float)/sizeof(uint8_t)),
      to_compress.Size(),
      // out_data.dptr<float>(),
      reinterpret_cast<float*>(out_data.dptr<uint8_t>()),
      to_compress.dptr<float>(),
      residual.dptr<float>(),
      -1 * param.threshold,
      param.threshold
    );
    cudaDeviceSynchronize();
    // zq_t.record("mxnet_gradient_compression_kernel");
    // printf("MGC::COMPRESSION:\t");
    // zq_t.print_by_us();
  }
  template<typename xpu>
  void MxnetGradientCompressionRImpl(
    const nnvm::NodeAttrs& attrs,
    const OpContext& ctx,
    const std::vector<TBlob>&inputs,
    const std::vector<OpReqType>& req,
    const std::vector<TBlob>& outputs
  ){
    // zq_cpp_lib::time_cost zq_t;
    // zq_t.start();
    using namespace mxnet_op;
    const MxnetGradientCompressionRParam& param
      = nnvm::get<MxnetGradientCompressionRParam>(attrs.parsed);
    const TBlob& to_decompress = inputs[0];
    const TBlob& out_data = outputs[0];
    mshadow::Stream<xpu>*s = ctx.get_stream<xpu>();
    // zq_t.record("initialize");
    Kernel<mxnet_gradient_compressionr_kernel<xpu>,xpu>::Launch(
      s,
      out_data.Size(),
      out_data.dptr<float>(),
      // to_decompress.dptr<float>(),
      reinterpret_cast<float*>(to_decompress.dptr<uint8_t>()),
      -1 * param.threshold,
      param.threshold
    );
    cudaDeviceSynchronize();
    // zq_t.record("mxnet_gradient_compression_kernel");
    // printf("MGC::DECOMPRESSION:\t");
    // zq_t.print_by_us();
  }

}
}

#endif