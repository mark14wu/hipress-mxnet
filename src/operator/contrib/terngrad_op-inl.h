/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file quad_function-inl.h
 * \brief Operator implementing quadratic function.
 * For using as an example in the tutorial of adding operators
 * in MXNet backend.
 */
#ifndef MXNET_OPERATOR_CONTRIB_TERNGRAD_OP_INL_H_
#define MXNET_OPERATOR_CONTRIB_TERNGRAD_OP_INL_H_

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

#include <chrono>
#include <iostream>
#include <chrono>
#include <string>
#include <fstream>

#include <thrust/random.h>
#include <thrust/sort.h>  //sort()
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/functional.h>  //greater<float>
#include <thrust/copy.h>  //copy_if
#include <thrust/iterator/counting_iterator.h>  // counting_iterator
#include <thrust/transform.h> //trnasform
#include <thrust/extrema.h> //minmax_elemetn
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h> //
// #include <thrust/system/cpp/execution_policy.h>

#include "get_policy-inl.h"
#include "ZQ_CPP_LIB/naive_random.hpp"


namespace mxnet {
namespace op {

struct TernGradParam : public dmlc::Parameter<TernGradParam> {
  int bitwidth;
  int random;
  DMLC_DECLARE_PARAMETER(TernGradParam){
    DMLC_DECLARE_FIELD(bitwidth)
      .set_default(2)
      .describe("Number of bits used to quantize original value with type float32 as default. Its value should be  1, 2, 4 or 8");
    DMLC_DECLARE_FIELD(random)
      .set_default(0)
      .describe("Determine whether add random factor.");
  };
};

struct TernGradRParam : public dmlc::Parameter<TernGradRParam>{
  int bitwidth;
  int tail;
  int is_add_to;
  DMLC_DECLARE_PARAMETER(TernGradRParam){
    DMLC_DECLARE_FIELD(bitwidth)
      .set_default(2)
      .describe("Number of bits used to quantize original value with type float32 as default. Its value should be  1, 2, 4 or 8");
    DMLC_DECLARE_FIELD(tail)
      .set_default(0)
      .describe("Number of tail space at last byte in quantized data");
    DMLC_DECLARE_FIELD(is_add_to)
      .set_default(0)
      .describe("0: write to; others: add to");
  };
};

inline bool TernGradOpShape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  const TernGradParam& param = nnvm::get<TernGradParam>(attrs.parsed);
  size_t data_per_byte = 8 / param.bitwidth;
  size_t segment_size = (in_attrs->at(0)[0] + data_per_byte - 1)/data_per_byte;
  size_t output_size = 10+segment_size;

  //SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  if (out_attrs->at(0).ndim() == 0U){ // out not provided
    out_attrs->at(0) = in_attrs->at(0); // copy ndim
    out_attrs->at(0)[0] = output_size;
  }
  else{
    CHECK_GE(out_attrs->at(0)[0],output_size);
  };
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool TernGradROpShape(const nnvm::NodeAttrs& attrs,
                              mxnet::ShapeVector* in_attrs,
                              mxnet::ShapeVector* out_attrs){
  CHECK_EQ(in_attrs->size(),1U);
  CHECK_EQ(out_attrs->size(),1U);
  const TernGradRParam& param = nnvm::get<TernGradRParam>(attrs.parsed);
  size_t data_per_byte = 8 / param.bitwidth;
  size_t output_size = (in_attrs->at(0)[0] - 10)*data_per_byte - param.tail;
  //SHAPE_ASSIGN_CHECK(*out_attrs, 0, in_attrs->at(0));
  if (out_attrs->at(0).ndim() == 0U){
    out_attrs->at(0) = in_attrs->at(0);
    out_attrs->at(0)[0] = output_size;
  }
  else{
    CHECK_GE(out_attrs->at(0)[0],output_size);
  };
  return out_attrs->at(0).ndim() != 0U && out_attrs->at(0).Size() != 0U;
}

inline bool TernGradOpType(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);

  out_attrs->at(0) = 3; //KUint8=3
  return in_attrs->at(0) == 0; //KFloat32 = 0
}
inline bool TernGradROpType(const nnvm::NodeAttrs& attrs,
                              std::vector<int>* in_attrs,
                              std::vector<int>* out_attrs){
  CHECK_EQ(in_attrs->size(), 1U);
  CHECK_EQ(out_attrs->size(), 1U);
  out_attrs->at(0) = 0; //KFloat32 = 0
  return in_attrs->at(0) == 3; //KUint8=3
}

#define to_array(obj,to_type,from_type) ((to_type*)(obj.dptr<from_type>()))



struct compress_without_random{
  float* in_float;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_val;
  float gap_inverse;
  compress_without_random(
    float* a,
    uint8_t b,
    uint8_t c,
    float d,
    float e
  ){
    in_float = a;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_val = d;
    gap_inverse = e;
  }
  __host__ __device__
  uint8_t operator()(const int32_t& i){
    uint8_t qval = 0;
    int j;
    float thetimes;
    uint8_t t;
#pragma unroll
    for (j = 0; j < (1<<data_per_byte_lg2); j++){
      thetimes = (in_float[(i<<data_per_byte_lg2) + j] - min_val) * gap_inverse;
      t = nearbyint(thetimes);
      qval |= (t << (bitwidth*j));
    };
    return qval;
  }
};


struct compress_with_random{
  float* in_float;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_val;
  float gap_inverse;
  unsigned long long timestamp;
  compress_with_random(
    float* a,
    uint8_t b,
    uint8_t c,
    float d,
    float e,
    unsigned long long f
  ){
    in_float = a;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_val = d;
    gap_inverse = e;
    timestamp = f;
  }
  __host__ __device__
  uint8_t operator()(const int32_t& i){
    uint8_t qval = 0;
    int j;
    float thetimes;
    uint8_t t;
    zq_cpp_lib::naive_real_random<float> r(0.0,1.0);
    r.srand(timestamp+i);
#pragma unroll
    for (j = 0; j < (1<<data_per_byte_lg2); j++){
      thetimes = (in_float[(i<<data_per_byte_lg2) + j] - min_val) * gap_inverse;
      thetimes += r();
      t = static_cast<uint8_t>(thetimes);
      qval |= (t << (bitwidth*j));
    };
    return qval;
  }
};

struct decompress_write_to{
  uint8_t* in_uint8_t;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_f;
  float gap;
  decompress_write_to(
    uint8_t* a,
    uint8_t b,
    uint8_t c,
    float d,
    float e
  ){
    in_uint8_t = a;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_f = d;
    gap = e;
  }
  __host__ __device__
  float operator()(const int32_t& i){
    int32_t input_index = (i >> data_per_byte_lg2) + 10;
    uint8_t input_offset = i & ((1 << data_per_byte_lg2) - 1);
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (in_uint8_t[input_index] >> (input_offset * bitwidth)) & mask;
    return static_cast<float>(qval*gap + min_f);
  }
};
struct decompress_add_to{
  uint8_t* in_uint8_t;
  float* out_float;
  uint8_t bitwidth;
  uint8_t data_per_byte_lg2;
  float min_f;
  float gap;
  decompress_add_to(
    uint8_t* a,
    float* out_float_,
    uint8_t b,
    uint8_t c,
    float d,
    float e
  ){
    in_uint8_t = a;
    out_float = out_float_;
    bitwidth = b;
    data_per_byte_lg2 = c;
    min_f = d;
    gap = e;
  }
  __host__ __device__
  float operator()(const int32_t& i){
    int32_t input_index = (i >> data_per_byte_lg2) + 10;
    uint8_t input_offset = i & ((1 << data_per_byte_lg2) - 1);
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (in_uint8_t[input_index] >> (input_offset * bitwidth)) & mask;
    return static_cast<float>(qval*gap + min_f + out_float[i]);
  }
};



template<typename xpu, typename policy_t>
void TernGradROpForward_gpu(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  using namespace mxnet_op;
  policy_t policy = get_policy<policy_t>::get(ctx);
  const TernGradRParam& param = nnvm::get<TernGradRParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);

  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  auto in_uint8_t = to_array(in_data,uint8_t,uint8_t);
  auto out_float = to_array(out_data,float,float);
  uint8_t header[10];
  // cudaMemcpy(header,in_uint8_t,10*sizeof(uint8_t),cudaMemcpyDeviceToHost);
  get_policy<policy_t>::memcpyOutSync(header,in_uint8_t,10*sizeof(uint8_t));
  float min_val = *((float*)(header+2));
  float max_val = *((float*)(header+6));
  float gap = (max_val - min_val) / ((1 << param.bitwidth) - 1.0f);
  uint8_t bitwidth = param.bitwidth; 
  uint8_t tail = param.tail;
  CHECK_EQ(bitwidth,header[0]);
  //ChECK_EQ(tail,header[1]);  // if tail != header, use 0 to replace

  uint8_t lg2[9] = {0,0,1,1,2,2,2,2,3};
  uint8_t bitwidth_lg2 = lg2[bitwidth];
  CHECK_EQ(1<<bitwidth_lg2, bitwidth);

  uint8_t data_per_byte_lg2 = 3 - bitwidth_lg2;
  //uint8_t data_per_byte = 1<<data_per_byte_lg2;


  // mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();


  thrust::counting_iterator<int32_t> index_sequence_begin(0);
  if (param.is_add_to){
    thrust::transform(
      policy,
      index_sequence_begin,
      index_sequence_begin + (((in_data.Size()-10)<<data_per_byte_lg2)-tail),
      out_float,
      decompress_add_to(
        in_uint8_t,
        out_float,
        bitwidth,
        data_per_byte_lg2,
        min_val,
        gap
      )
    );
  }
  else{
    thrust::transform(
      policy,
      index_sequence_begin,
      index_sequence_begin + (((in_data.Size()-10) << data_per_byte_lg2) - tail),
      out_float,
      decompress_write_to(
        in_uint8_t,
        bitwidth,
        data_per_byte_lg2,
        min_val,
        gap
      )
    );
  }
}


template<typename xpu, typename policy_t>
void TernGradOpForward_gpu(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  policy_t policy = get_policy<policy_t>::get(ctx);

  
  
  const TernGradParam& param = nnvm::get<TernGradParam>(attrs.parsed);
  CHECK_EQ(inputs.size(), 1U);
  CHECK_EQ(outputs.size(), 1U);
  CHECK_EQ(req.size(), 1U);
  using namespace mxnet_op;

  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  auto in_float = to_array(in_data,float,float);
  auto out_uint8_t = to_array(out_data,uint8_t,uint8_t);

  float min_val,max_val;
  // MIN_MAX(in_float,in_data.Size(),&min_val, &max_val);
  auto min_max = thrust::minmax_element(
    policy,
    in_float,
    in_float+in_data.Size()
  );
  // cudaMemcpyAsync(&min_val,min_max.first, sizeof(min_val), cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s));
  // cudaMemcpyAsync(&max_val,min_max.second,sizeof(max_val), cudaMemcpyDeviceToHost, mshadow::Stream<gpu>::GetStream(s));
  // cudaStreamSynchronize(mshadow::Stream<gpu>::GetStream(s));
  get_policy<policy_t>::memcpyOut(&min_val,min_max.first,sizeof(min_val),ctx);
  get_policy<policy_t>::memcpyOut(&max_val,min_max.second,sizeof(max_val),ctx);
  get_policy<policy_t>::streamSynchronize(ctx);
  float gap = (max_val - min_val) / ((1 << param.bitwidth) - 1.0f);
  float gap_inverse = 1. / (gap + 1e-8);

  uint8_t lg2[9] = {0,0,1,1,2,2,2,2,3};
  uint8_t bitwidth=(uint8_t)param.bitwidth;
  uint8_t bitwidth_lg2 = lg2[bitwidth];
  CHECK_EQ(1<<bitwidth_lg2, bitwidth);

  uint8_t data_per_byte_lg2 = 3 - bitwidth_lg2;
  uint8_t data_per_byte = 1<<data_per_byte_lg2;
  uint8_t tail = in_data.Size()%data_per_byte;
  tail = tail == 0? 0 : data_per_byte - tail;

  uint8_t header[10];
  ((float*)(header+2))[0] = min_val;
  ((float*)(header+6))[0] = max_val;
  header[0] = bitwidth;
  header[1] = tail;
  // cudaMemcpyAsync(out_uint8_t,header,sizeof(uint8_t)*10,cudaMemcpyHostToDevice,mshadow::Stream<gpu>::GetStream(s));
  get_policy<policy_t>::memcpyIn(out_uint8_t, header, sizeof(uint8_t)*10, ctx);

  thrust::counting_iterator<int32_t> index_sequence_begin(0);
  if (param.random){
    thrust::transform(
      policy,
      index_sequence_begin,
      index_sequence_begin + (in_data.Size() >> data_per_byte_lg2),
      out_uint8_t+10,
      compress_with_random(
        in_float,
        bitwidth,
        data_per_byte_lg2,
        min_val,
        gap_inverse,
        static_cast<unsigned long long>(
          std::chrono::high_resolution_clock::now()
          .time_since_epoch()
          .count()
        )
      )
    );
  }
  else{
    thrust::transform(
      policy,
      index_sequence_begin,
      index_sequence_begin + (in_data.Size() >> data_per_byte_lg2),
      out_uint8_t+10,
      compress_without_random(
        in_float,
        bitwidth,
        data_per_byte_lg2,
        min_val,
        gap_inverse
      )
    );
  }

  uint8_t qval = 0;
  if (tail){
    float tail_data[8];
    // cudaMemcpy(tail_data,in_float+in_data.Size()-data_per_byte,sizeof(float)*(data_per_byte-tail),cudaMemcpyDeviceToHost);
    // in_data.Size() - data_per_byte?
    get_policy<policy_t>::memcpyOut(tail_data,in_float+(in_data.Size()-(data_per_byte-tail)),sizeof(float)*(data_per_byte-tail),ctx);
    get_policy<policy_t>::streamSynchronize(ctx);
    for (auto i = 0; i < data_per_byte - tail; i++){
      uint8_t t = nearbyint((tail_data[i] - min_val)*gap_inverse);
      qval = qval | ( t << (bitwidth*i));
    };
    // cudaMemcpyAsync(out_uint8_t+out_data.Size()-1,&qval,sizeof(uint8_t),cudaMemcpyHostToDevice,mshadow::Stream<gpu>::GetStream(s));
    get_policy<policy_t>::memcpyIn(out_uint8_t+(out_data.Size()-1), &qval, sizeof(uint8_t), ctx);
  };

}




}  // namespace op
}  // namespace mxnet

#endif  // MXNET_OPERATOR_CONTRIB_TERNGRAD_OP_INL_H_
