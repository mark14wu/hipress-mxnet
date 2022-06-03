#ifndef MXNET_OPERATOR_CONTRIB_TOPK_ADAM_INL_H
#define MXNET_OPERATOR_CONTRIB_TOPK_ADAM_INL_H

#include <mxnet/operator_util.h>
#include <vector>
#include "../mshadow_op.h"
#include "../mxnet_op.h"
#include "../operator_common.h"
#include "../elemwise_op_common.h"
#include "../tensor/init_op.h"

#include <thrust/random.h>
#include <thrust/sort.h>  //sort()
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/functional.h>  //greater<float>
#include <thrust/copy.h>  //copy_if
#include <thrust/iterator/counting_iterator.h>  // counting_iterator

#include "get_policy-inl.h"
#include "ZQ_CPP_LIB/naive_random.hpp"
#include "ZQ_CPP_LIB/time_cost.hpp"
// #define DEBUG


namespace mxnet {
namespace op {


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

struct topk_adam_param : public dmlc::Parameter<topk_adam_param> {
  double s_percent;
  double sample_rate;
  DMLC_DECLARE_PARAMETER(topk_adam_param){
    DMLC_DECLARE_FIELD(s_percent)
      .set_default(0.001)
      .describe("Range of values:(0.001,0.1], determines how many values should be sent out");
    DMLC_DECLARE_FIELD(sample_rate)
      .set_default(0.001)
      .describe("input.size * sample_rate = sample_cnt");
  };
};

inline bool topk_adam_shape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->at(0).ndim(), 1U) << "Please provide an output vector with ndim = 1";
  auto in_size = in_attrs->at(0)[0];
  auto out_size = out_attrs->at(0)[0];
  auto min_out_size = in_size * 4;
  CHECK_GE(out_size, min_out_size) << "Out size should >= " << min_out_size << "for input size = " << in_size;
  return true;
}

inline bool topk_adam_type(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U) << "Input Data";
  CHECK_EQ(out_attrs->size(), 1U) << "Please provide output space";
  CHECK_EQ(in_attrs->at(0), 0) << "Data type should be float32";
  CHECK_EQ(out_attrs->at(0), 3) <<"output type should be uint8_t";
  return true;
}

struct generate_sample_G{
  float* sample_G;
  float* G;
  int32_t N;  //G_size
  uint64_t t;
  generate_sample_G(
    float* sample_G_,
    float* G_,
    int32_t N_,
    uint64_t t_
  ):
  sample_G(sample_G_),
  G(G_),
  N(N_),
  t(t_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    zq_cpp_lib::naive_int_random<uint32_t> r(0,N-1);
    r.srand(t+x);
    sample_G[x] = abs(G[r()]);
  }
};

struct generate_S_index{
  float* G;
  float threshold;
  generate_S_index(
    float* G_,
    float threshold_
  ):
  G(G_),
  threshold(threshold_)
  {}
  __host__ __device__
  bool operator()(const int32_t& x){
    return (G[x] > threshold) || (G[x] < -threshold);
  }
};

struct greater{
  const float threshold;
  greater(float t): threshold(t){}

  __host__ __device__
  bool operator()(const float&x) const {
    return (x>threshold) || (x<-threshold);
  }
};

struct cmp_float_data_by_int32_index{
  float* G;
  cmp_float_data_by_int32_index(float* G_)
  :G(G_){}
  __host__ __device__
  bool operator()(const int32_t&x, const int32_t& y){
    return abs(G[x]) > abs(G[y]);
  }
};

struct generate_S_value_without_clear{
  int32_t* S_index;
  float* S_value;
  float* G;
  generate_S_value_without_clear(
    int32_t* S_index_,
    float* S_value_,
    float* G_
  ):
  S_index(S_index_),
  S_value(S_value_),
  G(G_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    int32_t i = S_index[x];
    S_value[x] = G[i];
  }
};

struct generate_S_value{
  int32_t* S_index;
  float* S_value;
  float* G;
  generate_S_value(
    int32_t* S_index_,
    float* S_value_,
    float* G_
  ):
  S_index(S_index_),
  S_value(S_value_),
  G(G_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    int32_t i = S_index[x];
    S_value[x] = G[i];
    G[i] = 0;
  }
};

template<typename xpu, typename policy_t>
void topk_adam_encode(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  policy_t policy = get_policy<policy_t>::get(ctx);
  const topk_adam_param& param = nnvm::get<topk_adam_param>(attrs.parsed);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  float* G = to_array(in_data, float, float);
  auto out_uint8_t = to_array(out_data, uint8_t, uint8_t);
  int32_t sample_cnt = static_cast<int32_t>(std::ceil(in_data.Size()*param.sample_rate));
  int32_t expected_selected = static_cast<int32_t>(std::ceil(in_data.Size()*param.s_percent));
  int32_t* header = reinterpret_cast<int32_t*>(out_uint8_t);
  float* sample_G = reinterpret_cast<float*>(out_uint8_t);
  int32_t N = in_data.Size();
  // zt.record("initialize");

  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(sample_cnt),
    generate_sample_G(
      sample_G,
      G,
      N,
      std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count()
    ) 
  );
  // zt.record("generate_sample_G");

  thrust::sort(
    policy,
    sample_G,
    sample_G+sample_cnt,
    thrust::greater<float>()
  );
  // zt.record("sort sample_G");

  float threshold;
  int32_t threshold_index = static_cast<int32_t>(sample_cnt*param.s_percent);
  get_policy<policy_t>::memcpyOut(
    &threshold,
    sample_G + threshold_index,
    sizeof(float),
    ctx
  );
  // zt.record("memcpyOut threshold");
  
  get_policy<policy_t>::streamSynchronize(ctx);
  int32_t* S_index = reinterpret_cast<int32_t*>(out_uint8_t+4);
  int32_t* S_index_end = thrust::copy_if(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(N),
    G,
    S_index,
    greater(threshold)
  );
  // zt.record("copy_if S_index");

  int32_t selected_num = S_index_end - S_index;
  // printf("selected_num=%d\texpected_selected=%d\tthreshold_index=%d\tthreshold=%f\n",selected_num,expected_selected,threshold_index,threshold);
  if (selected_num > expected_selected){
    thrust::sort(
      policy,
      S_index,
      S_index_end,
      cmp_float_data_by_int32_index(G)
    );
    selected_num = expected_selected;
    S_index_end = S_index + selected_num;
  }
  // printf("selected_num=%d\n",selected_num);
  // zt.record("sort S_index");
  
  int32_t out_size = 4 + selected_num*2*4;
  get_policy<policy_t>::memcpyIn(
    header,
    &out_size,
    sizeof(int32_t),
    ctx
  );
  if (unlikely(selected_num == 0)){
    return ;
  }
  // zt.record("memcpyIn header");

  float* S_value = reinterpret_cast<float*>(S_index_end);
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(selected_num),
    generate_S_value(
      S_index,
      S_value,
      G
    )
  );
  // zt.record("generate_S_value");
}

struct topk_adam_server_param : public dmlc::Parameter<topk_adam_server_param> {
  DMLC_DECLARE_PARAMETER(topk_adam_server_param){};
};

inline bool topk_adam_server_shape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) { 
  return true;
}

inline bool topk_adam_server_type(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  return true;
  // CHECK_EQ(in_attrs->size(), 1U) << "Input: data";
  // CHECK_EQ(out_attrs->size(), 1U) << "Please provide output space";
  // CHECK_EQ(in_attrs->at(0) ,0) << "data type should be float32";
  // CHECK_EQ(out_attrs->at(0),3) <<"output type should be uint8_t";
  // return true;
}

template<typename xpu, typename policy_t>
void topk_adam_server_encode(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  // zq_cpp_lib::time_cost zt;
  // zt.start();
  policy_t policy = get_policy<policy_t>::get(ctx);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  auto in_float = to_array(in_data, float, float);
  auto out_uint8_t = to_array(out_data, uint8_t, uint8_t);
  int32_t* header = reinterpret_cast<int32_t*>(out_uint8_t);
  float* G = in_float;
  int32_t N = in_data.Size();
  // zt.record("initialize");

  float threshold = 0;
  get_policy<policy_t>::streamSynchronize(ctx);
  int32_t* S_index = reinterpret_cast<int32_t*>(out_uint8_t+4);
  int32_t* S_index_end = thrust::copy_if(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(N),
    G,
    S_index,
    greater(threshold)
  );
  // zt.record("copy_if S_index");

  int32_t selected_num = S_index_end - S_index;
  int32_t out_size = 4 + selected_num*2*4;
  get_policy<policy_t>::memcpyIn(
    header,
    &out_size,
    sizeof(int32_t),
    ctx
  );
  if (unlikely(selected_num == 0)){
    return ;
  }
  // zt.record("memcpyIn header");

  float* S_value = reinterpret_cast<float*>(S_index_end);
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(selected_num),
    generate_S_value_without_clear(
      S_index,
      S_value,
      G
    )
  );
  // zt.record("generate_S_value");
  // zt.print_by_us();
}


struct topk_adam_r_param : public dmlc::Parameter<topk_adam_r_param> {
  int is_add_to;
  DMLC_DECLARE_PARAMETER(topk_adam_r_param){
    DMLC_DECLARE_FIELD(is_add_to)
      .set_default(1)
      .describe("1: add_to; 0:write_to")
      ;
  };
};

inline bool topk_adam_r_shape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  return true;                             
  // CHECK_EQ(out_attrs->at(0).ndim(),1U)
  //   << "Please provide an output vector with ndim = 1";
  // return true;
}

inline bool topk_adam_r_type(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  return true;                            
  // CHECK_EQ(in_attrs->size(), 1U) << "Input: data";
  // CHECK_EQ(out_attrs->size(), 1U) << "Please provide output space";
  // CHECK_EQ(in_attrs->at(0), 3) << "data type should be uint8_t";
  // CHECK_EQ(out_attrs->at(0), 0) << "output type should be float32";
  // return true;
}


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

struct generate_G_write_to{
  float* G;
  float* S_value;
  int32_t* S_index;
  generate_G_write_to(
    float* G_,
    float* S_value_,
    int32_t* S_index_
  ):
  G(G_),
  S_value(S_value_),
  S_index(S_index_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    G[S_index[x]] = S_value[x];
  }
};

struct generate_G_add_to{
  float* G;
  float* S_value;
  int32_t* S_index;
  generate_G_add_to(
    float* G_,
    float* S_value_,
    int32_t* S_index_
  ):
  G(G_),
  S_value(S_value_),
  S_index(S_index_)
  {}
  __host__ __device__
  void operator()(const int32_t& x){
    G[S_index[x]] += S_value[x];
  }
};

template<typename xpu, typename policy_t>
void topk_adam_decode(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  policy_t policy = get_policy<policy_t>::get(ctx);
  const topk_adam_r_param& param = nnvm::get<topk_adam_r_param>(attrs.parsed);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  auto in_uint8_t = to_array(in_data, uint8_t, uint8_t);
  auto out_float = to_array(out_data, float, float);
  int32_t M;
  int32_t* header = reinterpret_cast<int32_t*>(in_uint8_t);
  float* G = out_float;
  int32_t* S_index = reinterpret_cast<int32_t*>(in_uint8_t+4);
  get_policy<policy_t>::memcpyOut(
    &M,
    header,
    sizeof(int32_t),
    ctx
  );
  int32_t S_index_size = (M-4)/8;
  float* S_value = reinterpret_cast<float*>(S_index+S_index_size);
  if (unlikely(in_data.Size() < M)){
    printf("input space provided is not enough! in_data.Size()=%d\tM=%d\n",
      static_cast<int32_t>(in_data.Size()), 
      M); 
    CHECK_EQ(0,1);
  }
  // we don't check outputspace is enough or not, user should be careful about this.
  if (param.is_add_to){
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(S_index_size),
      generate_G_add_to(
        G,
        S_value,
        S_index
      )
    ); 
  }
  else{
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(S_index_size),
      generate_G_write_to(
        G,
        S_value,
        S_index
      )
    ); 
  }
  
}

}
}
#endif