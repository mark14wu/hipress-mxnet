#ifndef MXNET_OPERATOR_CONTRIB_ecq_sgd_INL_H
#define MXNET_OPERATOR_CONTRIB_ecq_sgd_INL_H

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
#include <thrust/transform.h> //trnasform

#include "get_policy-inl.h"

// #define DEBUG

#include "ZQ_CPP_LIB/time_cost.hpp"
#include "ZQ_CPP_LIB/naive_random.hpp"

namespace mxnet {
namespace op {


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

struct ecq_sgd_param : public dmlc::Parameter<ecq_sgd_param> {
  double alpha;
  double beta;
  int32_t bitwidth;
  DMLC_DECLARE_PARAMETER(ecq_sgd_param){
    DMLC_DECLARE_FIELD(alpha)
      .set_default(1.0)
      .describe("range: (0,?), meaning: influence factor of residual")
      ;
    DMLC_DECLARE_FIELD(beta)
      .set_default(0.8)
      .describe("range: (0,1], meaning: decay factor of residual.")
      ;
    DMLC_DECLARE_FIELD(bitwidth)
      .set_default(2)
      .describe("range: {2,4,8}, meaning: bit width used to express a quantized data.")
      ;
  };
};


inline bool ecq_sgd_shape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->at(0).ndim(), 1U)
    << "Please provide an output vector with ndim = 1";
  auto g_size = in_attrs->at(0)[0];
  auto h_size = in_attrs->at(1)[0];
  auto q_size = out_attrs->at(0)[0];
  CHECK_EQ(g_size, h_size)
    << "Gradients(g) size should be equal to residual(h) size";
  const ecq_sgd_param& param = nnvm::get<ecq_sgd_param>(attrs.parsed);
  uint8_t data_per_byte = 8/param.bitwidth;
  // int32_t min_q_size = static_cast<int32_t>(ceil(g_size*1.0/data_per_byte)) + g_size*4;
  int32_t q_size_1 = static_cast<int32_t>(ceil(g_size*1.0/data_per_byte))+10;
  int32_t q_size_2 = g_size * 4;
  int32_t min_q_size = q_size_1 > q_size_2 ? q_size_1 : q_size_2;
  
  CHECK_GE(q_size, min_q_size)
    << "Output(q) size should >= "
    << min_q_size 
    << " for input size = "
    << g_size
    ;
  return true;
}

inline bool ecq_sgd_type(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << "Input: data(g), residual(h).";
  CHECK_EQ(out_attrs->size(), 1U) << "Please provide output space";
  CHECK_EQ(in_attrs->at(0), 0) << "data(g) type should be float32";
  CHECK_EQ(in_attrs->at(1), 0) << "residual(h) type should be float32";
  CHECK_EQ(out_attrs->at(0),3) << "output(q) type should be uint8";
  return true;
}

struct generate_max_GH_index{
  float* G;
  float* H;
  double alpha;
  generate_max_GH_index(
    float* G_,
    float* H_,
    double alpha_
  ):
  G(G_),
  H(H_),
  alpha(alpha_)
  {}
  __host__ __device__
  int32_t operator()(const int32_t&x, const int32_t&y){
    return fabs(G[x] + alpha*H[x]) > fabs(G[y] + alpha*H[y])
      ? x
      : y;
  }
};

struct generate_GH{
  float* G;
  float* H;
  float* GH;
  double alpha;
  generate_GH(
    float* G_,
    float* H_,
    float* GH_,
    double alpha_
  ):
  G(G_),
  H(H_),
  GH(GH_),
  alpha(alpha_)
  {}
  __host__ __device__
  void operator()(const int32_t&x){
      // GH[x] = fabs(G[x] + h);
      GH[x] = fabs(G[x] + alpha*H[x]);
      // H[x] *= alpha; //4.4 -> 6.0ms
      // H[x] /= 2; //4.4->5.9ms
  }
};

struct generate_Q_update_H{
  float* GH;
  float* H;
  uint8_t* Q;
  double alpha;
  double beta;
  int32_t N;
  float max_GH;
  uint8_t data_per_byte;
  uint8_t bitwidth;
  float gap;
  float gap_inverse;
  uint64_t t;
  generate_Q_update_H(
    float* GH_,
    float* H_,
    uint8_t* Q_,
    double alpha_,
    double beta_,
    int32_t N_,
    float max_GH_,
    uint8_t data_per_byte_,
    uint8_t bitwidth_,
    float gap_,
    float gap_inverse_,
    uint64_t t_
  ):
  GH(GH_),
  H(H_),
  Q(Q_),
  alpha(alpha_),
  beta(beta_),
  N(N_),
  max_GH(max_GH_),
  data_per_byte(data_per_byte_),
  bitwidth(bitwidth_),
  gap(gap_),
  gap_inverse(gap_inverse_),
  t(t_)
  {}
  __host__ __device__
  void operator()(const int32_t x){
    uint8_t q = 0;
    
    int32_t head = x * data_per_byte;
    int32_t tail = head+data_per_byte;
    float f;
    int32_t j;
    zq_cpp_lib::naive_real_random<float>r;
    r.srand(t+x);
    #pragma unroll
    for (j=head; j < tail; j++){
      f = (GH[j] + max_GH)*gap_inverse;
      uint8_t t = static_cast<uint8_t>(f+r());
      q = q | (t << (bitwidth*(j-head)));
      H[j] = beta*H[j] + (f-t)*gap; //7.8ms-5ms
      // H[j] = beta*H[j] + (f-t); //7.8ms-5ms
      // H[j] = (f-t)*gap; //5.2ms-5ms
      // float g_ = t * gap - max_GH;
      // H[j] = beta*H[j] + G[j] - g_; //9ms - 5ms
      // H[j] = max_GH;//5ms-5ms
      // H[j] = beta*H[j] + G[j] - max_GH; //8.5ms - 5ms 
      // H[j] = beta*H[j]; //7.2ms - 5ms 
      // H[j] = 1+H[j]; //5.3ms - 5ms 
    }
    Q[x] = q;
  }
};
struct generate_Q_with_0{
  uint8_t* Q;
  generate_Q_with_0(
    uint8_t* Q_
  ):Q(Q_){};
  __host__ __device__
  void operator()(const int32_t& x){
    Q[x] = 0;
  }
};
struct update_H_with_0{
  float* H;
  double beta;
  update_H_with_0(
    float* H_,
    double beta_
  ): H(H_), beta(beta_){};
  __host__ __device__
  void operator()(const int32_t& x){
    H[x] = H[x] * beta;
  }
};

template<typename xpu, typename policy_t>
void ecq_sgd_func(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
                          
  // struct timeval t_list[10];
  // std::string t_comment[10];
  // int32_t t_index = 0;
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  policy_t policy = get_policy<policy_t>::get(ctx);
  const ecq_sgd_param& param = nnvm::get<ecq_sgd_param>(attrs.parsed);
  const TBlob& in_data = inputs[0];
  const TBlob& residual_data = inputs[1];
  const TBlob& out_data = outputs[0];
  uint8_t* out_uint8_t = to_array(out_data, uint8_t, uint8_t);
  float* G = to_array(in_data, float, float);
  float* H = to_array(residual_data, float, float);
  uint8_t* Q = out_uint8_t+10;
  if (unlikely(param.bitwidth!=2 && param.bitwidth!=4 && param.bitwidth!=8)){
    printf("bitwidth's range: {2,4,8}\tInput param.bitwidth=%d\n", param.bitwidth+0);
    CHECK_EQ(0,1);
  }
  uint8_t bitwidth = static_cast<uint8_t>(param.bitwidth);
  uint8_t s = (1<<(bitwidth-1)) - 1; // (1<<bitwidth - 2) / 2
  uint8_t data_per_byte = 8 / bitwidth;
  int32_t N = in_data.Size();
  int32_t M = static_cast<int32_t>(ceil(N*1.0/data_per_byte));
  float* GH = reinterpret_cast<float*>(out_uint8_t);

  // t_comment[t_index] = "initialize";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(N),
    generate_GH(
      G,
      H,
      GH,
      param.alpha
    )
  );
  // t_comment[t_index] = "generate_GH";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  float* max_GH_p = thrust::max_element(
    policy,
    GH,
    GH+N
  );
  float max_GH;
  get_policy<policy_t>::memcpyOut(
    &max_GH,
    max_GH_p,
    sizeof(float),
    ctx
  );
  // t_comment[t_index] = "find max element in GH";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();

  int32_t out_true_size = 4+4+1+1+M;
  
  if (unlikely(max_GH < 1e-6)){
    // printf("max_GH is 0! Please check your data!");
    // CHECK_EQ(0,1);
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(M-1),
      generate_Q_with_0(
        Q
      )
    );
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      update_H_with_0(
        H,
        param.beta
      )
    );
  }
  else{
    float gap = max_GH / s;
    float gap_inverse = 1.0 / gap;

    // printf("data_per_byte=%d\n",data_per_byte+0);
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(M-1),
      generate_Q_update_H(
        GH,
        H,
        Q,
        param.alpha,
        param.beta,
        N,
        max_GH,
        data_per_byte,
        bitwidth,
        gap,
        gap_inverse,
        std::chrono::high_resolution_clock::now()
        .time_since_epoch()
        .count()
      )
    );
  }

  // get_policy<policy_t>::streamSynchronize(ctx);
  // t_comment[t_index] = "generate_Q_update_H";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  uint8_t header[10];
  memcpy(header,&out_true_size,sizeof(int32_t));
  memcpy(header+4, &max_GH, sizeof(float));
  header[8]=bitwidth;
  header[9]=static_cast<uint8_t>(M*data_per_byte-N);
  get_policy<policy_t>::memcpyIn(
    out_uint8_t,
    header,
    sizeof(uint8_t)*10,
    ctx
  );
  // t_comment[t_index] = "copy several bytes";
  // t_list[t_index++] = zq_cpp_lib::get_timestamp();
  // printf("time cost:\t");
  // for (int32_t i = 0; i < t_index-1; i++){
  //   if (t_comment[i+1] != ""){
  //     printf("(%s)",t_comment[i+1].c_str());
  //   }
  //   auto t_cost = zq_cpp_lib::get_cost_time_by_us(t_list[i],t_list[i+1]);
  //   printf("%.0lf\t", t_cost);
  // }
  // printf("\n");
}


struct ecq_sgd_r_param : public dmlc::Parameter<ecq_sgd_r_param> {
  int32_t is_add_to;
  DMLC_DECLARE_PARAMETER(ecq_sgd_r_param){
    DMLC_DECLARE_FIELD(is_add_to)
      .set_default(0)
      .describe("0: write to; 1: add to")
      ;
  };
};

inline bool ecq_sgd_r_shape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->at(0).ndim(), 1U)
    <<"Please provide an output vector with ndim = 1";
  return true;
}

inline bool ecq_sgd_r_type(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->at(0),3)
    << "input tyep should be uint8";
  CHECK_EQ(out_attrs->at(0),0)
    << "output type should be float32";
  return true;
}


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

struct generate_G{
  uint8_t* Q;
  float* G;
  uint8_t data_per_byte;
  uint8_t bitwidth;
  float max_abs_G;
  float gap;
  generate_G(
    uint8_t* Q_,
    float* G_,
    uint8_t data_per_byte_,
    uint8_t bitwidth_,
    float max_abs_G_,
    float gap_
  ):
  Q(Q_),
  G(G_),
  data_per_byte(data_per_byte_),
  bitwidth(bitwidth_),
  max_abs_G(max_abs_G_),
  gap(gap_)
  {}
  __host__ __device__
  void operator()(const int32_t& x) {
    int32_t Q_index = x / data_per_byte;
    int32_t Q_offset = x % data_per_byte;
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (Q[Q_index] >> (Q_offset*bitwidth))&mask;
    G[x] = qval*gap - max_abs_G;
  }
};

struct generate_G_add_to{
  uint8_t* Q;
  float* G;
  uint8_t data_per_byte;
  uint8_t bitwidth;
  float max_abs_G;
  float gap;
  generate_G_add_to(
    uint8_t* Q_,
    float* G_,
    uint8_t data_per_byte_,
    uint8_t bitwidth_,
    float max_abs_G_,
    float gap_
  ):
  Q(Q_),
  G(G_),
  data_per_byte(data_per_byte_),
  bitwidth(bitwidth_),
  max_abs_G(max_abs_G_),
  gap(gap_)
  {}
  __host__ __device__
  void operator()(const int32_t& x) {
    int32_t Q_index = x / data_per_byte;
    int32_t Q_offset = x % data_per_byte;
    uint8_t mask = (1 << bitwidth) - 1;
    uint8_t qval = (Q[Q_index] >> (Q_offset*bitwidth))&mask;
    G[x] += qval*gap - max_abs_G;
  }
};

template<typename xpu, typename policy_t>
void ecq_sgd_r_func(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  policy_t policy = get_policy<policy_t>::get(ctx);
  const ecq_sgd_r_param& param = nnvm::get<ecq_sgd_r_param>(attrs.parsed);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  auto in_uint8_t = to_array(in_data, uint8_t, uint8_t);
  auto out_float = to_array(out_data, float, float);
  uint8_t header[10];
  get_policy<policy_t>::memcpyOut(
    header,
    in_uint8_t,
    sizeof(uint8_t)*10,
    ctx
  );
  uint8_t* Q = in_uint8_t + 10;
  float* G = out_float;
  int32_t in_size_used;
  float max_abs_G;
  memcpy(&in_size_used, header, sizeof(int32_t));
  memcpy(&max_abs_G, header+4, sizeof(float));
  uint8_t bitwidth = header[8];
  uint8_t unused_nums = header[9];
  int32_t M = in_size_used - 10;
  uint8_t data_per_byte = 8 / bitwidth;
  int32_t N = M * data_per_byte - unused_nums;
  uint8_t s = (1<<(bitwidth-1))-1;
  float gap = max_abs_G / s;
  if (unlikely(out_data.Size() < N)){
    printf("Output space too small: out_data.Size() < N.  %d vs. %d\n",
      static_cast<int32_t>(out_data.Size()),N);
    CHECK_EQ(0,1);
  }
  if (param.is_add_to){
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      generate_G_add_to(
        Q,
        G,
        data_per_byte,
        bitwidth,
        max_abs_G,
        gap
      )
    );
  }
  else{
    thrust::for_each(
      policy,
      thrust::counting_iterator<int32_t>(0),
      thrust::counting_iterator<int32_t>(N),
      generate_G(
        Q,
        G,
        data_per_byte,
        bitwidth,
        max_abs_G,
        gap
      )
    );
  }

}

}
}
#endif