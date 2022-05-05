#ifndef MXNET_OPERATOR_CONTRIB_ada_comp_INL_H
#define MXNET_OPERATOR_CONTRIB_ada_comp_INL_H

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


namespace mxnet {
namespace op {

struct ada_comp_param : public dmlc::Parameter<ada_comp_param> {
  int32_t T;
  DMLC_DECLARE_PARAMETER(ada_comp_param){
    DMLC_DECLARE_FIELD(T)
      .set_default(64)
      .describe("size of bins")
      ;
  };
};

inline bool ada_comp_shape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->at(0).ndim(),1U)
    << "Please provide an output vector with ndim = 1";
  auto data_size = in_attrs->at(0)[0];
  auto residual_size = in_attrs->at(1)[0];
  auto out_size = out_attrs->at(0)[0];
  CHECK_EQ(data_size, residual_size)
    << "data size should be equal to residual size";
  const ada_comp_param& param = nnvm::get<ada_comp_param>(attrs.parsed);
  CHECK(32<=param.T && param.T <= 1024)
    << "param.T should be in range[2,1024]";
  double N = static_cast<double>(data_size);
  auto min_out_size = static_cast<int32_t>(
    2*4 + 4*N+ceil(N/8)
  );
  CHECK_GE(out_size,min_out_size)
    << "Out size should >= " << min_out_size << " for input size = " << data_size;
  return true;
}

inline bool ada_comp_type(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 2U) << "Input: data, residual.";
  CHECK_EQ(out_attrs->size(), 1U) << "Please provide output space";
  CHECK_EQ(in_attrs->at(0),0)
    <<"data type should be float32";
  CHECK_EQ(in_attrs->at(1),0)
    <<"residual type should be float32";
  CHECK_EQ(out_attrs->at(0),3)
    <<"Output type should be uint8_t";
  return true;
}


#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

struct generate_G_H{
  float *W,*R,*H;
  generate_G_H(
    float* W_,
    float* R_,
    float* H_
  ): W(W_), R(R_), H(H_)
  {}
  __host__ __device__
  void operator()(const int32_t&x)const{
    R[x] = W[x] + R[x];
    H[x] = W[x] + R[x];
  }
};


template <typename T>
struct abs_max{
  __host__ __device__
  T operator()(const T &a, const T &b) const{
    T c = fabs(a);
    T d = fabs(b);
    return c > d ? c : d;
  }
};

template<typename policy_t>
struct generate_g_max{
  policy_t policy;
  float *G, *g_max;
  int32_t T;
  int32_t N;
  generate_g_max(
    policy_t policy_,
    float* G_,
    float* g_max_,
    int32_t T_,
    int32_t N_
  ):
  policy(policy_),
  G(G_),
  g_max(g_max_),
  T(T_),
  N(N_)
  {}
  __host__ __device__
  void operator()(const int32_t&x) const{
    int32_t head = x * T;
    int32_t tail = head+T < N ? head+T : N;
    g_max[x] = thrust::reduce(
      policy,
      G+head,
      G+tail,
      static_cast<float>(0),
      abs_max<float>()
    );
  }
};

struct generate_Index{
  float *H;
  float *g_max;
  int32_t T;
  generate_Index(
    float *H_,
    float* g_max_,
    int32_t T_
  ):
  H(H_),
  g_max(g_max_),
  T(T_)
  {}
  __host__ __device__
  bool operator()(const int32_t&x) const{
    return fabs(H[x]) >= g_max[x/T];
  }
};
struct generate_Gq{
  int32_t *Index;
  float *G;
  float *R;
  uint8_t *Gq;
  int32_t Index_size;
  float ave_g_max;
  generate_Gq(
    int32_t *Index_,
    float *G_,
    float *R_,
    uint8_t *Gq_,
    int32_t Index_size_,
    float ave_g_max_
  ):
  Index(Index_),
  G(G_),
  R(R_),
  Gq(Gq_),
  Index_size(Index_size_),
  ave_g_max(ave_g_max_)
  {}
  __host__ __device__
  void operator()(const int32_t&x) const{
    uint8_t q = 0;
    int32_t head = x * 8;
    // int32_t tail = std::min(head+8, Index_size);
    int32_t tail = head+8 < Index_size ? head+8 : Index_size;
    for (int32_t i = head; i < tail; i++){
      auto j = Index[i];
      uint8_t tmp = !!(G[j]>0);
      q = q | (tmp << (i - head));
      tmp = tmp * 2 - 1;
      R[j] = R[j] - tmp * ave_g_max;
    }
    Gq[x] = q;
  }
};

template<typename xpu, typename policy_t>
void ada_comp_func(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {
  policy_t policy = get_policy<policy_t>::get(ctx);
  const ada_comp_param& param = nnvm::get<ada_comp_param>(attrs.parsed);
  const TBlob& in_data = inputs[0];
  const TBlob& residual = inputs[1];
  const TBlob& out_data = outputs[0];
  int32_t N = in_data.Size();
  auto out_uint8_t = to_array(out_data, uint8_t, uint8_t);
  auto residual_float = to_array(residual, float, float);
  auto in_float = to_array(in_data, float, float);
  auto W = in_float;
  auto R = residual_float;
  auto H = reinterpret_cast<float*>(out_uint8_t+2*4);
  auto G = residual_float;
  auto g_max = reinterpret_cast<float*>(out_uint8_t+static_cast<int32_t>(2*4+N*4));
  //g_max_size = N/T * 4
  auto Index = reinterpret_cast<int32_t*>(out_uint8_t+2*4);
  double N_double = static_cast<double>(N);

  // printf("generate_G_H...\n");
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(N),
    generate_G_H(
      W,
      R,
      H
    )
  );
  
  
  auto g_max_size = static_cast<int32_t>(ceil(N_double/param.T));
  // printf("generate g_max...\n");
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(g_max_size),
    generate_g_max<policy_t>(
      policy,
      G,
      g_max,
      param.T,
      N
    )
  );

  // printf("generate sum_g_max...\n");
  float sum_g_max = thrust::reduce(
    policy,
    g_max,
    g_max + g_max_size
  );
  float ave_g_max = sum_g_max / g_max_size;

  // printf("generate Index...\n");
  auto Index_end = thrust::copy_if(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(N),
    Index,
    generate_Index(
      H,
      g_max,
      param.T
    )
  );
  auto Index_size = Index_end - Index;
  auto Gq_size = static_cast<int32_t>(ceil(static_cast<double>(Index_size)/8));
  int32_t output_size = 2*4 + Index_size * 4 + Gq_size;

  // printf("copy in header...\n");
  get_policy<policy_t>::memcpyIn(
    reinterpret_cast<int32_t*>(out_uint8_t),
    &output_size,
    sizeof(int32_t),
    ctx
  );
  get_policy<policy_t>::memcpyIn(
    reinterpret_cast<float*>(out_uint8_t+4),
    &ave_g_max,
    sizeof(float),
    ctx
  );
  auto Gq = reinterpret_cast<uint8_t*>(Index_end);
  // printf("generate Gq...\n");
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(Gq_size),
    generate_Gq(
      Index,
      G,
      R,
      Gq,
      Index_size,
      ave_g_max
    )
  );
  
}


inline bool ada_comp_r_shape(const nnvm::NodeAttrs& attrs,
                             mxnet::ShapeVector* in_attrs,
                             mxnet::ShapeVector* out_attrs) {
  CHECK_EQ(out_attrs->at(0).ndim(),1U)
    << "Please provide an output vector with ndim = 1";
  return true;
}

inline bool ada_comp_r_type(const nnvm::NodeAttrs& attrs,
                            std::vector<int>* in_attrs,
                            std::vector<int>* out_attrs) {
  CHECK_EQ(in_attrs->size(), 1U)
    << "Input: data";
  CHECK_EQ(out_attrs->size(), 1U)
    << "Please provide output space";
  CHECK_EQ(in_attrs->at(0),3)
    << "data type should be uint8_t";
  CHECK_EQ(out_attrs->at(0),0)
    << "Output type should be float32";
  return true;
}

struct generate_out{
  uint8_t *Gq;
  int32_t *Index;
  float *G;
  int32_t Index_size;
  float v;
  generate_out(
    uint8_t *Gq_,
    int32_t *Index_,
    float *G_,
    int32_t Index_size_,
    float v_
  ):
  Gq(Gq_),
  Index(Index_),
  G(G_),
  Index_size(Index_size_),
  v(v_)
  {};
  __host__ __device__
  void operator()(const int32_t& x) const{
    uint8_t q = Gq[x];
    int32_t head = x*8;
    // int32_t tail = std::min(Index_size, head+8);
    int32_t tail = Index_size < head+8 ? Index_size : head+8;
    for (int32_t i = head; i < tail; i++){
      int32_t j = Index[i];
      int32_t t = !!(q&(1<<(i-head)));
      t = t*2 - 1;
      G[j] = v*t;
    }
  }
};

template<typename xpu, typename policy_t>
void ada_comp_r_func(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  policy_t policy = get_policy<policy_t>::get(ctx);
  const TBlob& in_data = inputs[0];
  const TBlob& out_data = outputs[0];
  auto in_uint8_t = to_array(in_data, uint8_t, uint8_t);
  auto out_float = to_array(out_data, float, float);
  int32_t M;
  float v;
  get_policy<policy_t>::memcpyOut(
    &M,
    reinterpret_cast<int32_t*>(in_uint8_t),
    sizeof(int32_t),
    ctx
  );
  get_policy<policy_t>::memcpyOut(
    &v,
    reinterpret_cast<float*>(in_uint8_t+4),
    sizeof(float),
    ctx
  );
  get_policy<policy_t>::streamSynchronize(ctx);
  CHECK_GE(in_data.Size(), M)
    << "in_data.Size() < M !";
  int32_t Index_size = static_cast<int32_t>(floor(static_cast<double>(M-2*4)*32/33)/4);
  int32_t Gq_size = static_cast<int32_t>(ceil(static_cast<double>(Index_size)/8));
  CHECK_EQ(Index_size*4+Gq_size+2*4,M)  // could be remove later after verifing....
    << "2*4 + Index_size*4 + Gq_size != M";
  int32_t* Index = reinterpret_cast<int32_t*>(in_uint8_t+8);
  uint8_t *Gq = in_uint8_t + (2*4 + Index_size*4);
  float *G = out_float;
  thrust::for_each(
    policy,
    thrust::counting_iterator<int32_t>(0),
    thrust::counting_iterator<int32_t>(Gq_size),
    generate_out(
      Gq,
      Index,
      G,
      Index_size,
      v
    )
  );
}

}
}
#endif