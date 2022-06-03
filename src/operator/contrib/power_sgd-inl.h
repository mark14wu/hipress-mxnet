#ifndef MXNET_OPERATOR_CONTRIB_POWER_SGD_INL_H
#define MXNET_OPERATOR_CONTRIB_POWER_SGD_INL_H

#include "../operator_common.h"
#include <thrust/iterator/counting_iterator.h>  // counting_iterator
#include "../linalg.h"
#include "get_policy-inl.h"
#include "../tensor/matrix_op-inl.h"

namespace mxnet {
namespace op {

struct power_sgd_param: public dmlc::Parameter<power_sgd_param> {
  DMLC_DECLARE_PARAMETER(power_sgd_param) {
  };
};

inline bool power_sgd_shape(
  const nnvm::NodeAttrs& attrs,
  mxnet::ShapeVector* in_attrs,
  mxnet::ShapeVector* out_attrs
  ) {
  return true;
}

inline bool power_sgd_type(
  const nnvm::NodeAttrs& attrs,
  std::vector<int>* in_attrs,
  std::vector<int>* out_attrs
  ) {
  return true;
}

template<typename xpu, typename policy_t>
void power_sgd_encode1(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  policy_t policy = get_policy<policy_t>::get(ctx);
  Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& in_grad = inputs[0];
  const TBlob& in_q = inputs[1];
  const TBlob& in_residual = inputs[2];
  const TBlob& in_matrix = inputs[3];
  const TBlob& out_p = outputs[0];

  float* G = to_array(in_grad, float, float);
  // float* Q = to_array(in_q, float, float);
  float* R = to_array(in_residual, float, float);
  float* M_float = to_array(in_matrix, float, float);
  // float* P = to_array(out_p, float, float);
  mshadow::Tensor<xpu, 2, float> M = in_matrix.get<xpu, 2, float>(s);
  mshadow::Tensor<xpu, 2, float> P = out_p.get<xpu, 2, float>(s);
  mshadow::Tensor<xpu, 2, float> Q = in_q.get<xpu, 2, float>(s);

  int32_t N = in_grad.Size();

  thrust::transform(
    policy,
    G,
    G + N,
    R,
    M_float,
    thrust::plus<float>()
  );
  // zt.record("M = Grad + Residual");

  linalg_gemm(M, Q, P, false, false, s);
  // zt.record("P = M * Q");
}

template<typename xpu, typename policy_t>
void power_sgd_encode2(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();
  auto dev_mask = ctx.run_ctx.ctx.dev_mask();
  auto dev_id = ctx.run_ctx.ctx.dev_id;

  const TBlob& in_p = inputs[0];
  const TBlob& in_matrix = inputs[1];
  const TBlob& out_q = outputs[0];

  mxnet::TShape P_shape = in_p.shape_;
  mxnet::TShape P_t_shape = mxnet::TShape({P_shape[1], P_shape[0]});

  mshadow::Tensor<xpu, 2, float> P_t_mem = 
  ctx.requested[0].get_space_typed<xpu, 2, float>(
    mshadow::Shape2(P_shape[1], P_shape[0]),
    s
  );

  TBlob in_p_t(reinterpret_cast<float*>(P_t_mem.dptr_), P_t_shape, dev_mask, dev_id);
  
  mshadow::Tensor<xpu, 2, float> P = in_p.get<xpu, 2, float>(s);
  mshadow::Tensor<xpu, 2, float> P_T = in_p_t.get<xpu, 2, float>(s);
  mshadow::Tensor<xpu, 2, float> M = in_matrix.get<xpu, 2, float>(s);
  mshadow::Tensor<xpu, 2, float> Q = out_q.get<xpu, 2, float>(s);

  // QR Decomposition
  TransposeImpl<xpu>(ctx.run_ctx, in_p, in_p_t, mxnet::TShape({1, 0}));
  int ws_size(linalg_gelqf_workspace_query(P_T, s));
  mshadow::Tensor<xpu, 1, float> work = ctx.requested[0].get_space_typed<xpu, 1, float>(mshadow::Shape1(ws_size), s);
  linalg_gelqf(P_T, work, s);
  linalg_orglq(P_T, work, s);
  TransposeImpl<xpu>(ctx.run_ctx, in_p_t, in_p, mxnet::TShape({1, 0}));
  // zt.record("P = Orthogonal(P)");

  linalg_gemm(M, P, Q, true, false, s);
  // zt.record("Q = M^T * P");
}

template<typename xpu, typename policy_t>
void power_sgd_decode(const nnvm::NodeAttrs& attrs,
                        const OpContext& ctx,
                        const std::vector<TBlob>& inputs,
                        const std::vector<OpReqType>& req,
                        const std::vector<TBlob>& outputs) {

  policy_t policy = get_policy<policy_t>::get(ctx);
  mshadow::Stream<xpu> *s = ctx.get_stream<xpu>();

  const TBlob& in_grad = inputs[0];
  const TBlob& in_q = inputs[1];
  const TBlob& in_residual = inputs[2];
  const TBlob& in_matrix = inputs[3];
  const TBlob& in_p = inputs[4];

  float* G = to_array(in_grad, float, float);
  // float* Q = to_array(in_q, float, float);
  float* R = to_array(in_residual, float, float);
  float* M_float = to_array(in_matrix, float, float);
  mshadow::Tensor<xpu, 2, float> M = in_matrix.get<xpu, 2, float>(s);
  mshadow::Tensor<xpu, 2, float> P = in_p.get<xpu, 2, float>(s);
  mshadow::Tensor<xpu, 2, float> Q = in_q.get<xpu, 2, float>(s);
  // float* P = to_array(in_p, float, float);

  int32_t N = in_grad.Size();

  linalg_gemm(P, Q, M, false, true, s);
  // zt.record("M = P * Q^T");

  thrust::transform(
    policy,
    G,
    G + N,
    M_float,
    R,
    thrust::minus<float>()
  );
  // zt.record("Residual = Grad - M");

  thrust::copy(
    policy,
    M_float,
    M_float + N,
    G
  );
  // zt.record("G = M[:N]");
}

}
}
#endif