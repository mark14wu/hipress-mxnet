#ifndef _SRC_OPERATOR_CONTRIB_GET_POLICY_INL_H
#define _SRC_OPERATOR_CONTRIB_GET_POLICY_INL_H
#include <mxnet/operator_util.h>
#include "../mshadow_op.h"
#include "../mxnet_op.h"

#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/cuda/execution_policy.h>
#include <thrust/system/omp/execution_policy.h> //

#define to_array(obj,to_type,from_type) ((to_type*)(obj.dptr<from_type>()))


namespace mxnet {
namespace op {


template <typename T>
struct get_policy{};

template<>
struct get_policy<thrust::system::omp::detail::par_t>{
  inline static thrust::system::omp::detail::par_t get(
    const OpContext& ctx
  ){
    return thrust::omp::par;
  }
  inline static void memcpyOut(
    void* dst,
    void* src,
    uint32_t len,
    const OpContext& ctx
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyIn(
    void* dst,
    void* src,
    uint32_t len,
    const OpContext& ctx
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyOutSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyInSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void streamSynchronize(
    const OpContext& ctx
  ){
    return;
  }
};

template<>
struct get_policy<thrust::detail::host_t>{
  inline static thrust::detail::host_t get(
    const OpContext& ctx
  ){
    return thrust::host;
  }
  inline static void memcpyOut(
    void* dst,
    void* src,
    uint32_t len,
    const OpContext& ctx
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyIn(
    void* dst,
    void* src,
    uint32_t len,
    const OpContext& ctx
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyOutSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void memcpyInSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    memcpy(dst,src,len);
  }
  inline static void streamSynchronize(
    const OpContext& ctx
  ){
    return;
  }
};

template<>
struct get_policy<thrust::cuda_cub::par_t::stream_attachment_type>{
  inline static thrust::cuda_cub::par_t::stream_attachment_type get(
    const OpContext& ctx
  ){
    auto stream = mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>());
    return thrust::cuda::par.on(stream);
  }
  inline static void memcpyOut(
    void* dst,
    void* src,
    uint32_t len,
    const OpContext& ctx
  ){
    auto stream = mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>());
    cudaMemcpyAsync(dst,src,len,cudaMemcpyDeviceToHost,stream);
  }
  inline static void memcpyIn(
    void* dst,
    void* src,
    uint32_t len,
    const OpContext& ctx
  ){
    auto stream = mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>());
    cudaMemcpyAsync(dst,src,len,cudaMemcpyHostToDevice,stream);
  }  
  inline static void memcpyOutSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    cudaMemcpy(dst,src,len,cudaMemcpyDeviceToHost);
  }
  inline static void memcpyInSync(
    void* dst,
    void* src,
    uint32_t len
  ){
    cudaMemcpy(dst,src,len,cudaMemcpyHostToDevice);
  }
  inline static void streamSynchronize(
    const OpContext& ctx
  ){
    auto stream = mshadow::Stream<gpu>::GetStream(ctx.get_stream<gpu>());
    cudaStreamSynchronize(stream);
  }
};

}   //  op
}   //  mxnet

#endif
