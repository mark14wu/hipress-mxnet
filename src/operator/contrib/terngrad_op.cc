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
 * \file quadratic_op.cc
 * \brief CPU Implementation of quadratic op
 */
#include "./terngrad_op-inl.h"
#include <thrust/execution_policy.h>  //thrust::device
#include <thrust/system/omp/execution_policy.h>

namespace mxnet {
namespace op {

DMLC_REGISTER_PARAMETER(TernGradParam);
DMLC_REGISTER_PARAMETER(TernGradRParam);

NNVM_REGISTER_OP(_contrib_terngrad)
.set_attr_parser(ParamParser<TernGradParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TernGradOpShape)
.set_attr<nnvm::FInferType>("FInferType", TernGradOpType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(TernGradParam::__FIELDS__())
;



NNVM_REGISTER_OP(_contrib_terngradr)
.set_attr_parser(ParamParser<TernGradRParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TernGradROpShape)
.set_attr<nnvm::FInferType>("FInferType", TernGradROpType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(TernGradRParam::__FIELDS__())
;

NNVM_REGISTER_OP(_contrib_terngradr)
.set_attr<FCompute>("FCompute<cpu>", TernGradROpForward_gpu<cpu,thrust::detail::host_t>)
;

NNVM_REGISTER_OP(_contrib_terngrad)
.set_attr<FCompute>("FCompute<cpu>", TernGradOpForward_gpu<cpu,thrust::detail::host_t>)
;


NNVM_REGISTER_OP(_contrib_terngrad_omp)
.set_attr_parser(ParamParser<TernGradParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TernGradOpShape)
.set_attr<nnvm::FInferType>("FInferType", TernGradOpType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(TernGradParam::__FIELDS__())
;



NNVM_REGISTER_OP(_contrib_terngradr_omp)
.set_attr_parser(ParamParser<TernGradRParam>)
.set_num_inputs(1)
.set_num_outputs(1)
.set_attr<nnvm::FListInputNames>("FListInputNames",
  [](const NodeAttrs& attrs) {
    return std::vector<std::string>{"data"};
  })
.set_attr<mxnet::FInferShape>("FInferShape", TernGradROpShape)
.set_attr<nnvm::FInferType>("FInferType", TernGradROpType)
.set_attr<nnvm::FInplaceOption>("FInplaceOption",
  [](const NodeAttrs& attrs) {
    return std::vector<std::pair<int, int> >{{0, 0}};
  })
.add_argument("data", "NDArray-or-Symbol", "Input ndarray")
.add_arguments(TernGradRParam::__FIELDS__())
;

NNVM_REGISTER_OP(_contrib_terngradr_omp)
.set_attr<FCompute>("FCompute<cpu>", TernGradROpForward_gpu<cpu,thrust::system::omp::detail::par_t>)
;

NNVM_REGISTER_OP(_contrib_terngrad_omp)
.set_attr<FCompute>("FCompute<cpu>", TernGradOpForward_gpu<cpu,thrust::system::omp::detail::par_t>)
;

}  // namespace op
}  // namespace mxnet
