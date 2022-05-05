#include <mxnet-cpp/MxNetCpp.h>
#include <vector>
#include <unistd.h>
#include <stdio.h>

typedef ::mxnet::cpp::Operator Operator;
typedef ::mxnet::cpp::Symbol Symbol;

int main () {
	std::vector<float> result;
	auto ctxx = ::mxnet::cpp::Context::cpu();
  std::map<std::string, ::mxnet::cpp::NDArray> args_map;
  args_map["data"] = ::mxnet::cpp::NDArray(::mxnet::cpp::Shape(10), ctxx);
  auto input_data = Symbol::Variable("data");
  auto terngrad = Operator("_contrib_terngrad").SetParam("bitwidth", 2).SetParam("random", 0).SetInput("data", input_data).CreateSymbol("terngrad");
  auto *exec = terngrad.SimpleBind(ctxx, args_map);
  exec->Forward(false);

  float right[13];
  float left[13];
  float output[13];
  for (int i = 0; i < 13; i++) {
    right[i] = 1;
    left[i] = 3;
    output[i] = 2;
  }

  //NDArray(const mx_float *data, const Shape &shape, const Context &context);
  auto lhs = Symbol::Variable("lhs");
  auto rhs = Symbol::Variable("rhs");
  auto add = elemwise_add("add", lhs, rhs);
  std::map<std::string, ::mxnet::cpp::NDArray> args_map_add;
  args_map_add["lhs"] = ::mxnet::cpp::NDArray(left, ::mxnet::cpp::Shape(13), ctxx);
  args_map_add["rhs"] = ::mxnet::cpp::NDArray(right, ::mxnet::cpp::Shape(13), ctxx);
  auto *exec_add = add.SimpleBind(ctxx, args_map_add);
  exec_add->Forward(false);

  auto l = Symbol::Variable("lhs");
  auto r = Symbol::Variable("rhs");
  auto add1 = elemwise_add("add1", l, r);
  args_map_add["lhs"] = exec_add->outputs[0];
  args_map_add["rhs"] = ::mxnet::cpp::NDArray(output, ::mxnet::cpp::Shape(13), ctxx);
  auto *exec_add1 = add1.SimpleBind(ctxx, args_map_add);
  exec_add1->Forward(false);

  exec_add1->outputs[0].WaitToRead();
  auto encoder_result = exec_add1->outputs[0].GetData();
  for (int i = 0; i < exec_add1->outputs[0].Size(); i++) {
  	float t = encoder_result[i];
    result.push_back(t);
  }
  for (auto elem : result) {
  	std::cout << elem << std::endl;
  }
  return 0;
}