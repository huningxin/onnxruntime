// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "core/common/common.h"
#include "core/common/logging/logging.h"
#include "core/common/safeint.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/common.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/webnn_provider_factory.h"
#include "model.h"

namespace onnxruntime {
namespace webnn {

Model::Model(const ::ml::Graph& graph, const logging::Logger& logger, uint32_t flags)
    : graph_(graph), logger_(logger), flags_(flags) {
}

Model::~Model() {}

Status Model::Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                      const std::unordered_map<std::string, OnnxTensorData>& outputs) {
  ::ml::NamedInputs named_inputs = ::ml::CreateNamedInputs();
  for (const auto& input: inputs) {
    const std::string& name = input.first;
    const struct OnnxTensorData tensor = input.second;
    ml_inputs_[name].resource.buffer = tensor.buffer;
    if (tensor.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph has unsupported type, name: ",
                             name, " type: ", tensor.tensor_info.data_type);
    }
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    ml_inputs_[name].resource.byteLength = SafeInt<uint32_t>(num_elements * sizeof(float));
    named_inputs.Set(name.c_str(), &ml_inputs_[name]);
  }
  ::ml::NamedOutputs named_outputs = ::ml::CreateNamedOutputs();
  for (const auto& output: outputs) {
    const std::string& name = output.first;
    const struct OnnxTensorData tensor = output.second;
    ml_outputs_[name].buffer = tensor.buffer;
    if (tensor.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph has unsupported type, name: ",
                             name, " type: ", tensor.tensor_info.data_type);
    }
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    ml_outputs_[name].byteLength = num_elements * sizeof(float);
    named_outputs.Set(name.c_str(), &ml_outputs_[name]);
  }

  ::ml::ComputeGraphStatus status = graph_.Compute(named_inputs, named_outputs);
  if (status != ::ml::ComputeGraphStatus::Success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to compute WebNN graph");
  }

  return Status::OK();
}

bool Model::IsScalarOutput(const std::string& output_name) const {
  return Contains(scalar_outputs_, output_name);
}

const OnnxTensorInfo& Model::GetInputOutputInfo(const std::string& name) const {
  return input_output_info_.at(name);
}

}  // namespace webnn
}  // namespace onnxruntime