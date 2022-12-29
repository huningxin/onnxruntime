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

Model::Model(const ::wnn::Context& context, const ::wnn::Graph& graph, const logging::Logger& logger,
             uint32_t device_flags, uint32_t power_flags)
    : context_(context),
      graph_(graph),
      logger_(logger),
      device_flags_(device_flags),
      power_flags_(power_flags) {
}

Model::~Model() {}

Status Model::Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                      const std::unordered_map<std::string, OnnxTensorData>& outputs) {
  ::wnn::NamedInputs named_inputs = ::wnn::CreateNamedInputs();
  for (const auto& input: inputs) {
    const std::string& name = input.first;
    const struct OnnxTensorData tensor = input.second;
    wnn_inputs_[name].resource.arrayBufferView.buffer = tensor.buffer;
    if (tensor.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph has unsupported type, name: ",
                             name, " type: ", tensor.tensor_info.data_type);
    }
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    wnn_inputs_[name].resource.arrayBufferView.byteLength = SafeInt<uint32_t>(num_elements * sizeof(float));
    named_inputs.Set(name.c_str(), &wnn_inputs_[name]);
  }
  ::wnn::NamedOutputs named_outputs = ::wnn::CreateNamedOutputs();
  for (const auto& output: outputs) {
    const std::string& name = output.first;
    const struct OnnxTensorData tensor = output.second;
    wnn_outputs_[name].arrayBufferView.buffer = tensor.buffer;
    if (tensor.tensor_info.data_type != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The input of graph has unsupported type, name: ",
                             name, " type: ", tensor.tensor_info.data_type);
    }
    auto num_elements = SafeInt<size_t>(Product(tensor.tensor_info.shape));
    wnn_outputs_[name].arrayBufferView.byteLength = num_elements * sizeof(float);
    named_outputs.Set(name.c_str(), &wnn_outputs_[name]);
  }

  context_.ComputeSync(graph_, named_inputs, named_outputs);

  return Status::OK();
}

bool Model::IsScalarOutput(const std::string& output_name) const {
  return Contains(scalar_outputs_, output_name);
}

const OnnxTensorInfo& Model::GetInputOutputInfo(const std::string& name) const {
  return input_output_info_.at(name);
}

void Model::SetInputMap(std::unordered_map<std::string, size_t>&& input_map) {
  input_map_ = std::move(input_map);
}

void Model::SetOutputMap(std::unordered_map<std::string, size_t>&& output_map) {
  output_map_ = std::move(output_map);
}

size_t Model::GetMappedInputIdx(const std::string& name) const {
  return input_map_.at(name);
}

size_t Model::GetMappedOutputIdx(const std::string& name) const {
  return output_map_.at(name);
}

}  // namespace webnn
}  // namespace onnxruntime