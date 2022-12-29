// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/platform/ort_mutex.h"

#include <webnn/webnn_cpp.h>

namespace onnxruntime {
namespace webnn {

struct OnnxTensorInfo {
  const int32_t data_type;  // Uses TensorProto::DataType
  const std::vector<int64_t> shape;
};

struct OnnxTensorData {
  OnnxTensorInfo tensor_info;
  void* buffer{nullptr};
};

class Model {
  friend class ModelBuilder;

 public:
  ~Model();
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Model);

  onnxruntime::common::Status Predict(const std::unordered_map<std::string, OnnxTensorData>& inputs,
                                      const std::unordered_map<std::string, OnnxTensorData>& outputs);

  bool IsScalarOutput(const std::string& output_name) const;

  // Mutex for exclusive lock to this model object
  OrtMutex& GetMutex() { return mutex_; }

  // Input and output names in the onnx model's order
  const std::vector<std::string>& GetInputs() const { return inputs_; }
  void SetInputs(std::vector<std::string>&& inputs) { inputs_ = std::move(inputs); }

  const std::vector<std::string>& GetOutputs() const { return outputs_; }
  void SetOutputs(std::vector<std::string>&& outputs) { outputs_ = std::move(outputs); }

  const OnnxTensorInfo& GetInputOutputInfo(const std::string& name) const;

  // Set the mapping between input/output name and ORT kernel context
  // input/output index, at execution time
  void SetInputMap(std::unordered_map<std::string, size_t>&& input_map);
  void SetOutputMap(std::unordered_map<std::string, size_t>&& output_map);

  // Get the ORT kernel context input/output index with given name
  size_t GetMappedInputIdx(const std::string& name) const;
  size_t GetMappedOutputIdx(const std::string& name) const;

 private:
  ::wnn::Context context_;
  ::wnn::Graph graph_;
  const logging::Logger& logger_;
  uint32_t device_flags_;
  uint32_t power_flags_;

  std::unordered_map<std::string, ::wnn::Input> wnn_inputs_;
  std::unordered_map<std::string, ::wnn::Resource> wnn_outputs_;

  std::unordered_set<std::string> scalar_outputs_;

  std::vector<std::string> inputs_;
  std::vector<std::string> outputs_;

  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;

  std::unordered_map<std::string, size_t> input_map_;
  std::unordered_map<std::string, size_t> output_map_;

  OrtMutex mutex_;

  Model(const ::wnn::Context& context, const ::wnn::Graph& path, const logging::Logger& logger, uint32_t device_flags, uint32_t power_flags);

  void SetInputOutputInfo(std::unordered_map<std::string, OnnxTensorInfo>&& input_output_info) {
    input_output_info_ = std::move(input_output_info);
  }

  void SetScalarOutputs(std::unordered_set<std::string>&& scalar_outputs) {
    scalar_outputs_ = std::move(scalar_outputs);
  }
};

}  // namespace webnn
}  // namespace onnxruntime