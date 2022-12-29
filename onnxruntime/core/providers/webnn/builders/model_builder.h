// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/graph/graph_viewer.h>

#include <webnn/webnn_cpp.h>

#include "model.h"

namespace onnxruntime {
namespace webnn {

class IOpBuilder;

class ModelBuilder {
 public:
  ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger, uint32_t device_flags, uint32_t power_flags);
  ~ModelBuilder() = default;

  Status Compile(std::unique_ptr<Model>& model) ORT_MUST_USE_RESULT;

  // Accessors for members
  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  const InitializedTensorSet& GetInitializerTensors() const { return graph_viewer_.GetAllInitializedTensors(); }

  const ::wnn::GraphBuilder& GetBuilder() const { return builder_; }
  const ::wnn::Context& GetContext() const { return context_; }
  const ::wnn::Operand& GetOperand(const std::string& name) const { return operands_.at(name); }
  void AddOperand(const std::string& name, const ::wnn::Operand& operand);

  // Find if an output has a fuseable activation (e.g., Relu)
  ::wnn::FusionOperator FindActivation(const Node& node, const NodeArg& output);

  const std::unordered_set<std::string>&
  GetFusedActivations() const { return fused_activations_; }

  // The initializer will be processed separately, skip it as an initializer
  void AddInitializerToSkip(const std::string& tensor_name);

  // There are some input which will not be used, add it to a list which will not
  // be added to CoreML model, since CoreML does not like input unused
  void AddInputToSkip(const std::string& input_name);

  std::string GetUniqueName(const std::string& base_name);

 private:
  const GraphViewer& graph_viewer_;
  const logging::Logger& logger_;
  uint32_t device_flags_;
  uint32_t power_flags_;

  ::wnn::GraphBuilder builder_;
  ::wnn::Context context_;
  std::vector<std::vector<uint8_t>> unpacked_tensors_;
  std::unordered_map<std::string, ::wnn::Operand> operands_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;

  std::unordered_set<std::string> scalar_outputs_;
  std::unordered_map<std::string, OnnxTensorInfo> input_output_info_;

  std::unordered_set<std::string> skipped_initializers_;
  std::unordered_set<std::string> skipped_inputs_;

  std::unordered_set<std::string> fused_activations_;

  uint32_t name_token_{0};
  std::unordered_set<std::string> unique_names_;

  // All activation nodes (e.g., Relu) as a map <NodeIndex, FusionOperator>
  std::unordered_map<NodeIndex, ::wnn::FusionOperator> activation_nodes_;

  // Convert the onnx model to WebNN operands
  Status Initialize() ORT_MUST_USE_RESULT;

  void PreprocessInitializers();
  // Preprocess all the activation nodes (e.g., Relu) for easy query later
  void PreprocessActivations();

  // Copy and process all the initializers to WebNN constants
  Status RegisterInitializers() ORT_MUST_USE_RESULT;

  Status AddOperations() ORT_MUST_USE_RESULT;
  Status RegisterModelInputs() ORT_MUST_USE_RESULT;
  Status RegisterModelOutputs() ORT_MUST_USE_RESULT;
  Status RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) ORT_MUST_USE_RESULT;

  // Record the onnx scalar output names
  void AddScalarOutput(const std::string& output_name);

  static const IOpBuilder* GetOpBuilder(const Node& node);
};

}  // namespace webnn
}  // namespace onnxruntime
