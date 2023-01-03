// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/graph/graph_viewer.h>

#include "model.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/val.h>
#endif

namespace onnxruntime {
namespace webnn {

class IOpBuilder;

class ModelBuilder {
 public:
 ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
              const emscripten::val& context, const emscripten::val& builder);
  ~ModelBuilder() = default;

  Status Compile(std::unique_ptr<Model>& model) ORT_MUST_USE_RESULT;

  // Accessors for members
  const GraphViewer& GetGraphViewer() const { return graph_viewer_; }
  const InitializedTensorSet& GetInitializerTensors() const { return graph_viewer_.GetAllInitializedTensors(); }


  const emscripten::val& GetBuilder() const { return wnn_builder_; }
  const emscripten::val& GetContext() const { return wnn_context_; }
  const emscripten::val& GetOperand(const std::string& name) const { return wnn_operands_.at(name); }
  void AddOperand(const std::string& name, const emscripten::val& operand);

  // Find if an output has a fuseable activation (e.g., Relu)
  emscripten::val FindActivation(const Node& node, const NodeArg& output);

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

  emscripten::val wnn_context_ = emscripten::val::object();
  emscripten::val wnn_builder_ = emscripten::val::object();
  std::vector<std::vector<uint8_t>> unpacked_tensors_;
  std::unordered_map<std::string, emscripten::val> wnn_operands_;
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
  std::unordered_map<NodeIndex, emscripten::val> activation_nodes_;

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
