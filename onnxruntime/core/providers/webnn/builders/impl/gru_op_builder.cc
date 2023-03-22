// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class GruOpBuilder : public BaseOpBuilder {
  // Add operator related
 public:
  void AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const override;

 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& /* initializers */, const Node& /* node */,
                         const logging::Logger& /* logger */) const override;

  // Gru opset 7- has different attributes and algroithm of Equations.
  // We only support Gru opset 7+ here
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 7; }
};

// Add operator related

void GruOpBuilder::AddInitializersToSkip(ModelBuilder& model_builder, const Node& node) const {
  // Skip sequence_lens
  if (node.InputDefs().size() > 4) {
    model_builder.AddInitializerToSkip(node.InputDefs()[4]->Name());
    model_builder.AddInputToSkip(node.InputDefs()[4]->Name());
  }
}

Status GruOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                                           const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();
  const auto& initializers(model_builder.GetInitializerTensors());
  NodeAttrHelper helper(node);
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val weight = model_builder.GetOperand(input_defs[1]->Name());
  emscripten::val recurrentWeight = model_builder.GetOperand(input_defs[2]->Name());
  emscripten::val options = emscripten::val::object();
  emscripten::val output = emscripten::val::object();
  std::vector<int64_t> input_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[0], input_shape, logger), "Cannot get shape");

  // steps, input_defs[4] is sequence_lens, indicates steps in WebNN.
  auto steps = input_shape[0];
  if (input_defs.size() > 4 && Contains(initializers, input_defs[4]->Name())) {
    const auto& sequenceLens_tensor = *initializers.at(input_defs[4]->Name());
    const int32_t* raw_sequenceLens = sequenceLens_tensor.int32_data().empty()
                                          ? reinterpret_cast<const int32_t*>(sequenceLens_tensor.raw_data().data())
                                          : sequenceLens_tensor.int32_data().data();
    steps = raw_sequenceLens[0];
  }
  // hiddenSize
  std::vector<int64_t> recurrentWeight_shape;
  ORT_RETURN_IF_NOT(GetShape(*input_defs[2], recurrentWeight_shape, logger), "Cannot get shape");
  const auto hiddenSize = helper.Get("hidden_size", recurrentWeight_shape[2]);
  // Add bias, recurrentBias if present
  if (input_defs.size() > 3 /*&& Contains(initializers, input_defs[3]->Name())*/) {
    // ONNX's bias should be splited in half for WebNN Gru's bias and recurrentBias options
    emscripten::val biasOperand = model_builder.GetOperand(input_defs[3]->Name());
    emscripten::val splitOptions = emscripten::val::object();
    splitOptions.set("axis", 1);
    emscripten::val splittedOperands = model_builder.GetBuilder().call<emscripten::val>("split", biasOperand, 2,
                                                                                        splitOptions);
    options.set("bias", splittedOperands[0]);
    options.set("recurrentBias", splittedOperands[1]);
  }
  // Add initialHiddenState if present
  if (input_defs.size() > 5) {
    emscripten::val initialHiddenState = model_builder.GetOperand(input_defs[5]->Name());
    options.set("initialHiddenState", initialHiddenState);
  }

  // resetAfter, ONNX's linear_before_reset equals to WebNN's resetAfter
  const auto linear_before_reset = helper.Get("linear_before_reset", static_cast<int32_t>(0));
  options.set("resetAfter", linear_before_reset == 0 ? false : true);
  // Set returnSequence to true as ONNX's Gru has two outputs.
  options.set("returnSequence", true);
  // direction, ONNX's direction: forward (default), reverse, or bidirectional.
  const auto direction = helper.Get("direction", "forward");
  if (direction == "reverse") {
    options.set("direction", emscripten::val("backward"));
  } else if (direction == "bidirectional") {
    options.set("direction", emscripten::val("both"));
  } else {
    options.set("direction", emscripten::val("forward"));
  }
  // layout
  const auto layout = helper.Get("layout", static_cast<int32_t>(0));
  if (layout == 1) {
    options.set("layout", emscripten::val("rzn"));
  }
  // Get ONNX's activations attribute, default is {"Sigmoid", "Tanh"}
  // Only support "Relu", "Tanh", "Sigmoid" at present
  emscripten::val activationOperators = emscripten::val::array();
  const auto activations = helper.Get("activations", std::vector<std::string>{"Sigmoid", "Tanh"});
  for (auto activation : activations) {
    emscripten::val activationOperator = emscripten::val::object();
    if (activation == "Relu") {
      activationOperator = model_builder.GetBuilder().call<emscripten::val>("relu");
    } else if (activation == "Sigmoid") {
      activationOperator = model_builder.GetBuilder().call<emscripten::val>("sigmoid");
    } else if (activation == "Tanh") {
      activationOperator = model_builder.GetBuilder().call<emscripten::val>("tanh");
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GruOpBuilder::AddToModelBuilderImpl, unsupported activation: ", activation);
    }
    activationOperators.call<emscripten::val>("push", activationOperator);
  }
  options.set("activations", activationOperators);
  output = model_builder.GetBuilder().call<emscripten::val>("gru", input, weight, recurrentWeight, (long)steps,
                                                            (long)hiddenSize, options);
  // Reverse outputs and insert to model builder
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output[1]));
  model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(output[0]));
  return Status::OK();
}

// Operator support related

bool GruOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                     const logging::Logger& logger) const {
  const auto& input_defs = node.InputDefs();

  // values in sequence_lens should be the same
  if (input_defs.size() > 4 && Contains(initializers, input_defs[4]->Name())) {
    const auto& sequenceLens_tensor = *initializers.at(input_defs[4]->Name());
    const int32_t* raw_sequenceLens = sequenceLens_tensor.int32_data().empty()
                                          ? reinterpret_cast<const int32_t*>(sequenceLens_tensor.raw_data().data())
                                          : sequenceLens_tensor.int32_data().data();
    auto sequenceLens_size = SafeInt<size_t>(Product(sequenceLens_tensor.dims()));
    if (sequenceLens_size > 1) {
      const int32_t sequenceLen0 = raw_sequenceLens[0];
      for (unsigned i = 1; i < sequenceLens_size; i++) {
        if (raw_sequenceLens[i] != sequenceLen0) {
          LOGS(logger, VERBOSE) << "Gru: values in sequence_lens should be the same.";
          return false;
        }
      }
    }
  }

  {  // check attributes
    NodeAttrHelper helper(node);
    // Return error if following unsupported attributes present
    const char* unsupported_attrs[3] = {"activation_alpha", "activation_beta", "clip"};
    for (int i = 0; i < 3; i++) {
      if (helper.HasAttr(unsupported_attrs[i])) {
        LOGS(logger, VERBOSE) << "Gru unsupported attribute:" << unsupported_attrs[i];
        return false;
      }
    }

    // Return error if size of activations is 4 but direction != bidirectional
    const auto activations = helper.Get("activations", std::vector<std::string>{"Sigmod", "Tanh"});
    const auto direction = helper.Get("direction", "forward");
    if (activations.size() == 4 && direction != "bidirectional") {
      LOGS(logger, VERBOSE) << "Gru: a list of 4 activation functions must be bidirectional direction.";
      return false;
    }
  }

  return true;
}

void CreateGruOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<GruOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
