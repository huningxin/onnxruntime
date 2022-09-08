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

  ::wnn::Operand input = model_builder.GetOperand(input_defs[0]->Name());
  ::wnn::Operand weight = model_builder.GetOperand(input_defs[1]->Name());
  ::wnn::Operand recurrentWeight = model_builder.GetOperand(input_defs[2]->Name());
  ::wnn::GruOptions options;
  ::wnn::OperandArray output;

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
  if (input_defs.size() > 3 && Contains(initializers, input_defs[3]->Name())) {
    // ONNX's bias should be splited in half for WebNN Gru's bias and recurrentBias options
    ::wnn::Operand biasOperand = model_builder.GetOperand(input_defs[3]->Name());
    std::vector<uint32_t> splits = {2};
    ::wnn::SplitOptions splitOptions;
    splitOptions.axis = 1;
    ::wnn::OperandArray splittedOperands = model_builder.GetBuilder().Split(biasOperand, splits.data(),
                                                                           static_cast<uint32_t>(splits.size()), &splitOptions);
    options.bias = splittedOperands.Get(0);
    options.recurrentBias = splittedOperands.Get(1);
  }

  // Add initialHiddenState if present
  if (input_defs.size() > 5) {
    ::wnn::Operand initialHiddenState = model_builder.GetOperand(input_defs[5]->Name());
    options.initialHiddenState = initialHiddenState;
  }

  // resetAfter, ONNX's linear_before_reset equals to WebNN's resetAfter
  const auto linear_before_reset = helper.Get("linear_before_reset", static_cast<int32_t>(0));
  options.resetAfter = linear_before_reset == 0 ? false : true;
  // Set returnSequence to true as ONNX's Gru has two outputs.
  options.returnSequence = true;
  // direction, ONNX's direction: forward (default), reverse, or bidirectional.
  const auto direction = helper.Get("direction", "forward");
  if (direction == "reverse") {
    options.direction = ::wnn::RecurrentNetworkDirection::Backward;
  } else if (direction == "bidirectional") {
    options.direction = ::wnn::RecurrentNetworkDirection::Both;
  } else {
    options.direction = ::wnn::RecurrentNetworkDirection::Forward;
  }
  // layout
  const auto layout = helper.Get("layout", static_cast<int32_t>(0));
  if (layout == 1) {
    options.layout = ::wnn::RecurrentNetworkWeightLayout::Rzn;
  }
  // Get ONNX's activations attribute, default is {"Sigmoid", "Tanh"}
  // Only support "Relu", "Tanh", "Sigmoid" at present
  auto activationOperators = ::wnn::CreateOperatorArray();
  const auto activations = helper.Get("activations", std::vector<std::string>{"Sigmoid", "Tanh"});
  for (auto activation : activations) {
    wnn::FusionOperator activationOperator;
    if (activation == "Relu") {
      activationOperator = model_builder.GetBuilder().ReluOperator();
    } else if (activation == "Sigmoid") {
      activationOperator = model_builder.GetBuilder().SigmoidOperator();
    } else if (activation == "Tanh") {
      activationOperator = model_builder.GetBuilder().TanhOperator();
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GruOpBuilder::AddToModelBuilderImpl, unsupported activation: ", activation);
    }
    activationOperators.Set(activationOperator);
  }
  options.activations = activationOperators;

  output = model_builder.GetBuilder().Gru(input, weight, recurrentWeight, SafeInt<int32_t>(steps),
                                          SafeInt<int32_t>(hiddenSize), &options);

  // Reverse outputs and insert to model builder
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output.Get(1)));
  model_builder.AddOperand(node.OutputDefs()[1]->Name(), std::move(output.Get(0)));
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
