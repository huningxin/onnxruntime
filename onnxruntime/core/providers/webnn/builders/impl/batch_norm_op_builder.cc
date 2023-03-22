// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/builders/helper.h"
#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"
#include "builder_utils.h"

namespace onnxruntime {
namespace webnn {

class BatchNormalizationOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  bool IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                         const logging::Logger& logger) const override;

  // BatchNormalization opset 6- has unsupported attributes
  int GetMinSupportedOpSet(const Node& /* node */) const override { return 7; }
};

// Add operator related

Status BatchNormalizationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                          const Node& node,
                                                          const logging::Logger& /* logger */) const {
  const auto& input_defs = node.InputDefs();
  emscripten::val input = model_builder.GetOperand(input_defs[0]->Name());
  emscripten::val options = emscripten::val::object();
  options.set("scale", model_builder.GetOperand(input_defs[1]->Name()));
  options.set("bias", model_builder.GetOperand(input_defs[2]->Name()));

  emscripten::val mean = model_builder.GetOperand(input_defs[3]->Name());
  emscripten::val variance = model_builder.GetOperand(input_defs[4]->Name());
  NodeAttrHelper helper(node);
  options.set("epsilon", helper.Get("epsilon", 1e-5f));
  options.set("activation", model_builder.FindActivation(node, *node.OutputDefs()[0]));

  emscripten::val output = model_builder.GetBuilder().call<emscripten::val>("batchNormalization", input,
                                                                            mean, variance, options);
  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));

  return Status::OK();
}

// Operator support related

bool BatchNormalizationOpBuilder::IsOpSupportedImpl(const InitializedTensorSet& initializers, const Node& node,
                                                    const logging::Logger& logger) const {
  if (node.OutputDefs().size() != 1) {
    LOGS(logger, VERBOSE) << "Your onnx model may be in training mode, please export "
                             "it in test mode.";
    return false;
  }

  const auto& input_defs = node.InputDefs();
  std::vector<int64_t> input_shape;
  if (!GetShape(*input_defs[0], input_shape, logger))
    return false;

  const auto input_size = input_shape.size();
  // TODO, support 1d batch normalization (input is 3d)
  // To map 1d input {N,C,H} to 2d {N,C,H,1} first and then squeeze back after
  if (input_size != 4) {
    LOGS(logger, VERBOSE) << "BN only support 4d shape for now, input is "
                          << input_size << "d shape";
    return false;
  }

  NodeAttrHelper helper(node);
  const auto spatial = helper.Get("spatial", 1);
  if (spatial != 1) {
    LOGS(logger, VERBOSE) << "Non-spatial BN is not supported";
    return false;
  }

  const auto& scale_name = input_defs[1]->Name();
  const auto& b_name = input_defs[2]->Name();
  const auto& mean_name = input_defs[3]->Name();
  const auto& var_name = input_defs[4]->Name();
  if (!Contains(initializers, scale_name)) {
    LOGS(logger, VERBOSE) << "Scale of BN must be a constant initializer";
    return false;
  }
  if (!Contains(initializers, b_name)) {
    LOGS(logger, VERBOSE) << "B of BN must be a constant initializer";
    return false;
  }
  if (!Contains(initializers, mean_name)) {
    LOGS(logger, VERBOSE) << "Mean of BN must be a constant initializer";
    return false;
  }
  if (!Contains(initializers, var_name)) {
    LOGS(logger, VERBOSE) << "Var of BN must be a constant initializer";
    return false;
  }

  return true;
}

void CreateBatchNormalizationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.builders.push_back(std::make_unique<BatchNormalizationOpBuilder>());
  op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
}

}  // namespace webnn
}  // namespace onnxruntime
