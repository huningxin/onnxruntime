// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/builders/model_builder.h"
#include "core/providers/webnn/builders/op_builder_factory.h"

#include "base_op_builder.h"

#include <webnn/webnn_cpp.h>

namespace onnxruntime {
namespace webnn {

class ActivationOpBuilder : public BaseOpBuilder {
  // Add operator related
 private:
  Status AddToModelBuilderImpl(ModelBuilder& model_builder, const Node& node,
                               const logging::Logger& logger) const override ORT_MUST_USE_RESULT;

  // Operator support related
 private:
  int GetMinSupportedOpSet(const Node& node) const override;
};

// Add operator related

Status ActivationOpBuilder::AddToModelBuilderImpl(ModelBuilder& model_builder,
                                                  const Node& node,
                                                  const logging::Logger& /* logger */) const {
  const auto& op_type(node.OpType());
  ::ml::Operand input = model_builder.GetOperand(node.InputDefs()[0]->Name());
  ::ml::Operand output;
  if (op_type == "Relu") {
    output = model_builder.GetBuilder().Relu(input);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "ActivationOpBuilder::AddToModelBuilderImpl, unknown op: ", op_type);
  }

  model_builder.AddOperand(node.OutputDefs()[0]->Name(), std::move(output));
  return Status::OK();
}

// Operator support related

int ActivationOpBuilder::GetMinSupportedOpSet(const Node& /* node */) const {
  // All ops opset 5- uses consumed_inputs attribute which is not supported for now
  return 6;
}

void CreateActivationOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  if (op_registrations.op_builder_map.find(op_type) != op_registrations.op_builder_map.cend())
    return;

  static std::vector<std::string> op_types =
      {
          "Relu",
      };

  op_registrations.builders.push_back(std::make_unique<ActivationOpBuilder>());
  for (const auto& op_type : op_types) {
    op_registrations.op_builder_map.emplace(op_type, op_registrations.builders.back().get());
  }
}

}  // namespace webnn
}  // namespace onnxruntime
