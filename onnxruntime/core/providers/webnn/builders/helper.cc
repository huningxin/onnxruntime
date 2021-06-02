// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "helper.h"
#include <core/graph/graph_viewer.h>

#include "core/providers/common.h"
#include "op_builder_factory.h"

namespace onnxruntime {
namespace webnn {

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger) {
  const auto* shape_proto = node_arg.Shape();
  if (!shape_proto) {
    LOGS(logger, WARNING) << "NodeArg [" << node_arg.Name() << "] has no shape info";
    return false;
  }

  // We already checked the shape has no dynamic dimension
  for (const auto& dim : shape_proto->dim()) {
    shape.push_back(dim.dim_value());
  }

  return true;
}

bool IsNodeSupported(const Node& node, const GraphViewer& graph_viewer, const logging::Logger& logger) {
  const auto& op_builders = GetOpBuilders();
  if (Contains(op_builders, node.OpType())) {
    const auto* op_builder = op_builders.at(node.OpType());
    return op_builder->IsOpSupported(graph_viewer.GetAllInitializedTensors(), node, logger);
  } else {
    return false;
  }
}

bool IsInputSupported(const NodeArg& input, const std::string& parent_name, const logging::Logger& logger) {
  const auto& input_name = input.Name();
  const auto* shape_proto = input.Shape();
  // We do not support input with no shape
  if (!shape_proto) {
    LOGS(logger, VERBOSE) << "Input [" << input_name << "] of [" << parent_name
                          << "] has not shape";
    return false;
  }

  for (const auto& dim : shape_proto->dim()) {
    // For now we do not support dynamic shape
    if (!dim.has_dim_value()) {
      LOGS(logger, WARNING) << "Dynamic shape is not supported for now, for input:" << input_name;
      return false;
    }
  }

  return true;
}

std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer, const logging::Logger& logger) {
  std::vector<std::vector<size_t>> supported_node_groups;

  for (const auto* input : graph_viewer.GetInputs()) {
    if (!IsInputSupported(*input, "graph", logger)) {
      return supported_node_groups;
    }
  }

  std::vector<size_t> supported_node_group;
  const auto& node_indices = graph_viewer.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    auto node_idx = node_indices[i];
    const auto* node(graph_viewer.GetNode(node_idx));
    bool supported = IsNodeSupported(*node, graph_viewer, logger);
    LOGS(logger, VERBOSE) << "Operator type: [" << node->OpType()
                          << "] index: [" << node_idx
                          << "] name: [" << node->Name()
                          << "] supported: [" << supported
                          << "]";
    if (supported) {
      supported_node_group.push_back(node_idx);
    } else {
      if (!supported_node_group.empty()) {
        supported_node_groups.push_back(supported_node_group);
        supported_node_group.clear();
      }
    }
  }

  if (!supported_node_group.empty()) {
    supported_node_groups.push_back(supported_node_group);
  }

  return supported_node_groups;
}


}  // namespace webnn
}  // namespace onnxruntime