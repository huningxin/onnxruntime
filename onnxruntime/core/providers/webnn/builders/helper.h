// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <core/common/status.h>
#include <core/graph/basic_types.h>
#include "core/providers/common.h"

#include <emscripten.h>
#include <emscripten/val.h>

namespace onnxruntime {

class GraphViewer;
class NodeArg;

namespace logging {
class Logger;
}

namespace webnn {

bool GetShape(const NodeArg& node_arg, std::vector<int64_t>& shape, const logging::Logger& logger);

bool IsInputSupported(const NodeArg& node_arg, const std::string& parent_name, const logging::Logger& logger);

// Get a list of groups of supported nodes, each group represents a subgraph supported by WebNN EP
std::vector<std::vector<NodeIndex>> GetSupportedNodes(const GraphViewer& graph_viewer,
                                                      const emscripten::val& wnn_builder_,
                                                      const logging::Logger& logger);

inline std::unordered_map<std::string, std::vector<std::string>> op_dependency = {
    {"GRU", {"Split"}}
};  // namespace webnn

inline std::unordered_map<std::string, std::string> op_map = {
    {"Add", "add"},
    {"Relu", "relu"},
    {"LeakyRelu", "leakyRelu"},
    {"Sigmoid", "sigmoid"},
    {"Tanh", "tanh"},
    {"BatchNormalization", "batchNormalization"},
    {"Clip", "clamp"},
    {"Conv", "conv2d"},
    {"ConvTranspose", "convTranspose2d"},
    {"Concat", "concat"},
    {"Gemm", "gemm"},
    {"MatMul", "matmul"},
    {"GRU", "gru"},
    {"GlobalAveragePool", "averagePool2d"},
    {"GlobalMaxPool", "maxPool2d"},
    {"AveragePool", "averagePool2d"},
    {"MaxPool", "maxPool2d"},
    {"Reshape", "reshape"},
    {"Resize", "resample2d"},
    {"Transpose", "transpose"}};

inline bool CheckSingleOp(const std::string& op_type, const emscripten::val& wnn_builder_) {
  return op_map.find(op_type) != op_map.end() && wnn_builder_[op_map[op_type]].as<bool>();
}

inline bool CheckDependency(const std::string& op_type, const emscripten::val& wnn_builder_) {
  if (!CheckSingleOp(op_type, wnn_builder_)) {
    return false;
  }
  if (Contains(op_dependency, op_type)) {
    for (auto& op : op_dependency[op_type]) {
      if (!CheckSingleOp(op, wnn_builder_)) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace onnxruntime
}  // namespace onnxruntime
