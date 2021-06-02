// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/execution_provider.h"
#include "core/providers/webnn/webnn_provider_factory.h"

namespace onnxruntime {
namespace webnn {
class Model;
}

class WebNNExecutionProvider : public IExecutionProvider {
 public:
  WebNNExecutionProvider(uint32_t webnn_flags);
  virtual ~WebNNExecutionProvider();

  std::vector<std::unique_ptr<ComputeCapability>>
  GetCapability(const onnxruntime::GraphViewer& graph_viewer,
                const std::vector<const KernelRegistry*>& /*kernel_registries*/) const override;

  // We implement the Compile that takes FusedNodeAndGraph instances
  FusionStyle GetFusionStyle() const override { return FusionStyle::FilteredGraphViewer; }

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
  common::Status Compile(const std::vector<FusedNodeAndGraph>& fused_nodes,
                         std::vector<NodeComputeInfo>& node_compute_funcs) override;
#endif

  std::shared_ptr<KernelRegistry> GetKernelRegistry() const override;

 private:
  // The bit flags which define bool options for WEBNN EP, bits are defined as
  // WEBNNFlags in include/onnxruntime/core/providers/webnn/webnn_provider_factory.h
  const uint32_t webnn_flags_;

  std::unordered_map<std::string, std::unique_ptr<onnxruntime::webnn::Model>> models_;
};
}  // namespace onnxruntime
