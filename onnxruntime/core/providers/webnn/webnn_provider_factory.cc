// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/webnn/webnn_provider_factory.h"
#include "core/session/abi_session_options_impl.h"
#include "webnn_execution_provider.h"

using namespace onnxruntime;

namespace onnxruntime {
struct WebNNProviderFactory : IExecutionProviderFactory {
  WebNNProviderFactory(uint32_t webnn_flags)
      : webnn_flags_(webnn_flags) {}
  ~WebNNProviderFactory() override {}

  std::unique_ptr<IExecutionProvider> CreateProvider() override;
  uint32_t webnn_flags_;
};

std::unique_ptr<IExecutionProvider> WebNNProviderFactory::CreateProvider() {
  return std::make_unique<WebNNExecutionProvider>(webnn_flags_);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_WebNN(uint32_t webnn_flags) {
  return std::make_shared<onnxruntime::WebNNProviderFactory>(webnn_flags);
}
}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_WebNN,
                    _In_ OrtSessionOptions* options, uint32_t webnn_flags) {
  options->provider_factories.push_back(onnxruntime::CreateExecutionProviderFactory_WebNN(webnn_flags));
  return nullptr;
}
