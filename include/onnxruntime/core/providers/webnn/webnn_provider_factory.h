// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"

// WebNNFlags are bool options we want to set for WebNN EP
// This enum is defined as bit flats, and cannot have negative value
// To generate an uint32_t webnn_flags for using with OrtSessionOptionsAppendExecutionProvider_WebNN below,
//   uint32_t webnn_flags = 0;
//   webnn_flags |= WEBNN_FLAG_USE_CPU_ONLY;
enum WebNNFlags {
  WEBNN_FLAG_USE_NONE = 0x000,

  WEBNN_FLAG_USE_CPU = 0x001,

  WEBNN_FLAG_USE_GPU = 0x002,

  WEBNN_FLAG_LAST = WEBNN_FLAG_USE_GPU,
};

#ifdef __cplusplus
extern "C" {
#endif

ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_WebNN,
                          _In_ OrtSessionOptions* options, uint32_t webnn_flags);

#ifdef __cplusplus
}
#endif
