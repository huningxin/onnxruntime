// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include "onnxruntime_c_api.h"

// WebNNDeviceFlags and WebNNPowerFlags are bool options we want to set for WebNN EP
// These enums are defined as bit flats, and cannot have negative values
// To generate uint32_t webnn_device_flags and webnn_power_flags for using with
// OrtSessionOptionsAppendExecutionProvider_WebNN below,
//   uint32_t webnn_device_flags = 0;
//   uint32_t webnn_power_flags = 0;
//   webnn_device_flags |= WEBNN_DEVICE_FLAG_USE_CPU;
//   webnn_power_flags |= WEBNN_POWER_FLAG_USE_LOW_POWER;
enum WebNNDeviceFlags {
  WEBNN_DEVICE_FLAG_USE_NONE = 0x000,

  WEBNN_DEVICE_FLAG_USE_GPU = 0x001,

  WEBNN_DEVICE_FLAG_USE_CPU = 0x002,

  WEBNN_DEVICE_FLAG_USE_LAST = WEBNN_DEVICE_FLAG_USE_CPU,
};

enum WebNNPowerFlags {
  WEBNN_POWER_FLAG_USE_NONE = 0x000,

  WEBNN_POWER_FLAG_USE_HIGH_PERFORMANCE = 0x001,

  WEBNN_POWER_FLAG_USE_LOW_POWER = 0x002,

  WEBNN_POWER_FLAG_USE_LAST = WEBNN_POWER_FLAG_USE_LOW_POWER,
};

ORT_EXPORT ORT_API_STATUS(OrtSessionOptionsAppendExecutionProvider_WebNN,
                          _In_ OrtSessionOptions* options, int webnn_device_flags, int webnn_power_flags);
