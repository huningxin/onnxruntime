// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/providers/provider_test_utils.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

TEST(Reshape_WebGPU, Int64DataType) {
  auto webgpu_ep = DefaultWebGpuExecutionProvider();
  if (!webgpu_ep) {
    GTEST_SKIP() << "WebGPU execution provider is not available.";
  }

  OpTester test("Reshape");
  test.AddInput<int64_t>("data", {2, 3}, {1, 2, 3, 4, 5, 6});
  test.AddInput<int64_t>("shape", {2}, {3, 2});
  test.AddOutput<int64_t>("reshaped", {3, 2}, {1, 2, 3, 4, 5, 6});

  test.ConfigEp(std::move(webgpu_ep)).RunWithConfig();
}

}  // namespace test
}  // namespace onnxruntime
