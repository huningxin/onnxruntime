// Copyright(c) Microsoft Corporation.All rights reserved.
// Licensed under the MIT License.
//

#include <winrt/Windows.Foundation.h>

#include "d3dx12.h" // The D3D12 Helper Library that you downloaded.
#include "DirectML.h" // The DirectML header from the Windows SDK.
#include <dxgi1_4.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cassert>
#include <fstream>
#include <iostream>
#include <iterator>
#include <vector>
#include <initguid.h>
#include <dxcore.h>

#include <onnxruntime_cxx_api.h>

#define USE_VPU 0

#ifdef USE_DML
#include "onnxruntime/core/providers/dml/dml_provider_factory.h"
#endif

// A stopwatch to measure the time passed (in milliseconds ) between current Stop call and the closest Start call that
// has been called before.
class Timer {
 public:
  void Start() {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    m_startTime = static_cast<double>(t.QuadPart);
  }

  double Stop() {
    LARGE_INTEGER stopTime;
    QueryPerformanceCounter(&stopTime);
    double t = static_cast<double>(stopTime.QuadPart) - m_startTime;
    LARGE_INTEGER tps;
    QueryPerformanceFrequency(&tps);
    return t / static_cast<double>(tps.QuadPart) * 1000;
  }

 private:
  double m_startTime;
};

using winrt::com_ptr;
using winrt::check_hresult;
using winrt::check_bool;
using winrt::handle;

std::string DriverDescription(com_ptr<IDXCoreAdapter>& adapter, bool selected = false) {
  // If the adapter is a software adapter then don't consider it for index selection
  size_t driverDescriptionSize;
  check_hresult(adapter->GetPropertySize(DXCoreAdapterProperty::DriverDescription,
                                         &driverDescriptionSize));
  CHAR* driverDescription = new CHAR[driverDescriptionSize];
  check_hresult(adapter->GetProperty(DXCoreAdapterProperty::DriverDescription,
                                     driverDescriptionSize, driverDescription));
  if (selected) {
    printf("Using adapter : %s\n", driverDescription);
  }

  std::string driverDescriptionStr = std::string(driverDescription);
  free(driverDescription);

  return driverDescriptionStr;
}

void InitWithDXCore(com_ptr<ID3D12Device>& d3D12Device,
                    com_ptr<ID3D12CommandQueue>& commandQueue,
                    com_ptr<ID3D12CommandAllocator>& commandAllocator,
                    com_ptr<ID3D12GraphicsCommandList>& commandList) {
  HMODULE library = nullptr;
  library = LoadLibrary("dxcore.dll");
  if (!library) {
    //throw hresult_invalid_argument(L"DXCore isn't support on this manchine.");
    std::wcout << L"DXCore isn't support on this manchine. ";
    return;
  }

  com_ptr<IDXCoreAdapterFactory> adapterFactory;
  check_hresult(DXCoreCreateAdapterFactory(IID_PPV_ARGS(adapterFactory.put())));

  com_ptr<IDXCoreAdapterList> adapterList;
  const GUID dxGUIDs[] = {DXCORE_ADAPTER_ATTRIBUTE_D3D12_CORE_COMPUTE};

  check_hresult(
      adapterFactory->CreateAdapterList(ARRAYSIZE(dxGUIDs), dxGUIDs, IID_PPV_ARGS(adapterList.put())));

  com_ptr<IDXCoreAdapter> currAdapter = nullptr;
  IUnknown* pAdapter = nullptr;
  com_ptr<IDXGIAdapter> dxgiAdapter;
  D3D_FEATURE_LEVEL d3dFeatureLevel = D3D_FEATURE_LEVEL_1_0_CORE;
  D3D12_COMMAND_LIST_TYPE commandQueueType = D3D12_COMMAND_LIST_TYPE_COMPUTE;
  for (UINT i = 0; i < adapterList->GetAdapterCount(); i++) {
    currAdapter = nullptr;
    check_hresult(adapterList->GetAdapter(i, currAdapter.put()));

    bool isHardware;
    check_hresult(currAdapter->GetProperty(DXCoreAdapterProperty::IsHardware, &isHardware));
#if USE_VPU == 1
    std::string adapterNameStr = "VPU";
    std::string driverDescriptionStr = DriverDescription(currAdapter);
    std::transform(driverDescriptionStr.begin(), driverDescriptionStr.end(),
                   driverDescriptionStr.begin(), ::tolower);
    std::transform(adapterNameStr.begin(), adapterNameStr.end(), adapterNameStr.begin(),
                   ::tolower);
    if (isHardware && strstr(driverDescriptionStr.c_str(), adapterNameStr.c_str())) {
      pAdapter = currAdapter.get();
      break;
    }
#else
    // Check if adapter selected has DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS attribute selected. If
    // so, then GPU was selected that has D3D12 and D3D11 capabilities. It would be the most stable
    // to use DXGI to enumerate GPU and use D3D_FEATURE_LEVEL_11_0 so that image tensorization for
    // video frames would be able to happen on the GPU.
    if (isHardware && currAdapter->IsAttributeSupported(DXCORE_ADAPTER_ATTRIBUTE_D3D12_GRAPHICS)) {
      d3dFeatureLevel = D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0;
      com_ptr<IDXGIFactory4> dxgiFactory4;
      HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(dxgiFactory4.put()));
      if (hr == S_OK) {
        // If DXGI factory creation was successful then get the IDXGIAdapter from the LUID
        // acquired from the selectedAdapter
        LUID adapterLuid;
        check_hresult(currAdapter->GetProperty(DXCoreAdapterProperty::InstanceLuid, &adapterLuid));
        check_hresult(dxgiFactory4->EnumAdapterByLuid(adapterLuid, __uuidof(IDXGIAdapter),
                                                      dxgiAdapter.put_void()));
        pAdapter = dxgiAdapter.get();
        break;
      }
    }
#endif
  }

  if (currAdapter == nullptr) {
    std::wcout << L"ERROR: No matching adapter with given adapter name: ";
    return;
  }
  DriverDescription(currAdapter, true);

  // create D3D12Device
  check_hresult(
      D3D12CreateDevice(pAdapter, d3dFeatureLevel, __uuidof(ID3D12Device), d3D12Device.put_void()));

  // create D3D12 command queue from device
  //com_ptr<ID3D12CommandQueue> d3d12CommandQueue;
  D3D12_COMMAND_QUEUE_DESC commandQueueDesc = {};
  commandQueueDesc.Type = commandQueueType;
  check_hresult(d3D12Device->CreateCommandQueue(&commandQueueDesc, __uuidof(ID3D12CommandQueue),
                                                commandQueue.put_void()));

  check_hresult(d3D12Device->CreateCommandAllocator(
      commandQueueType,
      __uuidof(commandAllocator),
      commandAllocator.put_void()));

  check_hresult(d3D12Device->CreateCommandList(
      0,
      commandQueueType,
      commandAllocator.get(),
      nullptr,
      __uuidof(commandList),
      commandList.put_void()));
}


int main(int argc, char* argv[]) {
#if USE_DML
  com_ptr<ID3D12Device> d3D12Device;
  com_ptr<ID3D12CommandQueue> commandQueue;
  com_ptr<ID3D12CommandAllocator> commandAllocator;
  com_ptr<ID3D12GraphicsCommandList> commandList;
#endif
  char* ep = (argc >= 2) ? argv[1] : "";

  //*************************************************************************
  // initialize  enviroment...one enviroment per process
  // enviroment maintains thread pools and other state info
  Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");

  // initialize session options if needed
  Ort::SessionOptions session_options;
  session_options.SetIntraOpNumThreads(1);

  // If onnxruntime.dll is built with CUDA enabled, we can uncomment out this line to use CUDA for this
  // session (we also need to include cuda_provider_factory.h above which defines it)
  // #include "cuda_provider_factory.h"
  // OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 1);

  if (std::string(ep) == std::string("dml")) {
    std::wcout << "Using DML" << std::endl;
#ifdef USE_DML
    // Set up Direct3D 12.
    InitWithDXCore(d3D12Device, commandQueue, commandAllocator, commandList);

	// Create the DirectML device.
    DML_CREATE_DEVICE_FLAGS dmlCreateDeviceFlags = DML_CREATE_DEVICE_FLAG_NONE;
    com_ptr<IDMLDevice> dmlDevice;
    check_hresult(DMLCreateDevice(
        d3D12Device.get(),
        dmlCreateDeviceFlags,
        __uuidof(dmlDevice),
        dmlDevice.put_void()));

    DML_FEATURE_DATA_TENSOR_DATA_TYPE_SUPPORT support_f16 = {false};
    DML_FEATURE_QUERY_TENSOR_DATA_TYPE_SUPPORT fp16_query = {
        DML_TENSOR_DATA_TYPE_FLOAT16};
    check_hresult(dmlDevice->CheckFeatureSupport(
        DML_FEATURE_TENSOR_DATA_TYPE_SUPPORT, sizeof(fp16_query), &fp16_query,
        sizeof(support_f16), &support_f16));

    std::wcout << "Support float16: " << support_f16.IsSupported << std::endl;

    session_options.DisableMemPattern();
    session_options.SetExecutionMode(ORT_SEQUENTIAL);
    OrtSessionOptionsAppendExecutionProviderEx_DML(session_options, dmlDevice.get(), commandQueue.get());
#else
    std::wcout << "DML is not enabled in this build" << std::endl;
    return -1;
#endif
  } else {
    std::wcout << "Using CPU" << std::endl;
  }

  // Sets graph optimization level
  // Available levels are
  // ORT_DISABLE_ALL -> To disable all optimizations
  // ORT_ENABLE_BASIC -> To enable basic optimizations (Such as redundant node removals)
  // ORT_ENABLE_EXTENDED -> To enable extended optimizations (Includes level 1 + more complex optimizations like node fusions)
  // ORT_ENABLE_ALL -> To Enable All possible opitmizations
  session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

  //*************************************************************************
  // create session and load model into memory
  // using squeezenet version 1.3
  // URL = https://github.com/onnx/models/tree/master/squeezenet
#ifdef _WIN32
  const wchar_t* model_path = L"SqueezeNet.onnx";
#else
  const char* model_path = "squeezenet.onnx";
#endif

  printf("Using Onnxruntime C++ API\n");
  Ort::Session session(env, model_path, session_options);

  //*************************************************************************
  // print model input layer (node names, types, shape etc.)
  Ort::AllocatorWithDefaultOptions allocator;

  // print number of model input nodes
  size_t num_input_nodes = session.GetInputCount();
  std::vector<const char*> input_node_names;
  std::vector<int64_t> input_node_dims;  // simplify... this model has only 1 input node {1, 3, 224, 224}.
                                         // Otherwise need vector<vector<>>

  printf("Number of inputs = %zu\n", num_input_nodes);

  // iterate over all input nodes
  for (int i = 0; i < num_input_nodes; i++) {
    // print input node names
    char* input_name = session.GetInputName(i, allocator);
    if (std::string(input_name) != std::string("data_0"))
      continue;
    printf("Input %d : name=%s\n", i, input_name);
    input_node_names.push_back(input_name);

    // print input node types
    Ort::TypeInfo type_info = session.GetInputTypeInfo(i);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();

    ONNXTensorElementDataType type = tensor_info.GetElementType();
    printf("Input %d : type=%d\n", i, type);

    // print input shapes/dims
    input_node_dims = tensor_info.GetShape();
    printf("Input %d : num_dims=%zu\n", i, input_node_dims.size());
    for (int j = 0; j < input_node_dims.size(); j++)
      printf("Input %d : dim %d=%jd\n", i, j, input_node_dims[j]);
  }

  // Results should be...
  // Number of inputs = 1
  // Input 0 : name = data_0
  // Input 0 : type = 1
  // Input 0 : num_dims = 4
  // Input 0 : dim 0 = 1
  // Input 0 : dim 1 = 3
  // Input 0 : dim 2 = 224
  // Input 0 : dim 3 = 224

  //*************************************************************************
  // Similar operations to get output node information.
  // Use OrtSessionGetOutputCount(), OrtSessionGetOutputName()
  // OrtSessionGetOutputTypeInfo() as shown above.

  //*************************************************************************
  // Score the model using sample data, and inspect values

  size_t input_tensor_size = 224 * 224 * 3;  // simplify ... using known dim values to calculate size
                                             // use OrtGetTensorShapeElementCount() to get official size!

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> output_node_names = {"softmaxout_1"};

  // initialize input data with values in [0.0, 1.0]
  for (unsigned int i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, input_tensor_values.data(), input_tensor_size, input_node_dims.data(), 4);
  assert(input_tensor.IsTensor());

  std::vector<Ort::Value> output_tensors;
  Timer timer;
  for (int i = 0; i < 10; i++) {
    timer.Start();
    // score model & input tensor, get back output tensor
    output_tensors = session.Run(Ort::RunOptions{nullptr}, input_node_names.data(), &input_tensor, 1, output_node_names.data(), 1);
    assert(output_tensors.size() == 1 && output_tensors.front().IsTensor());
    std::wcout << L"Predict " << i << " elapsed time: " << timer.Stop() << " ms\n";
  }

  // Get pointer to output tensor float values
  float* floatarr = output_tensors.front().GetTensorMutableData<float>();
  assert(abs(floatarr[0] - 0.000045) < 1e-6);

  // score the model, and print scores for first 5 classes
  for (int i = 0; i < 5; i++)
    printf("Score for class [%d] =  %f\n", i, floatarr[i]);

  // Results should be as below...
  // Score for class[0] = 0.000045
  // Score for class[1] = 0.003846
  // Score for class[2] = 0.000125
  // Score for class[3] = 0.001180
  // Score for class[4] = 0.001317
  printf("Done!\n");
  return 0;
}
