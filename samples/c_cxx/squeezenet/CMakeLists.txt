add_executable(squeezenet main.cc d3dx12.h)
target_link_libraries(squeezenet PRIVATE onnxruntime WindowsApp dxgi d3d12 DirectML)
target_compile_features(squeezenet PUBLIC cxx_std_17)