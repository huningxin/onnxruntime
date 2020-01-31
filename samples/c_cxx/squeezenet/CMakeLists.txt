add_executable(squeezenet main.cc)
target_link_libraries(squeezenet PRIVATE onnxruntime)