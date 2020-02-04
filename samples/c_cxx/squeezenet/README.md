# Build Instructions
See [../README.md](../README.md)

Build with "-Donnxruntime_USE_DML=ON" to use GPU and VPU.

# Prepare data
Download the SqueezeNet.onnx and SqueezeNet_fp16.onnx from [Windows-Machine-Learning models](https://github.com/microsoft/Windows-Machine-Learning/tree/master/SharedContent/models).

# Run
Command to run the application:
```
squeezenet.exe [fp32|fp16] [cpu|gpu|vpu]
```

For example, the command to run Squeezenet_fp16.onnx on VPU:
```
squeezenet.exe fp16 vpu
```