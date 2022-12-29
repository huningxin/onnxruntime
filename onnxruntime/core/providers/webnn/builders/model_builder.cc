// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>

#include "model_builder.h"
#include "model.h"
#include "helper.h"
#include "op_builder_factory.h"

#include "core/common/safeint.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/common.h"
#include "core/providers/shared/utils/utils.h"
#include "core/providers/webnn/webnn_provider_factory.h"

#ifdef __EMSCRIPTEN__
#include <emscripten.h>
#include <emscripten/html5.h>
#include <emscripten/html5_webnn.h>
#else
#include <webnn/webnn_proc.h>
#include <webnn/native/WebnnNative.h>
#endif

namespace onnxruntime {
namespace webnn {

ModelBuilder::ModelBuilder(const GraphViewer& graph_viewer, const logging::Logger& logger,
                           uint32_t device_flags, uint32_t power_flags)
    : graph_viewer_(graph_viewer),
      logger_(logger),
      device_flags_(device_flags),
      power_flags_(power_flags) {
}

Status ModelBuilder::Initialize() {
  // Create WebNN context and graph builder
  ::wnn::ContextOptions options;
  options.devicePreference = ::wnn::DevicePreference::Default;
  options.powerPreference = ::wnn::PowerPreference::Default;
  if (device_flags_ == WEBNN_DEVICE_FLAG_USE_GPU) {
    options.devicePreference = ::wnn::DevicePreference::Gpu;
  } else if (device_flags_ == WEBNN_DEVICE_FLAG_USE_CPU) {
    options.devicePreference = ::wnn::DevicePreference::Cpu;
  }
  if (power_flags_ == WEBNN_POWER_FLAG_USE_HIGH_PERFORMANCE) {
    options.powerPreference = ::wnn::PowerPreference::High_performance;
  } else if (power_flags_ == WEBNN_POWER_FLAG_USE_LOW_POWER) {
    options.powerPreference = ::wnn::PowerPreference::Low_power;
  }

#ifdef __EMSCRIPTEN__
  ::wnn::Context context = emscripten_webnn_create_context(&options);
#else
  std::unique_ptr<::webnn::native::Instance> instance;
  instance = std::make_unique<::webnn::native::Instance>();
  WebnnProcTable backendProcs = ::webnn::native::GetProcs();
  webnnProcSetProcs(&backendProcs);
  ::wnn::Context context = instance->CreateContext(&options);


#endif
  if (!context) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create WebNN context.");
  }
#ifndef __EMSCRIPTEN__
  context.SetUncapturedErrorCallback(
    [](WNNErrorType type, char const* message, void* userData) {
      ModelBuilder* builder = reinterpret_cast<ModelBuilder*>(userData);
      if (type != WNNErrorType_NoError) {
        LOGS(builder->logger_, ERROR) << "Uncaptured Error type is "
            << type << ", message is " << message;
      }
    },
    this);
#endif
  builder_ = ::wnn::CreateGraphBuilder(context);
  context_ = context;
  if (!builder_) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create WebNN graph builder.");
  }

  PreprocessInitializers();
  PreprocessActivations();
  ORT_RETURN_IF_ERROR(RegisterInitializers());
  ORT_RETURN_IF_ERROR(RegisterModelInputs());
  ORT_RETURN_IF_ERROR(AddOperations());
  ORT_RETURN_IF_ERROR(RegisterModelOutputs());

  return Status::OK();
}

/* static */ const IOpBuilder* ModelBuilder::GetOpBuilder(const Node& node) {
  const auto& op_builders = GetOpBuilders();
  const auto it = op_builders.find(node.OpType());
  if (it != op_builders.cend())
    return it->second;

  return nullptr;
}

void ModelBuilder::PreprocessInitializers() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      op_builder->AddInitializersToSkip(*this, *node);
    }
  }
}

void ModelBuilder::PreprocessActivations() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    const auto& op_type(node->OpType());

    if (op_type == "Relu") {
      activation_nodes_.emplace(node->Index(), builder_.ReluOperator());
    } else if (op_type == "LeakyRelu") {
      NodeAttrHelper helper(*node);
      wnn::LeakyReluOptions options;
      options.alpha = helper.Get("alpha", (float)0.0);
      activation_nodes_.emplace(node->Index(), builder_.LeakyReluOperator(&options));
    } else if (op_type == "Sigmoid") {
      activation_nodes_.emplace(node->Index(), builder_.SigmoidOperator());
    } else if (op_type == "Tanh") {
      activation_nodes_.emplace(node->Index(), builder_.TanhOperator());
    } else if (op_type == "Clip") {
      wnn::ClampOptions clamp_options;
      GetClipMinMax(GetInitializerTensors(), *node, clamp_options.minValue, clamp_options.maxValue, logger_);
      activation_nodes_.emplace(node->Index(), builder_.ClampOperator(&clamp_options));
    }
  }
}

Status ModelBuilder::RegisterInitializers() {
  for (const auto& pair : GetInitializerTensors()) {
    const auto& tensor = *pair.second;
    const auto& name = tensor.name();
    if (Contains(skipped_initializers_, name))
      continue;


    const auto& shape = tensor.dims();
    std::vector<int32_t> dims;
    if (shape.empty()) {
      // This is a scalar initializer, WebNN requires a shape, make this a {1} tensor
      dims = {1};
    } else {
      std::transform(shape.cbegin(), shape.cend(),
                     std::back_inserter(dims),
                     [](int64_t dim) -> int32_t { return SafeInt<int32_t>(dim); });
    }
    ::wnn::OperandDescriptor desc;
    desc.dimensions = dims.data();
    desc.dimensionsCount = SafeInt<uint32_t>(dims.size());

    auto data_type = tensor.data_type();
    if (data_type == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
      unpacked_tensors_.push_back({});
      std::vector<uint8_t>& unpacked_tensor = unpacked_tensors_.back();
      ORT_RETURN_IF_ERROR(onnxruntime::utils::UnpackInitializerData(tensor, unpacked_tensor));
      auto num_elements = SafeInt<size_t>(Product(tensor.dims()));
      desc.type = ::wnn::OperandType::Float32;
      wnn::ArrayBufferView bufferView;
      bufferView.buffer = reinterpret_cast<float*>(unpacked_tensor.data());
      bufferView.byteLength = num_elements * sizeof(float);
      operands_[name] = builder_.Constant(&desc, &bufferView);
    } else {
      // TODO: support other type
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "The initializer of graph has unsupported type, name: ",
                              tensor.name(), " type: ", data_type);
    }
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputOutput(const NodeArg& node_arg, bool is_input) {
  const auto& name = node_arg.Name();
  const std::string input_output_type = is_input ? "input" : "output";

  if (is_input) {
    // input should not be an initializer
    if (Contains(GetInitializerTensors(), name))
      return Status::OK();

    // This input will not be used
    if (Contains(skipped_inputs_, name))
      return Status::OK();
  }

  std::vector<int32_t> dims;
  {  // input_output shape
    const auto* shape_proto = node_arg.Shape();
    ORT_RETURN_IF(shape_proto == nullptr,
                  "shape_proto cannot be null for ", input_output_type, ": ", name);
    const auto& shape = shape_proto->dim();
    if (shape.empty()) {
      // If we have an empty shape, this is a scalar input,
      dims.push_back(1);

      // we need to change the shapes of these scalar outputs back to {} when WebNN EP returns these values to ORT
      if (!is_input) {
        AddScalarOutput(name);
      }
    } else {
      dims.reserve(shape.size());
      for (const auto& dim : shape) {
        if (!dim.has_dim_value()) {
          // FIXME: support dyanmic shape
          dims.push_back(1);
        } else {
          dims.push_back(SafeInt<int32_t>(dim.dim_value()));
        }
      }
    }
  }

  ::wnn::OperandDescriptor desc;
  desc.dimensions = dims.data();
  desc.dimensionsCount = SafeInt<uint32_t>(dims.size());

  int32_t data_type;
  {  // type
    const auto* type_proto = node_arg.TypeAsProto();
    if (!type_proto || !type_proto->tensor_type().has_elem_type()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "The  ", input_output_type, " of graph doesn't have elem_type: ", name);
    }

    data_type = type_proto->tensor_type().elem_type();
    switch (data_type) {
      case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
        desc.type = ::wnn::OperandType::Float32;
        break;
      default: {
        // TODO: support other type
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                               "The  ", input_output_type, " of graph doesn't have valid type, name: ", name,
                               " type: ", type_proto->tensor_type().elem_type());
      }
    }
  }

  if (is_input) {
    operands_[name] = builder_.Input(name.c_str(), &desc);
    input_names_.push_back(name);
  } else {
    output_names_.push_back(name);
  }

  std::vector<int64_t> shape;
  std::transform(dims.cbegin(), dims.cend(),
                 std::back_inserter(shape),
                 [](int32_t dim) -> int64_t { return SafeInt<int64_t>(dim); });
  input_output_info_.emplace(name, OnnxTensorInfo{data_type, shape});

  return Status::OK();
}

Status ModelBuilder::RegisterModelInputs() {
  for (const auto* node_arg : graph_viewer_.GetInputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, true /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::AddOperations() {
  const auto& node_indices = graph_viewer_.GetNodesInTopologicalOrder();
  for (size_t i = 0; i < node_indices.size(); i++) {
    const auto* node(graph_viewer_.GetNode(node_indices[i]));
    if (const auto* op_builder = GetOpBuilder(*node)) {
      ORT_RETURN_IF_ERROR(op_builder->AddToModelBuilder(*this, *node, logger_));
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Node [", node->Name(), "], type [", node->OpType(), "] is not supported");
    }
  }

  return Status::OK();
}

Status ModelBuilder::RegisterModelOutputs() {
  for (const auto* node_arg : graph_viewer_.GetOutputs()) {
    ORT_RETURN_IF_ERROR(RegisterModelInputOutput(*node_arg, false /* is_input */));
  }

  return Status::OK();
}

Status ModelBuilder::Compile(std::unique_ptr<Model>& model) {
  ORT_RETURN_IF_ERROR(Initialize());
  ::wnn::NamedOperands named_operands = ::wnn::CreateNamedOperands();
  for (auto name : output_names_) {
    named_operands.Set(name.c_str(), operands_[name]);
  }
  ::wnn::Graph graph = builder_.Build(named_operands);
  if (!graph) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to build WebNN graph.");
  }
  model.reset(new Model(std::move(context_), std::move(graph), logger_, device_flags_, power_flags_));
  model->SetInputs(std::move(input_names_));
  model->SetOutputs(std::move(output_names_));
  model->SetScalarOutputs(std::move(scalar_outputs_));
  model->SetInputOutputInfo(std::move(input_output_info_));
  return Status::OK();
}

::wnn::FusionOperator ModelBuilder::FindActivation(const Node& node, const NodeArg& output) {
  ::wnn::FusionOperator fused_op;

  for (auto it = node.OutputEdgesBegin(), end = node.OutputEdgesEnd(); it != end; ++it) {
    const auto& dst_node = it->GetNode();
    const auto* dst_input = dst_node.InputDefs()[it->GetDstArgIndex()];
    if (Contains(activation_nodes_, dst_node.Index())) {
      if (&output == dst_input) {
        fused_op = activation_nodes_.at(dst_node.Index());
      }
    } else {
      // if there is any other non-relu node using the output
      // will add relu separately
      if (&output == dst_input)
        return ::wnn::FusionOperator();
    }
  }

  // if output is a graph output, will add relu separately
  if (fused_op != nullptr) {
    for (const auto* graph_output : graph_viewer_.GetOutputs()) {
      if (&output == graph_output)
        return ::wnn::FusionOperator();
    }

    LOGS_DEFAULT(VERBOSE) << "Node [" << node.Name() << "] type [" << node.OpType()
                          << "], fused the output [" << output.Name() << "]";

    fused_activations_.insert(output.Name());
  }

  return fused_op;
}

void ModelBuilder::AddScalarOutput(const std::string& output_name) {
  scalar_outputs_.insert(output_name);
}

void ModelBuilder::AddOperand(const std::string& name, const ::wnn::Operand& operand) {
  operands_[name] = operand;
}

void ModelBuilder::AddInitializerToSkip(const std::string& tensor_name) {
  skipped_initializers_.insert(tensor_name);
}

void ModelBuilder::AddInputToSkip(const std::string& input_name) {
  skipped_inputs_.insert(input_name);
}

std::string ModelBuilder::GetUniqueName(const std::string& base_name) {
  std::string unique_name;
  do {
    std::ostringstream os;
    os << base_name << "_token_" << name_token_++;
    unique_name = os.str();
  } while (Contains(unique_names_, unique_name));

  return unique_name;
}

}  // namespace webnn
}  // namespace onnxruntime
