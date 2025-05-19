import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import onnxruntime
import onnx
from onnx2torch import convert

def model_cnn_2layer(in_ch, in_dim, width, linear_size=128):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 8 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model

def model_cnn_3layer(in_ch, in_dim, kernel_size, width):
    if kernel_size == 5:
        h = (in_dim - 4) // 4
    elif kernel_size == 3:
        h = in_dim // 4
    else:
        raise ValueError("Unsupported kernel size")
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4 * width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 8 * width, kernel_size=kernel_size, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8 * width, 8 * width, kernel_size=4, stride=4, padding=0),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * width * h * h, width * 64),
        nn.Linear(width * 64, 10)
    )
    return model


# Fully connected network with configurable layer sizes
# parameter in_dim: input dimension
# parameter layers: a list specifying the number of neurons in each hidden layer
# parameter activation: activation function to use between layers
# parameter out_dim: number of output classes
def model_fc_network(in_dim, layers, out_dim=10, activation=nn.ReLU):
    assert len(layers) >= 1, "Must have at least one hidden layer"
    
    units = [nn.Flatten()]
    prev_size = in_dim
    
    # Add hidden layers
    for layer_size in layers:
        units.append(nn.Linear(prev_size, layer_size))
        units.append(activation())
        prev_size = layer_size
    
    # Add output layer
    units.append(nn.Linear(prev_size, out_dim))
    
    model = nn.Sequential(*units)
    return model


# Initialize network weights using various methods
# parameter model: the neural network model
# parameter method: initialization method ('xavier_uniform', 'xavier_normal', 'kaiming_uniform', 'kaiming_normal', 'normal', or 'uniform')
# parameter gain: scaling factor for the initialization methods that support it
def initialize_weights(model, method='xavier_uniform', gain=1.0):
    for module in model.modules():
        if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
            if method == 'xavier_uniform':
                nn.init.xavier_uniform_(module.weight, gain=gain)
            elif method == 'xavier_normal':
                nn.init.xavier_normal_(module.weight, gain=gain)
            elif method == 'kaiming_uniform':
                nn.init.kaiming_uniform_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif method == 'kaiming_normal':
                nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in', nonlinearity='relu')
            elif method == 'normal':
                nn.init.normal_(module.weight, mean=0.0, std=0.01)
            elif method == 'uniform':
                nn.init.uniform_(module.weight, a=-0.01, b=0.01)
            else:
                raise ValueError(f"Unsupported initialization method: {method}")
            
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    return model

# Convert PyTorch model to ONNX format
# parameter model: PyTorch model to convert
# parameter dummy_input: sample input tensor with correct shape
# parameter output_path: path to save the ONNX model
# parameter dynamic_axes: dictionary of dynamic axes (optional)
# parameter opset_version: ONNX opset version to use (default: 11)
def convert_to_onnx(model, dummy_input, output_path, dynamic_axes=None, opset_version=11):
    # Set model to evaluation mode
    model.eval()
    
    # Export the model
    torch.onnx.export(
        model,                   # model being run
        dummy_input,             # model input (or a tuple for multiple inputs)
        output_path,             # where to save the model
        export_params=True,      # store the trained parameter weights inside the model file
        opset_version=opset_version,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        input_names=['input'],   # the model's input names
        output_names=['output'], # the model's output names
        dynamic_axes=dynamic_axes # variable length axes
    )
    
    print(f"Model exported to {output_path}")
    return output_path

# Verify ONNX model outputs match PyTorch model outputs
# parameter pytorch_model: original PyTorch model
# parameter onnx_path: path to exported ONNX model
# parameter test_input: input tensor to test both models
# parameter rtol: relative tolerance for comparison
# parameter atol: absolute tolerance for comparison
def verify_onnx_model(pytorch_model, onnx_path, test_input, rtol=1e-3, atol=1e-4):
    
    # Set PyTorch model to evaluation mode
    pytorch_model.eval()
    
    # Run PyTorch model
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    
    # Run ONNX model
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # Prepare input for ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
    
    # Run ONNX model
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare outputs
    pytorch_output_np = pytorch_output.cpu().numpy()
    onnx_output_np = ort_outputs[0]
    
    # Check if outputs are close
    is_close = np.allclose(pytorch_output_np, onnx_output_np, rtol=rtol, atol=atol)
    
    if is_close:
        print("PyTorch and ONNX model outputs match within tolerance!")
    else:
        print("WARNING: PyTorch and ONNX model outputs differ!")
        print(f"Max absolute difference: {np.max(np.abs(pytorch_output_np - onnx_output_np))}")
        print(f"Max relative difference: {np.max(np.abs((pytorch_output_np - onnx_output_np) / (pytorch_output_np + 1e-10)))}")
    
    return is_close, pytorch_output_np, onnx_output_np

# Load an ONNX model and convert it to a PyTorch model using onnx2pytorch
# parameter onnx_path: path to the ONNX model to load
# parameter device: device to run the model on ('cpu' or 'cuda')
def load_onnx_to_pytorch(onnx_path, device='cpu'):
    # Load ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert to PyTorch using onnx2pytorch
    pytorch_model = convert(onnx_model)
    pytorch_model.to(device)
    
    return pytorch_model

# Verify that a loaded PyTorch model matches the original ONNX model
# parameter pytorch_model: PyTorch model to verify
# parameter onnx_path: path to the ONNX model
# parameter test_input: input tensor to test both models
# parameter rtol: relative tolerance for comparison
# parameter atol: absolute tolerance for comparison
def verify_loaded_model(pytorch_model, onnx_path, test_input, rtol=1e-3, atol=1e-4):
    # Set PyTorch model to evaluation mode
    pytorch_model.eval()
    
    # Run PyTorch model
    with torch.no_grad():
        pytorch_output = pytorch_model(test_input)
    
    # Run ONNX model directly
    ort_session = onnxruntime.InferenceSession(onnx_path)
    
    # Prepare input for ONNX model
    ort_inputs = {ort_session.get_inputs()[0].name: test_input.cpu().numpy()}
    
    # Run ONNX model
    ort_outputs = ort_session.run(None, ort_inputs)
    
    # Compare outputs
    pytorch_output_np = pytorch_output.cpu().numpy()
    onnx_output_np = ort_outputs[0]
    
    # Check if outputs are close
    is_close = np.allclose(pytorch_output_np, onnx_output_np, rtol=rtol, atol=atol)
    
    if is_close:
        print("Loaded PyTorch model and ONNX model outputs match within tolerance!")
    else:
        print("WARNING: Loaded PyTorch model and ONNX model outputs differ!")
        print(f"Max absolute difference: {np.max(np.abs(pytorch_output_np - onnx_output_np))}")
        print(f"Max relative difference: {np.max(np.abs((pytorch_output_np - onnx_output_np) / (pytorch_output_np + 1e-10)))}")
    
    return is_close, pytorch_output_np, onnx_output_np