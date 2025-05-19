import torch
import os
import numpy as np
from pathlib import Path

from training_utils import load_mnist, ensemble_cert_train
from network_utils import convert_to_onnx

# File paths for the 5 MNIST IBP networks
MNIST_IBP_NETWORKS = [
    "project_networks/mnist_cnn_2layer_w1_ibp.onnx",
    "project_networks/mnist_cnn_2layer_w2_ibp.onnx",
    "project_networks/mnist_cnn_3layer_k3_w2_ibp.onnx",
    "project_networks/mnist_cnn_3layer_k5_w1_ibp.onnx",
    "project_networks/mnist_cnn_3layer_k5_w2_ibp.onnx"
]

# Create output directory for ensemble models
OUTPUT_DIR = "project_networks/ensemble_models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def main():
    # Set device
    device = 'cuda:2'
    print(f"Using device: {device}")
    
    # Load MNIST datasets
    train_loader = load_mnist(batch_size=128, train=True)
    test_loader = load_mnist(batch_size=128, train=False)
    print("MNIST dataset loaded")
    
    # Check if all required model files exist
    missing_models = [path for path in MNIST_IBP_NETWORKS if not os.path.exists(path)]
    if missing_models:
        print(f"Error: The following model files are missing: {missing_models}")
        return
    
    # Training hyperparameters 
    num_epochs = 20        # Number of training epochs
    epsilon = 0.3          # L-infinity perturbation radius
    tau = 0.4              # Size of box around adversarial examples
    adv_steps = 10         # Number of steps for PGD attack
    adv_step_size = 0.05   # Step size for PGD attack
    learning_rate = 0.001  # Learning rate for optimizers
    
    print(f"Starting ensemble certification training with {len(MNIST_IBP_NETWORKS)} models")
    print(f"Hyperparameters: epochs={num_epochs}, epsilon={epsilon}, tau={tau}")
    
    # Run ensemble certification training
    trained_models, history = ensemble_cert_train(
        model_paths=MNIST_IBP_NETWORKS,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=num_epochs,
        device=device,
        verbose=True,
        epsilon=epsilon,
        tau=tau,
        adv_steps=adv_steps,
        adv_step_size=adv_step_size,
        learning_rate=learning_rate
    )
    
    # Get a sample input for ONNX conversion
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[:1].to(device)  # Just use one sample
    
    # Save trained models
    for i, model in enumerate(trained_models):
        # Get original model name from file path
        original_name = os.path.basename(MNIST_IBP_NETWORKS[i]).replace("_ibp.onnx", "")
        
        # Create output path
        output_path = os.path.join(OUTPUT_DIR, f"{original_name}_ensemble1.onnx")
        
        # Save model in ONNX format
        convert_to_onnx(model, sample_input, output_path)
        
        # Get metrics from history
        final_val_acc = history[i]['val_acc'][-1] if 'val_acc' in history[i] else 0
        final_verified_acc = history[i]['val_verified_acc'][-1] if 'val_verified_acc' in history[i] else 0
        
        print(f"Model {i+1}/{len(trained_models)} saved to {output_path}")
        print(f"  Validation accuracy: {final_val_acc:.2f}%")
        print(f"  Verified accuracy: {final_verified_acc:.2f}%")
    
    # Save training history
    history_path = os.path.join(OUTPUT_DIR, "training_history.npy")
    np.save(history_path, history)
    print(f"Training history saved to {history_path}")

if __name__ == "__main__":
    main() 