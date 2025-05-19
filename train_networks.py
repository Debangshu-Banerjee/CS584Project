import torch
import os
from tqdm import tqdm
from pathlib import Path

from training_utils import load_mnist, load_cifar10, train_sabr
from network_utils import model_cnn_2layer, model_cnn_3layer, initialize_weights, convert_to_onnx
from send_notification import send_ntfy_message

# Create output directory if it doesn't exist
os.makedirs("project_networks", exist_ok=True)

def train_and_save_model(model, model_name, dataset_name, train_loader, test_loader, device, num_epochs=100, epsilon = 0.3):
    """
    Train a model using SABR method and save it in ONNX format
    
    Args:
        model: The neural network model
        model_name: Name for saving the model
        dataset_name: Name of the dataset (for file naming)
        train_loader: DataLoader for training
        test_loader: DataLoader for testing
        device: Device to train on
        num_epochs: Number of training epochs
    """
    print(f"Training {model_name} on {dataset_name}...")
    
    # Train with SABR method
    model, history = train_sabr(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        num_epochs=num_epochs,
        standard_epochs=20,  # Standard training epochs before IBP
        epsilon_schedule_length=50,  # Epochs to scale epsilon
        start_epsilon=0.0,
        end_epsilon=epsilon,
        ibp_weight_start=0.0,
        ibp_weight_end=1.0,
        tau=0.3,  # Size of box around adversarial example
        adv_steps=10,  # Steps for PGD attack (reduced for speed)
        adv_step_size=0.05,
        verbose=True
    )
    
    # Get final validation accuracy and verification percentage
    final_val_acc = history['val_acc'][-1]
    final_val_verified = history['val_verified_acc'][-1]
    
    # Get a sample input for ONNX conversion
    sample_input, _ = next(iter(test_loader))
    sample_input = sample_input[:1].to(device)  # Just use one sample
    
    # Save model in ONNX format
    model_path = f"project_networks/new_sabr/{dataset_name}_{model_name}_sabr.onnx"
    convert_to_onnx(model, sample_input, model_path)
    
    print(f"Saved model to {model_path}")
    
    # Send notification with metrics
    message = f"Training complete: {dataset_name}_{model_name}\nVal Accuracy: {final_val_acc:.4f}\nVal Verified: {final_val_verified:.4f}"
    send_ntfy_message(message)
    
    return model

def main():
    """Train all specified models on both datasets"""
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda:2'
    print(f"Using device: {device}")
    
    # Load datasets
    mnist_train = load_mnist(batch_size=128, train=True)
    mnist_test = load_mnist(batch_size=128, train=False)
    print("MNIST dataset loaded")
    
    cifar10_train = load_cifar10(batch_size=128, train=True)
    cifar10_test = load_cifar10(batch_size=128, train=False)
    print("CIFAR-10 dataset loaded")
    
    # Define model configurations
    model_configs = [
        # model_cnn_2layer configurations
        {
            "name": "cnn_2layer_w1",
            "create_fn": lambda in_ch, in_dim: model_cnn_2layer(in_ch, in_dim, width=1)
        },
        {
            "name": "cnn_2layer_w2",
            "create_fn": lambda in_ch, in_dim: model_cnn_2layer(in_ch, in_dim, width=2)
        },
        # model_cnn_3layer configurations
        {
            "name": "cnn_3layer_k5_w1",
            "create_fn": lambda in_ch, in_dim: model_cnn_3layer(in_ch, in_dim, kernel_size=5, width=1)
        },
        {
            "name": "cnn_3layer_k5_w2",
            "create_fn": lambda in_ch, in_dim: model_cnn_3layer(in_ch, in_dim, kernel_size=5, width=2)
        },
        {
            "name": "cnn_3layer_k3_w2",
            "create_fn": lambda in_ch, in_dim: model_cnn_3layer(in_ch, in_dim, kernel_size=3, width=2)
        }
    ]
    
    # Training parameters
    num_epochs = 100  # Adjust based on your time constraints
    
    # Train models on MNIST
    print("\n===== Training on MNIST =====")
    for config in model_configs:
        # MNIST: 1 channel, 28x28 images
        model = config["create_fn"](1, 28).to(device)
        model = initialize_weights(model)
        train_and_save_model(
            model=model,
            model_name=config["name"],
            dataset_name="mnist",
            train_loader=mnist_train,
            test_loader=mnist_test,
            device=device,
            num_epochs=num_epochs,
            epsilon=0.3
        )
    
    # Train models on CIFAR-10
    print("\n===== Training on CIFAR-10 =====")
    for config in model_configs:
        # CIFAR-10: 3 channels, 32x32 images
        model = config["create_fn"](3, 32).to(device)
        model = initialize_weights(model)
        train_and_save_model(
            model=model,
            model_name=config["name"],
            dataset_name="cifar10",
            train_loader=cifar10_train,
            test_loader=cifar10_test,
            device=device,
            num_epochs=num_epochs,
            epsilon=8/255
        )

if __name__ == "__main__":
    main() 