import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
# Load ONNX models
import onnx
import onnx2pytorch
import network_utils

def load_mnist(batch_size=128, train=True):
    """
    Load MNIST dataset
    
    Args:
        batch_size: Batch size for dataloader
        train: If True, returns training set, else test set
        device: Device to load data to ('cuda' or 'cpu')
    
    Returns:
        data_loader: DataLoader for MNIST
        device: The device parameter for convenient chaining
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    dataset = torchvision.datasets.MNIST(
        root='./data', 
        train=train,
        download=True,
        transform=transform
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    return data_loader

def load_cifar10(batch_size=128, train=True):
    """
    Load CIFAR10 dataset
    
    Args:
        batch_size: Batch size for dataloader
        train: If True, returns training set, else test set
        device: Device to load data to ('cuda' or 'cpu')
    
    Returns:
        data_loader: DataLoader for CIFAR10
        device: The device parameter for convenient chaining
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=train,
        download=True,
        transform=transform
    )
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=2,
        pin_memory=True  # Faster data transfer to GPU
    )
    
    return data_loader

class IBPWrapper(nn.Module):
    """
    Wrapper class to convert a PyTorch model to perform interval bound propagation.
    The input dimension is doubled, with the first half representing the lower bound
    and the second half representing the upper bound.
    """
    def __init__(self, model, device='cuda'):
        super(IBPWrapper, self).__init__()
        self.model = model
        self.device = device
        # Create IBP layers
        self.ibp_layers = self._create_ibp_layers(model)
        self.to(device)  # Move model to device
        
    def _create_ibp_layers(self, model):
        """Convert regular model layers to IBP-compatible layers"""
        ibp_layers = nn.ModuleList()
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                ibp_layers.append(IBPLinear(module, self.device))
            elif isinstance(module, nn.Conv2d):
                ibp_layers.append(IBPConv2d(module, self.device))
            elif isinstance(module, nn.ReLU):
                ibp_layers.append(IBPReLU())
            elif isinstance(module, nn.Flatten):
                ibp_layers.append(IBPFlatten())
            
        return ibp_layers
    
    def forward(self, x):
        """
        Forward pass with interval bound propagation
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
            
        Returns:
            Tensor with lower and upper bounds. Shape is 
            [batch_size, 2*output_dim]
        """
        for layer in self.ibp_layers:
            x = layer(x)
            
        return x
    
    def standard_forward(self, x):
        """
        Standard forward pass using the original model
        
        Args:
            x: Input tensor with shape [batch_size, channels, height, width]
            
        Returns:
            Output tensor with shape [batch_size, output_dim]
        """
        return self.model(x)

class IBPLinear(nn.Module):
    """IBP wrapper for linear layer"""
    def __init__(self, linear_layer, device='cuda'):
        super(IBPLinear, self).__init__()
        self.weight = linear_layer.weight.to(device)
        if linear_layer.bias is not None:
            self.bias = linear_layer.bias.to(device)
        else:
            self.bias = None
        self.device = device
    
    def forward(self, x):
        batch_size = x.shape[0]
        input_dim = x.shape[1] // 2
        
        # Split input into lower and upper bounds
        lower = x[:, :input_dim]
        upper = x[:, input_dim:]
        
        # Reshape for linear layer if needed
        lower_flat = lower.reshape(batch_size, -1)
        upper_flat = upper.reshape(batch_size, -1)
        
        # Use midpoint-radius formulation
        u = (lower_flat + upper_flat) / 2  # midpoint
        r = (upper_flat - lower_flat) / 2  # radius
        
        # Compute output midpoint and radius
        out_u = F.linear(u, self.weight, self.bias)
        out_r = F.linear(r, torch.abs(self.weight), None)
        
        # Compute lower and upper bounds
        lower_out = out_u - out_r
        upper_out = out_u + out_r
        
        # Concatenate bounds
        out = torch.cat([lower_out, upper_out], dim=1)
        return out

class IBPConv2d(nn.Module):
    """IBP wrapper for Conv2d layer"""
    def __init__(self, conv_layer, device='cuda'):
        super(IBPConv2d, self).__init__()
        self.weight = conv_layer.weight.to(device)
        if conv_layer.bias is not None:
            self.bias = conv_layer.bias.to(device)
        else:
            self.bias = None
        self.stride = conv_layer.stride
        self.padding = conv_layer.padding
        self.dilation = conv_layer.dilation
        self.groups = conv_layer.groups
        self.device = device
    
    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1] // 2
        
        # Split input into lower and upper bounds
        lower = x[:, :channels]
        upper = x[:, channels:]
        
        # Use midpoint-radius formulation
        u = (lower + upper) / 2  # midpoint
        r = (upper - lower) / 2  # radius
        
        # Compute output midpoint and radius
        out_u = F.conv2d(u, self.weight, self.bias, self.stride, 
                        self.padding, self.dilation, self.groups)
        out_r = F.conv2d(r, torch.abs(self.weight), None, self.stride, 
                        self.padding, self.dilation, self.groups)
        
        # Compute lower and upper bounds
        lower_out = out_u - out_r
        upper_out = out_u + out_r
        
        # Concatenate bounds
        out = torch.cat([lower_out, upper_out], dim=1)
        return out

class IBPReLU(nn.Module):
    """IBP wrapper for ReLU layer"""
    def __init__(self):
        super(IBPReLU, self).__init__()
    
    def forward(self, x):
        channels = x.shape[1] // 2
        
        # Split input into lower and upper bounds
        lower = x[:, :channels]
        upper = x[:, channels:]
        
        # Apply ReLU to bounds
        lower = F.relu(lower)
        upper = F.relu(upper)
        
        # Concatenate bounds
        out = torch.cat([lower, upper], dim=1)
        return out

class IBPFlatten(nn.Module):
    """IBP wrapper for Flatten layer"""
    def __init__(self):
        super(IBPFlatten, self).__init__()
    
    def forward(self, x):
        batch_size = x.shape[0]
        channels = x.shape[1] // 2
        
        # Split input into lower and upper bounds
        lower = x[:, :channels]
        upper = x[:, channels:]
        
        # Flatten
        lower = lower.reshape(batch_size, -1)
        upper = upper.reshape(batch_size, -1)
        
        # Concatenate bounds
        out = torch.cat([lower, upper], dim=1)
        return out

def test_accuracy(model, test_loader, device='cuda'):
    """
    Test the accuracy of a model on a test dataset
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test set
        device: Device to run evaluation on
        
    Returns:
        accuracy: Test accuracy
    """
    model.eval()
    model = model.to(device)  # Ensure model is on the correct device
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # For regular model
            if not isinstance(model, IBPWrapper):
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                
            # For IBP model, only use the lower bounds for prediction
            # (could use average of bounds or another strategy)
            else:
                outputs = model.standard_forward(inputs)
                _, predicted = outputs.max(1)
            
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
    accuracy = 100.0 * correct / total
    return accuracy

def batch_accuracy(model, inputs, targets, device='cuda'):
    """
    Calculate accuracy for a single batch of inputs
    
    Args:
        model: PyTorch model
        inputs: Input tensors (batch)
        targets: Target labels (batch)
        device: Device to run evaluation on
        
    Returns:
        accuracy: Batch accuracy (percentage)
    """
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        # For regular model
        if not isinstance(model, IBPWrapper):
            outputs = model(inputs)
            _, predicted = outputs.max(1)
        # For IBP model
        else:
            outputs = model.standard_forward(inputs)
            _, predicted = outputs.max(1)
        
        correct = predicted.eq(targets).sum().item()
        total = targets.size(0)
    
    accuracy = 100.0 * correct / total
    return accuracy

def pgd_attack(model, inputs, targets, epsilon=0.3, step_size=0.01, num_iterations=40, device='cuda'):
    """
    Performs Projected Gradient Descent (PGD) attack on a batch of inputs
    
    Args:
        model: PyTorch model
        inputs: Input tensors (batch)
        targets: Target labels (batch)
        epsilon: Maximum L-infinity norm of perturbation
        step_size: Step size for each iteration
        num_iterations: Maximum number of iterations
        device: Device to run attack on ('cuda' or 'cpu')
        
    Returns:
        perturbed_inputs: Adversarial examples
    """
    model.eval()  # Set model to evaluation mode
    inputs = inputs.clone().detach().to(device)
    targets = targets.clone().detach().to(device)
    
    # Initialize with random perturbation within epsilon ball
    delta = torch.zeros_like(inputs)
    delta.uniform_(-epsilon, epsilon)
    delta.requires_grad = True
    
    # Ensure the perturbations are within valid image bounds
    delta.data = torch.clamp(inputs + delta.data, 0, 1) - inputs
    
    for _ in range(num_iterations):
        # Set requires_grad attribute
        delta.requires_grad = True
        
        # Forward pass
        if isinstance(model, IBPWrapper):
            outputs = model.standard_forward(inputs + delta)
        else:
            outputs = model(inputs + delta)
        
        # Calculate loss
        loss = F.cross_entropy(outputs, targets)
        
        # Get gradient
        loss.backward()
        
        # Update perturbation with gradient ascent
        delta.data = delta.data + step_size * delta.grad.sign()
        
        # Project back to epsilon L-infinity ball
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        
        # Ensure the perturbed inputs stay within valid range [0, 1]
        delta.data = torch.clamp(inputs + delta.data, 0, 1) - inputs
        
        # Reset gradients
        delta.grad.zero_()
    
    # Create adversarial examples
    perturbed_inputs = torch.clamp(inputs + delta.detach(), 0, 1)
    
    return perturbed_inputs

def train(model, train_loader, val_loader=None, optimizer=None, criterion=None, 
          num_epochs=10, device='cuda', scheduler=None, verbose=True):
    """
    Standard neural network training function
    
    Args:
        model: PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        optimizer: PyTorch optimizer (if None, uses Adam with default params)
        criterion: Loss function (if None, uses CrossEntropyLoss)
        num_epochs: Number of training epochs
        device: Device to run training on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
        verbose: Whether to print progress
        
    Returns:
        model: Trained model
        history: Dictionary containing training metrics (loss and accuracy)
    """
    # Move model to device
    model = model.to(device)
    
    # Default optimizer and loss function if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters())
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            if isinstance(model, IBPWrapper):
                outputs = model.standard_forward(inputs)
            else:
                outputs = model(inputs)
                
            # Calculate loss
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = 100.0 * correct / total
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation phase
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Forward pass
                    if isinstance(model, IBPWrapper):
                        outputs = model.standard_forward(inputs)
                    else:
                        outputs = model(inputs)
                        
                    # Calculate loss
                    loss = criterion(outputs, targets)
                    
                    # Track statistics
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            # Calculate epoch metrics
            val_loss = val_loss / len(val_loader.dataset)
            val_acc = 100.0 * correct / total
            
            # Save metrics
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
    
    return model, history

def convert_ibp_to_standard(ibp_model):
    """
    Convert an IBP wrapper model back to a standard PyTorch model
    
    Args:
        ibp_model: Model wrapped with IBPWrapper
        
    Returns:
        standard_model: The unwrapped standard PyTorch model
    """
    if not isinstance(ibp_model, IBPWrapper):
        return ibp_model  # Already a standard model
    
    # The original model is stored in the 'model' attribute of IBPWrapper
    standard_model = ibp_model.model
    return standard_model

def train_ibp(model, train_loader, val_loader=None, optimizer=None, criterion=None,
             num_epochs=100, device='cuda', scheduler=None, verbose=True,
             start_epsilon=0.0, end_epsilon=0.3, epsilon_schedule_length=50,
             standard_epochs=20, ibp_weight_start=0.0, ibp_weight_end=1.0):
    """
    Training with Interval Bound Propagation (IBP)
    
    Starts with standard training and gradually transitions to IBP training 
    with a scaling epsilon parameter.
    
    Args:
        model: PyTorch model to train (will be wrapped with IBPWrapper if not already)
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        optimizer: PyTorch optimizer (if None, uses Adam with default params)
        criterion: Loss function (if None, uses CrossEntropyLoss)
        num_epochs: Number of training epochs
        device: Device to run training on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
        verbose: Whether to print progress
        start_epsilon: Initial perturbation radius
        end_epsilon: Final perturbation radius
        epsilon_schedule_length: Number of epochs over which to scale epsilon
        standard_epochs: Number of epochs to train with standard loss only
        ibp_weight_start: Initial weight for IBP loss component
        ibp_weight_end: Final weight for IBP loss component
        
    Returns:
        model: Trained standard model (unwrapped from IBPWrapper)
        history: Dictionary containing training metrics
    """
    # Check if model is wrapped with IBPWrapper, if not, wrap it
    if not isinstance(model, IBPWrapper):
        ibp_model = IBPWrapper(model, device)
    else:
        ibp_model = model
    
    # Move model to device
    ibp_model = ibp_model.to(device)
    
    # Default optimizer and loss function if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(ibp_model.parameters())
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_verified_acc': [],
        'val_verified_acc': [],
        'epsilon': []
    }
    
    for epoch in range(num_epochs):
        # Determine current epsilon based on epoch
        if epoch < standard_epochs:
            current_epsilon = 0.0
            ibp_weight = 0.0
        else:
            # Scale epsilon and IBP weight according to schedule
            schedule_step = min(epoch - standard_epochs, epsilon_schedule_length) / epsilon_schedule_length
            current_epsilon = start_epsilon + (end_epsilon - start_epsilon) * schedule_step
            ibp_weight = ibp_weight_start + (ibp_weight_end - ibp_weight_start) * schedule_step
        
        history['epsilon'].append(current_epsilon)
        
        # Training phase
        ibp_model.train()
        train_loss = 0.0
        train_standard_acc = 0.0
        train_verified_acc = 0.0
        total = 0
        
        for inputs, targets in train_loader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Standard forward pass for clean accuracy
            standard_outputs = ibp_model.standard_forward(inputs)
            _, standard_preds = standard_outputs.max(1)
            standard_correct = standard_preds.eq(targets).sum().item()
            
            # Calculate standard loss
            standard_loss = criterion(standard_outputs, targets)
            
            # Only do IBP forward pass if we're using IBP loss
            if current_epsilon > 0:
                # Create interval bounds for inputs
                lower = torch.clamp(inputs - current_epsilon, min=0)
                upper = torch.clamp(inputs + current_epsilon, max=1)
                ibp_inputs = torch.cat([lower, upper], dim=1)
                
                # IBP forward pass
                ibp_bounds = ibp_model(ibp_inputs)
                
                # Split output into lower and upper bounds
                out_size = ibp_bounds.size(1) // 2
                ibp_lower = ibp_bounds[:, :out_size]
                ibp_upper = ibp_bounds[:, out_size:]
                
                # Check if the lower bound of the target class is greater than the upper bounds of all other classes
                ibp_correct = 0
                for i in range(batch_size):
                    target = targets[i]
                    target_lower = ibp_lower[i, target]
                    # Get upper bounds for all classes except target
                    other_classes_upper = torch.cat([ibp_upper[i, :target], ibp_upper[i, target+1:]])
                    # Check if target's lower bound exceeds all other classes' upper bounds
                    if target_lower > other_classes_upper.max():
                        ibp_correct += 1
                
                # Calculate IBP loss (using lower bounds)
                ibp_loss = compute_ibp_loss(criterion, ibp_lower, ibp_upper, targets, device)
                
                # Combined loss with weighting
                loss = (1 - ibp_weight) * standard_loss + ibp_weight * ibp_loss
            else:
                # Only use standard loss in early epochs
                loss = standard_loss
                ibp_correct = 0
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * batch_size
            train_standard_acc += standard_correct
            train_verified_acc += ibp_correct
            total += batch_size
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_standard_acc = 100.0 * train_standard_acc / total
        train_verified_acc = 100.0 * train_verified_acc / total
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_standard_acc)
        history['train_verified_acc'].append(train_verified_acc)
        
        # Validation phase
        if val_loader is not None:
            ibp_model.eval()
            val_loss = 0.0
            val_standard_acc = 0.0
            val_verified_acc = 0.0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    batch_size = inputs.size(0)
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Standard forward pass
                    standard_outputs = ibp_model.standard_forward(inputs)
                    _, standard_preds = standard_outputs.max(1)
                    standard_correct = standard_preds.eq(targets).sum().item()
                    
                    # Calculate standard loss
                    standard_loss = criterion(standard_outputs, targets)
                    
                    # IBP evaluation if epsilon > 0
                    if current_epsilon > 0:
                        # Create interval bounds for inputs
                        lower = torch.clamp(inputs - current_epsilon, min=0)
                        upper = torch.clamp(inputs + current_epsilon, max=1)
                        ibp_inputs = torch.cat([lower, upper], dim=1)
                        
                        # IBP forward pass
                        ibp_bounds = ibp_model(ibp_inputs)
                        
                        # Split output into lower and upper bounds
                        out_size = ibp_bounds.size(1) // 2
                        ibp_lower = ibp_bounds[:, :out_size]
                        ibp_upper = ibp_bounds[:, out_size:]
                        
                        # Check if the lower bound of the target class is greater than the upper bounds of all other classes
                        ibp_correct = 0
                        for i in range(batch_size):
                            target = targets[i]
                            target_lower = ibp_lower[i, target]
                            # Get upper bounds for all classes except target
                            other_classes_upper = torch.cat([ibp_upper[i, :target], ibp_upper[i, target+1:]])
                            # Check if target's lower bound exceeds all other classes' upper bounds
                            if target_lower > other_classes_upper.max():
                                ibp_correct += 1
                        
                        # Calculate IBP loss (using lower bounds)
                        ibp_loss = compute_ibp_loss(criterion, ibp_lower, ibp_upper, targets, device)
                        
                        # Combined loss
                        loss = (1 - ibp_weight) * standard_loss + ibp_weight * ibp_loss
                    else:
                        loss = standard_loss
                        ibp_correct = 0
                    
                    # Track statistics
                    val_loss += loss.item() * batch_size
                    val_standard_acc += standard_correct
                    val_verified_acc += ibp_correct
                    total += batch_size
            
            # Calculate validation metrics
            val_loss = val_loss / len(val_loader.dataset)
            val_standard_acc = 100.0 * val_standard_acc / total
            val_verified_acc = 100.0 * val_verified_acc / total
            
            # Save metrics
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_standard_acc)
            history['val_verified_acc'].append(val_verified_acc)
            
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}, ε={current_epsilon:.5f}, IBP Weight={ibp_weight:.2f}')
                print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_standard_acc:.2f}%, Verified: {train_verified_acc:.2f}%')
                print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_standard_acc:.2f}%, Verified: {val_verified_acc:.2f}%')
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}, ε={current_epsilon:.5f}, IBP Weight={ibp_weight:.2f}')
                print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_standard_acc:.2f}%, Verified: {train_verified_acc:.2f}%')
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
    
    # Convert back to standard model before returning
    standard_model = convert_ibp_to_standard(ibp_model)
    
    return standard_model, history

def train_sabr(model, train_loader, val_loader=None, optimizer=None, criterion=None,
             num_epochs=100, device='cuda', scheduler=None, verbose=True,
             start_epsilon=0.0, end_epsilon=0.3, epsilon_schedule_length=50,
             standard_epochs=20, ibp_weight_start=0.0, ibp_weight_end=1.0,
             tau=0.1, adv_steps=10, adv_step_size=0.05):
    """
    
    Args:
        model: PyTorch model to train (will be wrapped with IBPWrapper if not already)
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        optimizer: PyTorch optimizer (if None, uses Adam with default params)
        criterion: Loss function (if None, uses CrossEntropyLoss)
        num_epochs: Number of training epochs
        device: Device to run training on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
        verbose: Whether to print progress
        start_epsilon: Initial perturbation radius
        end_epsilon: Final perturbation radius
        epsilon_schedule_length: Number of epochs over which to scale epsilon
        standard_epochs: Number of epochs to train with standard loss only
        ibp_weight_start: Initial weight for IBP loss component
        ibp_weight_end: Final weight for IBP loss component
        tau: Size of box around adversarial example (as fraction of epsilon)
        adv_steps: Number of steps for PGD attack
        adv_step_size: Step size for PGD attack
        
    Returns:
        model: Trained standard model (unwrapped from IBPWrapper)
        history: Dictionary containing training metrics
    """
    # Check if model is wrapped with IBPWrapper, if not, wrap it
    if not isinstance(model, IBPWrapper):
        ibp_model = IBPWrapper(model, device)
    else:
        ibp_model = model
    
    # Move model to device
    ibp_model = ibp_model.to(device)
    
    # Default optimizer and loss function if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(ibp_model.parameters())
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_verified_acc': [],
        'val_verified_acc': [],
        'epsilon': []
    }
    
    for epoch in range(num_epochs):
        # Determine current epsilon based on epoch
        if epoch < standard_epochs:
            current_epsilon = 0.0
            ibp_weight = 0.0
        else:
            # Scale epsilon and IBP weight according to schedule
            schedule_step = min(epoch - standard_epochs, epsilon_schedule_length) / epsilon_schedule_length
            current_epsilon = start_epsilon + (end_epsilon - start_epsilon) * schedule_step
            ibp_weight = ibp_weight_start + (ibp_weight_end - ibp_weight_start) * schedule_step
        
        history['epsilon'].append(current_epsilon)
        
        # Training phase
        ibp_model.train()
        train_loss = 0.0
        train_standard_acc = 0.0
        train_verified_acc = 0.0
        total = 0
        
        for inputs, targets in train_loader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Standard forward pass for clean accuracy
            standard_outputs = ibp_model.standard_forward(inputs)
            _, standard_preds = standard_outputs.max(1)
            standard_correct = standard_preds.eq(targets).sum().item()
            
            # Calculate standard loss
            standard_loss = criterion(standard_outputs, targets)
            
            # Only do adversarial training and IBP if we're past standard epochs
            if current_epsilon > 0:
                # Generate adversarial examples with PGD
                #with torch.no_grad():
                adv_inputs = pgd_attack(ibp_model.model, inputs, targets, 
                                        epsilon=(1-tau)*current_epsilon, 
                                        step_size=adv_step_size, 
                                        num_iterations=adv_steps, 
                                        device=device)
                
                # Create small box around adversarial examples using tau*epsilon
                box_size = tau * current_epsilon
                lower = torch.clamp(adv_inputs - box_size, min=0)
                upper = torch.clamp(adv_inputs + box_size, max=1)
                ibp_inputs = torch.cat([lower, upper], dim=1)
                
                # IBP forward pass on boxes around adversarial examples
                ibp_bounds = ibp_model(ibp_inputs)
                
                # Split output into lower and upper bounds
                out_size = ibp_bounds.size(1) // 2
                ibp_lower = ibp_bounds[:, :out_size]
                ibp_upper = ibp_bounds[:, out_size:]
                
                # Check if the lower bound of the target class is greater than the upper bounds of all other classes
                ibp_correct = 0
                for i in range(batch_size):
                    target = targets[i]
                    target_lower = ibp_lower[i, target]
                    # Get upper bounds for all classes except target
                    other_classes_upper = torch.cat([ibp_upper[i, :target], ibp_upper[i, target+1:]])
                    # Check if target's lower bound exceeds all other classes' upper bounds
                    if target_lower > other_classes_upper.max():
                        ibp_correct += 1
                
                # Calculate IBP loss (using lower bounds)
                ibp_loss = compute_ibp_loss(criterion, ibp_lower, ibp_upper, targets, device)
                
                # Combined loss with weighting
                loss = (1 - ibp_weight) * standard_loss + ibp_weight * ibp_loss
            else:
                # Only use standard loss in early epochs
                loss = standard_loss
                ibp_correct = 0
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item() * batch_size
            train_standard_acc += standard_correct
            train_verified_acc += ibp_correct
            total += batch_size
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader.dataset)
        train_standard_acc = 100.0 * train_standard_acc / total
        train_verified_acc = 100.0 * train_verified_acc / total
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_standard_acc)
        history['train_verified_acc'].append(train_verified_acc)
        
        # Validation phase
        if val_loader is not None:
            ibp_model.eval()
            val_loss = 0.0
            val_standard_acc = 0.0
            val_verified_acc = 0.0
            total = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    batch_size = inputs.size(0)
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Standard forward pass
                    standard_outputs = ibp_model.standard_forward(inputs)
                    _, standard_preds = standard_outputs.max(1)
                    standard_correct = standard_preds.eq(targets).sum().item()
                    
                    # Calculate standard loss
                    standard_loss = criterion(standard_outputs, targets)
                    
                    # IBP evaluation if epsilon > 0
                    if current_epsilon > 0:
                        box_size = current_epsilon
                        lower = torch.clamp(inputs - box_size, min=0)
                        upper = torch.clamp(inputs + box_size, max=1)
                        ibp_inputs = torch.cat([lower, upper], dim=1)
                        
                        # IBP forward pass
                        ibp_bounds = ibp_model(ibp_inputs)
                        
                        # Split output into lower and upper bounds
                        out_size = ibp_bounds.size(1) // 2
                        ibp_lower = ibp_bounds[:, :out_size]
                        ibp_upper = ibp_bounds[:, out_size:]
                        
                        # Check if the lower bound of the target class is greater than the upper bounds of all other classes
                        ibp_correct = 0
                        for i in range(batch_size):
                            target = targets[i]
                            target_lower = ibp_lower[i, target]
                            # Get upper bounds for all classes except target
                            other_classes_upper = torch.cat([ibp_upper[i, :target], ibp_upper[i, target+1:]])
                            # Check if target's lower bound exceeds all other classes' upper bounds
                            if target_lower > other_classes_upper.max():
                                ibp_correct += 1
                        
                        # Calculate IBP loss (using lower bounds)
                        ibp_loss = compute_ibp_loss(criterion, ibp_lower, ibp_upper, targets, device)
                        
                        # Combined loss
                        loss = (1 - ibp_weight) * standard_loss + ibp_weight * ibp_loss
                    else:
                        loss = standard_loss
                        ibp_correct = 0
                    
                    # Track statistics
                    val_loss += loss.item() * batch_size
                    val_standard_acc += standard_correct
                    val_verified_acc += ibp_correct
                    total += batch_size
            
            # Calculate validation metrics
            val_loss = val_loss / len(val_loader.dataset)
            val_standard_acc = 100.0 * val_standard_acc / total
            val_verified_acc = 100.0 * val_verified_acc / total
            
            # Save metrics
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_standard_acc)
            history['val_verified_acc'].append(val_verified_acc)
            
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}, ε={current_epsilon:.5f}, IBP Weight={ibp_weight:.2f}')
                print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_standard_acc:.2f}%, Verified: {train_verified_acc:.2f}%')
                print(f'  Val   - Loss: {val_loss:.4f}, Acc: {val_standard_acc:.2f}%, Verified: {val_verified_acc:.2f}%')
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}, ε={current_epsilon:.5f}, IBP Weight={ibp_weight:.2f}')
                print(f'  Train - Loss: {train_loss:.4f}, Acc: {train_standard_acc:.2f}%, Verified: {train_verified_acc:.2f}%')
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
    
    # Convert back to standard model before returning
    standard_model = convert_ibp_to_standard(ibp_model)
    
    return standard_model, history

def ibp_verified_batch_accuracy(model, inputs, targets, epsilon=0.3, device='cuda'):
    """
    Calculate verified accuracy for a single batch using IBP
    
    Args:
        model: PyTorch model (will be wrapped with IBPWrapper if not already)
        inputs: Input tensors (batch)
        targets: Target labels (batch)
        epsilon: Perturbation radius for IBP verification
        device: Device to run evaluation on
        
    Returns:
        verified_accuracy: Batch verified accuracy (percentage)
    """
    # Check if model is wrapped with IBPWrapper, if not, wrap it
    if not isinstance(model, IBPWrapper):
        model = IBPWrapper(model, device)
    
    model.eval()
    inputs = inputs.to(device)
    targets = targets.to(device)
    
    with torch.no_grad():
        # Create interval bounds for inputs
        lower = torch.clamp(inputs - epsilon, min=0)
        upper = torch.clamp(inputs + epsilon, max=1)
        ibp_inputs = torch.cat([lower, upper], dim=1)
        
        # IBP forward pass
        ibp_bounds = model(ibp_inputs)
        
        # Split output into lower and upper bounds
        out_size = ibp_bounds.size(1) // 2
        ibp_lower = ibp_bounds[:, :out_size]
        ibp_upper = ibp_bounds[:, out_size:]
        
        # Check if the lower bound of the target class is greater than the upper bounds of all other classes
        batch_size = targets.size(0)
        ibp_correct = 0
        
        for i in range(batch_size):
            target = targets[i]
            target_lower = ibp_lower[i, target]
            # Get upper bounds for all classes except target
            other_classes_upper = torch.cat([ibp_upper[i, :target], ibp_upper[i, target+1:]])
            # Check if target's lower bound exceeds all other classes' upper bounds
            if target_lower > other_classes_upper.max():
                ibp_correct += 1
        
        total = targets.size(0)
    
    verified_accuracy = 100.0 * ibp_correct / total
    return verified_accuracy

def test_verified_accuracy(model, test_loader, epsilon=0.3, device='cuda'):
    """
    Test the verified accuracy of a model on a test dataset using IBP
    
    Args:
        model: PyTorch model (will be wrapped with IBPWrapper if not already)
        test_loader: DataLoader for test set
        epsilon: Perturbation radius for IBP verification
        device: Device to run evaluation on
        
    Returns:
        verified_accuracy: Test verified accuracy
    """
    # Check if model is wrapped with IBPWrapper, if not, wrap it
    if not isinstance(model, IBPWrapper):
        model = IBPWrapper(model, device)
    
    model.eval()
    model = model.to(device)
    
    verified_correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Create interval bounds for inputs
            lower = torch.clamp(inputs - epsilon, min=0)
            upper = torch.clamp(inputs + epsilon, max=1)
            ibp_inputs = torch.cat([lower, upper], dim=1)
            
            # IBP forward pass
            ibp_bounds = model(ibp_inputs)
            
            # Split output into lower and upper bounds
            out_size = ibp_bounds.size(1) // 2
            ibp_lower = ibp_bounds[:, :out_size]
            ibp_upper = ibp_bounds[:, out_size:]
            
            # Check if the lower bound of the target class is greater than the upper bounds of all other classes
            batch_size = targets.size(0)
            
            for i in range(batch_size):
                target = targets[i]
                target_lower = ibp_lower[i, target]
                # Get upper bounds for all classes except target
                other_classes_upper = torch.cat([ibp_upper[i, :target], ibp_upper[i, target+1:]])
                # Check if target's lower bound exceeds all other classes' upper bounds
                if target_lower > other_classes_upper.max():
                    verified_correct += 1
            
            total += targets.size(0)
            
    verified_accuracy = 100.0 * verified_correct / total
    return verified_accuracy

def compute_ibp_loss(criterion, ibp_lower, ibp_upper, targets, device='cuda'):
    """
    Computes the IBP loss that tries to maximize the lower bound of the target class
    and minimize the upper bound of all other classes.
    
    Args:
        criterion: Loss criterion to use (e.g., nn.CrossEntropyLoss)
        ibp_lower: Tensor of lower bounds from IBP, shape [batch_size, num_classes]
        ibp_upper: Tensor of upper bounds from IBP, shape [batch_size, num_classes]
        targets: Tensor of target labels, shape [batch_size]
        device: Device to run computations on
        
    Returns:
        loss: The IBP loss value
    """
    batch_size = targets.size(0)
    num_classes = ibp_lower.size(1)
    
    # Create one-hot encoding for target classes
    lb_mask = torch.zeros(batch_size, num_classes, device=device)
    lb_mask.scatter_(1, targets.unsqueeze(1), 1)
    
    # Create mask for non-target classes
    hb_mask = 1 - lb_mask
    
    # Create combined output using lower bounds for target class and upper bounds for non-target classes
    # This pushes the lower bound of the target class up and the upper bounds of other classes down
    outputs = ibp_lower * lb_mask + ibp_upper * hb_mask
    
    # Compute loss
    loss = criterion(outputs, targets)
    
    return loss

def ensemble_cert_train(model_paths, train_loader, val_loader=None, criterion=None,
                      num_epochs=100, device='cuda', verbose=True,
                      epsilon=0.3, tau=0.1, adv_steps=10, adv_step_size=0.05, learning_rate=0.001):
    """
    Ensemble Certification Training
    
    Trains an ensemble of models where each model is trained on adversarial examples
    generated for all OTHER models in the ensemble, wrapped in small SABR boxes.
    
    Args:
        model_paths: List of paths to ONNX model files
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        criterion: Loss function (if None, uses CrossEntropyLoss)
        num_epochs: Number of training epochs
        device: Device to run training on ('cuda' or 'cpu')
        verbose: Whether to print progress
        epsilon: Perturbation radius for adversarial examples
        tau: Size of box around adversarial example (as fraction of epsilon)
        adv_steps: Number of steps for PGD attack
        adv_step_size: Step size for PGD attack
        learning_rate: Learning rate for optimizers
        
    Returns:
        models: List of trained models
        history: Dictionary containing training metrics for each model
    """
    
    models = []
    ibp_models = []
    optimizers = []
    
    for path in model_paths:
        # Use the updated load_onnx_to_pytorch method from network_utils
        pytorch_model = network_utils.load_onnx_to_pytorch(path, device)
        
        # Get a sample input from the training data
        sample_inputs, _ = next(iter(train_loader))
        sample_input = sample_inputs[0:1].to(device)  # Just use one sample
        
        # Verify that the loaded PyTorch model matches the ONNX model
        is_equivalent, _, _ = network_utils.verify_loaded_model(
            pytorch_model, path, sample_input, rtol=1e-3, atol=1e-4
        )
        
        # Assert that models are equivalent
        assert is_equivalent, f"Loaded PyTorch model does not match ONNX model at {path}"
        
        # Wrap with IBP
        ibp_model = IBPWrapper(pytorch_model, device)
        ibp_model.to(device)
        
        # Create optimizer
        optimizer = torch.optim.Adam(ibp_model.parameters(), lr=learning_rate)
        
        models.append(pytorch_model)
        ibp_models.append(ibp_model)
        optimizers.append(optimizer)
    
    num_models = len(models)
    
    # Default loss function if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Track metrics
    history = {i: {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_verified_acc': [],
        'val_verified_acc': [],
    } for i in range(num_models)}
    
    for epoch in range(num_epochs):
        # Training phase
        for ibp_model in ibp_models:
            ibp_model.train()
        
        model_train_stats = [{
            'loss': 0.0,
            'standard_acc': 0.0,
            'verified_acc': 0.0,
            'total': 0
        } for _ in range(num_models)]
        
        for inputs, targets in train_loader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Standard forward pass and loss for each model
            std_outputs = []
            std_preds = []
            std_corrects = []
            std_losses = []
            
            for i, ibp_model in enumerate(ibp_models):
                outputs = ibp_model.standard_forward(inputs)
                _, preds = outputs.max(1)
                correct = preds.eq(targets).sum().item()
                loss = criterion(outputs, targets)
                
                std_outputs.append(outputs)
                std_preds.append(preds)
                std_corrects.append(correct)
                std_losses.append(loss)
            
            # Generate adversarial examples for each model
            adv_inputs_list = []
            for i, model in enumerate(models):
                adv_inputs = pgd_attack(model, inputs, targets, 
                                      epsilon=(1-tau)*epsilon, 
                                      step_size=adv_step_size, 
                                      num_iterations=adv_steps, 
                                      device=device)
                adv_inputs_list.append(adv_inputs)
            
            # Create small boxes around adversarial examples
            box_size = tau * epsilon
            adv_boxes_list = []
            
            for adv_inputs in adv_inputs_list:
                lower = torch.clamp(adv_inputs - box_size, min=0)
                upper = torch.clamp(adv_inputs + box_size, max=1)
                adv_boxes = torch.cat([lower, upper], dim=1)
                adv_boxes_list.append(adv_boxes)
            
            # Train each model on other models' adversarial boxes
            for i, (ibp_model, optimizer) in enumerate(zip(ibp_models, optimizers)):
                optimizer.zero_grad()
                
                # Collect adversarial boxes from other models
                other_boxes = []
                other_targets = []
                
                for j in range(num_models):
                    if i != j:  # Skip the current model's own adversarial examples
                        other_boxes.append(adv_boxes_list[j])
                        other_targets.append(targets)
                
                # Concatenate all other models' boxes and targets
                if other_boxes:
                    combined_boxes = torch.cat(other_boxes, dim=0)
                    combined_targets = torch.cat(other_targets, dim=0)
                    
                    # IBP forward pass on boxes
                    ibp_bounds = ibp_model(combined_boxes)
                    
                    # Split output into lower and upper bounds
                    out_size = ibp_bounds.size(1) // 2
                    ibp_lower = ibp_bounds[:, :out_size]
                    ibp_upper = ibp_bounds[:, out_size:]
                    
                    # Check if the lower bound of the target class is greater than the upper bounds of all other classes
                    combined_batch_size = combined_targets.size(0)
                    ibp_correct = 0
                    
                    for k in range(combined_batch_size):
                        target = combined_targets[k]
                        target_lower = ibp_lower[k, target]
                        # Get upper bounds for all classes except target
                        other_classes_upper = torch.cat([ibp_upper[k, :target], ibp_upper[k, target+1:]])
                        # Check if target's lower bound exceeds all other classes' upper bounds
                        if target_lower > other_classes_upper.max():
                            ibp_correct += 1
                    
                    # Calculate IBP loss (using lower bounds)
                    ibp_loss = compute_ibp_loss(criterion, ibp_lower, ibp_upper, combined_targets, device)
                    
                    # Use IBP loss directly without weighting
                    loss = ibp_loss
                else:
                    # If there are no other models, just use standard loss
                    loss = std_losses[i]
                    ibp_correct = 0
                    combined_batch_size = 0
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Track statistics
                model_train_stats[i]['loss'] += loss.item() * batch_size
                model_train_stats[i]['standard_acc'] += std_corrects[i]
                if combined_batch_size > 0:
                    model_train_stats[i]['verified_acc'] += ibp_correct
                model_train_stats[i]['total'] += batch_size
        
        # Calculate epoch metrics for each model
        for i in range(num_models):
            train_loss = model_train_stats[i]['loss'] / model_train_stats[i]['total']
            train_standard_acc = 100.0 * model_train_stats[i]['standard_acc'] / model_train_stats[i]['total']
            
            if model_train_stats[i]['total'] > 0:
                train_verified_acc = 100.0 * model_train_stats[i]['verified_acc'] / (model_train_stats[i]['total'] * (num_models - 1))
            else:
                train_verified_acc = 0.0
            
            # Save metrics
            history[i]['train_loss'].append(train_loss)
            history[i]['train_acc'].append(train_standard_acc)
            history[i]['train_verified_acc'].append(train_verified_acc)
        
        # Validation phase
        if val_loader is not None:
            for ibp_model in ibp_models:
                ibp_model.eval()
            
            model_val_stats = [{
                'loss': 0.0,
                'standard_acc': 0.0,
                'verified_acc': 0.0,
                'total': 0
            } for _ in range(num_models)]
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    batch_size = inputs.size(0)
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    for i, ibp_model in enumerate(ibp_models):
                        # Standard forward pass
                        standard_outputs = ibp_model.standard_forward(inputs)
                        _, standard_preds = standard_outputs.max(1)
                        standard_correct = standard_preds.eq(targets).sum().item()
                        
                        # Calculate standard loss
                        standard_loss = criterion(standard_outputs, targets)
                        
                        # IBP evaluation with full epsilon
                        box_size = epsilon
                        lower = torch.clamp(inputs - box_size, min=0)
                        upper = torch.clamp(inputs + box_size, max=1)
                        ibp_inputs = torch.cat([lower, upper], dim=1)
                        
                        # IBP forward pass
                        ibp_bounds = ibp_model(ibp_inputs)
                        
                        # Split output into lower and upper bounds
                        out_size = ibp_bounds.size(1) // 2
                        ibp_lower = ibp_bounds[:, :out_size]
                        ibp_upper = ibp_bounds[:, out_size:]
                        
                        # Check if the lower bound of the target class is greater than the upper bounds of all other classes
                        ibp_correct = 0
                        for j in range(batch_size):
                            target = targets[j]
                            target_lower = ibp_lower[j, target]
                            # Get upper bounds for all classes except target
                            other_classes_upper = torch.cat([ibp_upper[j, :target], ibp_upper[j, target+1:]])
                            # Check if target's lower bound exceeds all other classes' upper bounds
                            if target_lower > other_classes_upper.max():
                                ibp_correct += 1
                        
                        # Calculate IBP loss (using lower bounds)
                        ibp_loss = compute_ibp_loss(criterion, ibp_lower, ibp_upper, targets, device)
                        
                        # Use IBP loss directly for evaluation
                        loss = ibp_loss
                        
                        # Track statistics
                        model_val_stats[i]['loss'] += loss.item() * batch_size
                        model_val_stats[i]['standard_acc'] += standard_correct
                        model_val_stats[i]['verified_acc'] += ibp_correct
                        model_val_stats[i]['total'] += batch_size
            
            # Calculate validation metrics for each model
            for i in range(num_models):
                val_loss = model_val_stats[i]['loss'] / model_val_stats[i]['total']
                val_standard_acc = 100.0 * model_val_stats[i]['standard_acc'] / model_val_stats[i]['total']
                val_verified_acc = 100.0 * model_val_stats[i]['verified_acc'] / model_val_stats[i]['total']
                
                # Save metrics
                history[i]['val_loss'].append(val_loss)
                history[i]['val_acc'].append(val_standard_acc)
                history[i]['val_verified_acc'].append(val_verified_acc)
            
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}')
                for i in range(num_models):
                    print(f'  Model {i+1}:')
                    print(f'    Train - Loss: {history[i]["train_loss"][-1]:.4f}, Acc: {history[i]["train_acc"][-1]:.2f}%, Verified: {history[i]["train_verified_acc"][-1]:.2f}%')
                    print(f'    Val   - Loss: {history[i]["val_loss"][-1]:.4f}, Acc: {history[i]["val_acc"][-1]:.2f}%, Verified: {history[i]["val_verified_acc"][-1]:.2f}%')
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}')
                for i in range(num_models):
                    print(f'  Model {i+1}:')
                    print(f'    Train - Loss: {history[i]["train_loss"][-1]:.4f}, Acc: {history[i]["train_acc"][-1]:.2f}%, Verified: {history[i]["train_verified_acc"][-1]:.2f}%')
    
    # Convert back to standard models before returning
    standard_models = [convert_ibp_to_standard(ibp_model) for ibp_model in ibp_models]
    
    return standard_models, history

def train_threshold_network(existing_model, threshold_model, train_loader, val_loader=None,
                           optimizer=None, criterion=None, num_epochs=100, device='cuda',
                           scheduler=None, verbose=True, epsilon=0.3, n_samples=5, p_norm=float('inf')):
    """
    Trains a threshold network for conformal prediction.
    
    The threshold network has the same architecture as the existing model but with a single output.
    Training uses IBP to bound both networks and minimizes the distance between the upper bound
    of the threshold network and the lower bound of the target class of the existing model.
    Additionally, samples n random points in the L_p ball around the input and trains the
    threshold network to match the output of the existing model's target class for these points.
    
    Args:
        existing_model: Existing classifier model (fixed, not trained)
        threshold_model: Threshold network to be trained (single output)
        train_loader: DataLoader for training data
        val_loader: Optional DataLoader for validation data
        optimizer: PyTorch optimizer (if None, uses Adam with default params)
        criterion: Loss function (if None, uses MSELoss)
        num_epochs: Number of training epochs
        device: Device to run training on ('cuda' or 'cpu')
        scheduler: Optional learning rate scheduler
        verbose: Whether to print progress
        epsilon: Perturbation radius
        n_samples: Number of random samples to take in the L_p ball
        p_norm: The p-norm to use for the perturbation ball (default: inf)
        
    Returns:
        threshold_model: Trained threshold model
        history: Dictionary containing training metrics
    """
    # Wrap both models with IBP if not already wrapped
    if not isinstance(existing_model, IBPWrapper):
        ibp_existing_model = IBPWrapper(existing_model, device)
    else:
        ibp_existing_model = existing_model
    
    if not isinstance(threshold_model, IBPWrapper):
        ibp_threshold_model = IBPWrapper(threshold_model, device)
    else:
        ibp_threshold_model = threshold_model
    
    # Move models to device
    ibp_existing_model = ibp_existing_model.to(device)
    ibp_threshold_model = ibp_threshold_model.to(device)
    
    # Freeze the existing model
    for param in ibp_existing_model.parameters():
        param.requires_grad = False
    
    # Default optimizer and loss function if not provided
    if optimizer is None:
        optimizer = torch.optim.Adam(ibp_threshold_model.parameters())
    if criterion is None:
        criterion = nn.MSELoss()
    
    # Track metrics
    history = {
        'train_loss': [],
        'train_ibp_loss': [],
        'train_sample_loss': [],
        'val_loss': [],
        'val_ibp_loss': [],
        'val_sample_loss': []
    }
    
    for epoch in range(num_epochs):
        # Training phase
        ibp_existing_model.eval()  # Evaluation mode for existing model
        ibp_threshold_model.train()  # Training mode for threshold model
        
        train_loss = 0.0
        train_ibp_loss = 0.0
        train_sample_loss = 0.0
        total_batches = 0
        
        for inputs, targets in train_loader:
            batch_size = inputs.size(0)
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Create interval bounds for inputs
            lower = torch.clamp(inputs - epsilon, min=0)
            upper = torch.clamp(inputs + epsilon, max=1)
            ibp_inputs = torch.cat([lower, upper], dim=1)
            
            # IBP forward pass for existing model
            with torch.no_grad():
                existing_bounds = ibp_existing_model(ibp_inputs)
                
                # Split output into lower and upper bounds
                existing_out_size = existing_bounds.size(1) // 2
                existing_lower = existing_bounds[:, :existing_out_size]
                existing_upper = existing_bounds[:, existing_out_size:]
                
                # Get lower bounds of target classes
                target_lower_bounds = torch.gather(existing_lower, 1, targets.unsqueeze(1))
            
            # IBP forward pass for threshold model
            threshold_bounds = ibp_threshold_model(ibp_inputs)
            
            # Split output into lower and upper bounds
            threshold_out_size = threshold_bounds.size(1) // 2
            threshold_lower = threshold_bounds[:, :threshold_out_size]
            threshold_upper = threshold_bounds[:, threshold_out_size:]
            
            # IBP loss: minimize distance between threshold upper bound and target class lower bound
            ibp_loss = criterion(threshold_upper, target_lower_bounds)
            
            # Sample n random points in the L_p ball and compute sample loss
            sample_loss = 0.0
            for _ in range(n_samples):
                # Generate random perturbation within epsilon ball
                if p_norm == float('inf'):
                    # For L-infinity norm
                    delta = torch.zeros_like(inputs, device=device)
                    delta.uniform_(-epsilon, epsilon)
                else:
                    # For other L_p norms
                    delta = torch.randn_like(inputs, device=device)
                    delta_norm = torch.norm(delta.view(batch_size, -1), p=p_norm, dim=1)
                    delta_norm = delta_norm.view(batch_size, 1, 1, 1)
                    delta = delta * epsilon / delta_norm
                
                # Ensure perturbed inputs are within [0, 1]
                perturbed_inputs = torch.clamp(inputs + delta, 0, 1)
                
                # Forward pass with perturbed inputs for both models
                with torch.no_grad():
                    existing_outputs = ibp_existing_model.standard_forward(perturbed_inputs)
                    # Get the output of the target class
                    target_outputs = torch.gather(existing_outputs, 1, targets.unsqueeze(1))
                
                threshold_outputs = ibp_threshold_model.standard_forward(perturbed_inputs)
                
                # Sample loss: match threshold output to target class output
                sample_loss += criterion(threshold_outputs, target_outputs)
            
            sample_loss /= n_samples
            
            # Combined loss
            loss = ibp_loss + sample_loss
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track statistics
            train_loss += loss.item()
            train_ibp_loss += ibp_loss.item()
            train_sample_loss += sample_loss.item()
            total_batches += 1
        
        # Calculate epoch metrics
        train_loss /= total_batches
        train_ibp_loss /= total_batches
        train_sample_loss /= total_batches
        
        # Save metrics
        history['train_loss'].append(train_loss)
        history['train_ibp_loss'].append(train_ibp_loss)
        history['train_sample_loss'].append(train_sample_loss)
        
        # Validation phase
        if val_loader is not None:
            ibp_existing_model.eval()
            ibp_threshold_model.eval()
            
            val_loss = 0.0
            val_ibp_loss = 0.0
            val_sample_loss = 0.0
            total_batches = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    batch_size = inputs.size(0)
                    inputs, targets = inputs.to(device), targets.to(device)
                    
                    # Create interval bounds for inputs
                    lower = torch.clamp(inputs - epsilon, min=0)
                    upper = torch.clamp(inputs + epsilon, max=1)
                    ibp_inputs = torch.cat([lower, upper], dim=1)
                    
                    # IBP forward pass for existing model
                    existing_bounds = ibp_existing_model(ibp_inputs)
                    
                    # Split output into lower and upper bounds
                    existing_out_size = existing_bounds.size(1) // 2
                    existing_lower = existing_bounds[:, :existing_out_size]
                    existing_upper = existing_bounds[:, existing_out_size:]
                    
                    # Get lower bounds of target classes
                    target_lower_bounds = torch.gather(existing_lower, 1, targets.unsqueeze(1))
                    
                    # IBP forward pass for threshold model
                    threshold_bounds = ibp_threshold_model(ibp_inputs)
                    
                    # Split output into lower and upper bounds
                    threshold_out_size = threshold_bounds.size(1) // 2
                    threshold_lower = threshold_bounds[:, :threshold_out_size]
                    threshold_upper = threshold_bounds[:, threshold_out_size:]
                    
                    # IBP loss: minimize distance between threshold upper bound and target class lower bound
                    ibp_loss = criterion(threshold_upper, target_lower_bounds)
                    
                    # Sample n random points in the L_p ball and compute sample loss
                    sample_loss = 0.0
                    for _ in range(n_samples):
                        # Generate random perturbation within epsilon ball
                        if p_norm == float('inf'):
                            # For L-infinity norm
                            delta = torch.zeros_like(inputs, device=device)
                            delta.uniform_(-epsilon, epsilon)
                        else:
                            # For other L_p norms
                            delta = torch.randn_like(inputs, device=device)
                            delta_norm = torch.norm(delta.view(batch_size, -1), p=p_norm, dim=1)
                            delta_norm = delta_norm.view(batch_size, 1, 1, 1)
                            delta = delta * epsilon / delta_norm
                        
                        # Ensure perturbed inputs are within [0, 1]
                        perturbed_inputs = torch.clamp(inputs + delta, 0, 1)
                        
                        # Forward pass with perturbed inputs for both models
                        existing_outputs = ibp_existing_model.standard_forward(perturbed_inputs)
                        # Get the output of the target class
                        target_outputs = torch.gather(existing_outputs, 1, targets.unsqueeze(1))
                        
                        threshold_outputs = ibp_threshold_model.standard_forward(perturbed_inputs)
                        
                        # Sample loss: match threshold output to target class output
                        sample_loss += criterion(threshold_outputs, target_outputs)
                    
                    sample_loss /= n_samples
                    
                    # Combined loss
                    loss = ibp_loss + sample_loss
                    
                    # Track statistics
                    val_loss += loss.item()
                    val_ibp_loss += ibp_loss.item()
                    val_sample_loss += sample_loss.item()
                    total_batches += 1
            
            # Calculate validation metrics
            val_loss /= total_batches
            val_ibp_loss /= total_batches
            val_sample_loss /= total_batches
            
            # Save metrics
            history['val_loss'].append(val_loss)
            history['val_ibp_loss'].append(val_ibp_loss)
            history['val_sample_loss'].append(val_sample_loss)
            
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}')
                print(f'  Train - Loss: {train_loss:.4f}, IBP Loss: {train_ibp_loss:.4f}, Sample Loss: {train_sample_loss:.4f}')
                print(f'  Val   - Loss: {val_loss:.4f}, IBP Loss: {val_ibp_loss:.4f}, Sample Loss: {val_sample_loss:.4f}')
        else:
            if verbose:
                print(f'Epoch {epoch+1}/{num_epochs}')
                print(f'  Train - Loss: {train_loss:.4f}, IBP Loss: {train_ibp_loss:.4f}, Sample Loss: {train_sample_loss:.4f}')
        
        # Step scheduler if provided
        if scheduler is not None:
            scheduler.step()
    
    # Convert back to standard model before returning
    standard_threshold_model = convert_ibp_to_standard(ibp_threshold_model)
    
    return standard_threshold_model, history
