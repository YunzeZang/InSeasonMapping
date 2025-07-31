import torch
import torch.optim as optim
from deepModel.DCM import *
import os
from deepModel.TempCNN import *
from deepModel.Transformer import *
from torch.utils.data import random_split
import random
from torch.optim.lr_scheduler import ReduceLROnPlateau


def seed_torch(seed=2025):
    """
    Sets random seed for reproducibility across multiple libraries.
    
    Args:
        seed (int): Random seed value (default: 2025)
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU setups
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = True

def train(model, train_loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    
    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimization algorithm
        device: Device to run computations on (e.g., 'cuda:0')
        
    Returns:
        float: Average training loss for the epoch
    """
    model.train()  # Set the model to training mode
    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        # Skip batches with only one sample (potential batch norm issues)
        if inputs.shape[0] == 1:
            continue
            
        # Reset gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Compute loss
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Accumulate loss
        running_loss += loss.item()

    return running_loss / len(train_loader)

def validate(model, valid_loader, criterion, device):
    """
    Validates the model on the validation set.
    
    Args:
        model: The neural network model
        valid_loader: DataLoader for validation data
        criterion: Loss function
        device: Device to run computations on
        
    Returns:
        tuple: (average validation loss, accuracy percentage)
    """
    model.eval()  # Set model to evaluation mode
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation
        for inputs, targets in valid_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            if inputs.shape[0] == 1:
                continue
                
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()

    avg_loss = running_loss / len(valid_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy

def DL_predict(model, valid_loader, device):
    """
    Generates predictions using the trained model.
    
    Args:
        model: Trained neural network model
        valid_loader: DataLoader for prediction data
        device: Device to run computations on
        
    Returns:
        torch.Tensor: Model predictions (class probabilities)
    """
    model.eval()
    outputs_list = []

    with torch.no_grad():
        for inputs in valid_loader:
            inputs = inputs[0].to(device)
            outputs = model(inputs)
            # Extract probability for class 1 (binary classification)
            outputs = [prob[1] for prob in outputs]
            outputs_list += outputs

    return torch.tensor(outputs_list)

def loadModel(DLModelType, modelParameter, checkPath='', device='cuda:0'):
    """
    Initializes or loads a deep learning model with its components.
    
    Args:
        DLModelType (str): Type of model ('DCM', 'TempCNN', or 'Transformer')
        modelParameter (dict): Parameters for model initialization
        checkPath (str): Path to checkpoint file (optional)
        device (str): Device to load model onto
        
    Returns:
        tuple: (model, optimizer, scheduler, criterion)
    """
    # Initialize model based on type
    if DLModelType == 'DCM':
        model = DCM(**modelParameter)
    elif DLModelType == 'TempCNN':
        model = TempCNN(**modelParameter)
    elif DLModelType == 'Transformer':
        model = Transformer(**modelParameter)
    else:
        raise ValueError(f"Unknown model type: {DLModelType}")

    # Initialize training components
    optimizer = optim.Adam(model.parameters(), lr=0.00005)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, min_lr=1e-6)
    criterion = nn.CrossEntropyLoss()

    # Load checkpoint if available
    if os.path.exists(checkPath):
        checkpoint = torch.load(checkPath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint['epoch']
        print(f"Restored model from epoch {start_epoch}")

    model.to(device)
    return model, optimizer, scheduler, criterion


# Training configuration constants
batch_size = 16
num_epochs = 500
patience = 50  # Early stopping patience
DL_model_list = ['Transformer', 'DCM', 'TempCNN']  # Supported deep learning models
ML_model_list = ['RF', 'SVM']  # Supported machine learning models
random_seed = 2025

# Initialize random seeds for reproducibility
seed_torch(seed=random_seed)