import torch
import torch.nn as nn
import torch.utils.data
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TempCNN(nn.Module):
    """Temporal CNN model for sequence classification.
    
    Args:
        input_dim (int): Number of input features
        num_classes (int): Number of output classes
        hidden_dims (int): Dimension of hidden layers
        kernel_size (int): Size of convolutional kernels
        dropout (float): Dropout probability
        max_seq_len (int): Maximum sequence length
    """
    def __init__(self, input_dim=15, num_classes=2, hidden_dims=128, 
                 kernel_size=7, dropout=0.2, max_seq_len=70):
        super(TempCNN, self).__init__()
        self.modelname = self._get_name()

        # Three convolutional blocks with batch norm and dropout
        self.conv_bn_relu1 = Conv1D_BatchNorm_Relu_Dropout(
            input_dim, hidden_dims, kernel_size=kernel_size,
            drop_probability=dropout)
        self.conv_bn_relu2 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size,
            drop_probability=dropout)
        self.conv_bn_relu3 = Conv1D_BatchNorm_Relu_Dropout(
            hidden_dims, hidden_dims, kernel_size=kernel_size,
            drop_probability=dropout)

        # Fully connected layers
        self.flatten = Flatten()
        self.dense = FC_BatchNorm_Relu_Dropout(
            hidden_dims * max_seq_len, 4 * hidden_dims, 
            drop_probability=dropout)
        self.decoder = nn.Linear(4 * hidden_dims, num_classes)
        self.softmax = nn.Softmax(dim=1)  # Convert outputs to probabilities

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output probabilities of shape (batch, num_classes)
        """
        # Transpose to (batch, input_dim, seq_len) for Conv1D
        x = x.transpose(1, 2)
        
        # Apply convolutional blocks
        x = self.conv_bn_relu1(x)
        x = self.conv_bn_relu2(x)
        x = self.conv_bn_relu3(x)

        # Flatten and pass through dense layers
        x = self.flatten(x)
        x = self.dense(x)
        x = self.decoder(x)
        x = self.softmax(x)
        return x

    def predict_proba(self, data):
        """Generate class probability predictions for input data.
        
        Args:
            data: Input data (numpy array or torch.Tensor)
            
        Returns:
            numpy.ndarray: Predicted class probabilities
        """
        self.eval()  # Set model to evaluation mode
        
        # Get the device the model is currently on
        device = next(self.parameters()).device  
        self.to(device)  # Ensure model is on the correct device

        # Convert numpy arrays to PyTorch tensors if needed
        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        # Convert to Dataset if input is a tensor
        if isinstance(data, torch.Tensor):
            data = TensorDataset(data)

        # Create dataloader for batch processing
        loader = DataLoader(data, batch_size=4096, shuffle=False)

        classificationResults = []
        with torch.no_grad():  # Disable gradient calculation
            for batch in loader:
                x = batch[0].to(device)
                outputs = self.forward(x)
                classificationResults.extend(outputs.cpu().numpy())

        return np.array(classificationResults)


class Conv1D_BatchNorm_Relu_Dropout(torch.nn.Module):
    """1D Convolutional block with batch norm, ReLU and dropout.
    
    Args:
        input_dim (int): Number of input channels
        hidden_dims (int): Number of output channels
        kernel_size (int): Size of convolutional kernel
        drop_probability (float): Dropout probability
    """
    def __init__(self, input_dim, hidden_dims, kernel_size=5, drop_probability=0.5):
        super(Conv1D_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dims, kernel_size, padding=(kernel_size // 2)),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        """Forward pass through the convolutional block."""
        return self.block(X)


class Flatten(nn.Module):
    """Flatten layer to convert spatial dimensions to features."""
    def forward(self, input):
        return input.view(input.size(0), -1)


class FC_BatchNorm_Relu_Dropout(torch.nn.Module):
    """Fully connected block with batch norm, ReLU and dropout.
    
    Args:
        input_dim (int): Number of input features
        hidden_dims (int): Number of output features
        drop_probability (float): Dropout probability
    """
    def __init__(self, input_dim, hidden_dims, drop_probability=0.5):
        super(FC_BatchNorm_Relu_Dropout, self).__init__()

        self.block = nn.Sequential(
            nn.Linear(input_dim, hidden_dims),
            nn.BatchNorm1d(hidden_dims),
            nn.ReLU(),
            nn.Dropout(p=drop_probability)
        )

    def forward(self, X):
        """Forward pass through the fully connected block."""
        return self.block(X)