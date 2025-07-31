import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np


class DCM(nn.Module):
    """Deep Crop Model (DCM) with LSTM and attention mechanism for crop classification.
    
    Args:
        seed (int): Random seed for reproducibility
        input_dim (int): Dimension of input features
        hidden_size (int): Size of LSTM hidden layers
        num_layers (int): Number of LSTM layers
        bidirectional (bool): Whether to use bidirectional LSTM
        dropout (float): Dropout probability
        num_classes (int): Number of output classes
        max_seq_len (int): Maximum sequence length
    """
    def __init__(self, seed=42, input_dim=15, hidden_size=128, num_layers=8, 
                 bidirectional=True, dropout=0.2, num_classes=2, max_seq_len=70):
        super().__init__()
        self._set_reproducible(seed)

        # LSTM layer for temporal feature extraction
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout,
        )
        
        # Attention mechanism to weight important time steps
        num_directions = 2 if bidirectional else 1
        self.attention = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=1,
        )
        
        # Final classification layer
        self.fc = nn.Linear(
            in_features=num_directions * hidden_size,
            out_features=num_classes,
        )

    def _set_reproducible(self, seed, cudnn=False):
        """Set random seeds for reproducibility."""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cudnn:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output probabilities of shape (batch, num_classes)
        """
        self.lstm.flatten_parameters()
        
        # LSTM output: (batch, seq_len, num_directions*hidden_size)
        lstm_out, _ = self.lstm(x)
        
        # Compute attention weights using softmax along sequence length
        attn_weights = F.softmax(F.relu(self.attention(lstm_out)), dim=1)
        
        # Apply attention weights to LSTM outputs
        fc_in = attn_weights.permute(0, 2, 1).bmm(lstm_out)
        fc_out = self.fc(fc_in)
        
        # Convert to probabilities using softmax
        prob = F.softmax(fc_out, dim=-1)
        return prob.squeeze()

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