import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class Transformer(nn.Module):
    """Transformer-based model for time series classification.
    
    Args:
        input_dim (int): Number of input features
        num_classes (int): Number of output classes
        d_model (int): Dimension of model embeddings
        n_head (int): Number of attention heads
        n_layers (int): Number of transformer layers
        d_inner (int): Dimension of inner feedforward layer
        activation (str): Activation function ('relu' or 'gelu')
        dropout (float): Dropout probability
        max_len (int): Maximum sequence length for positional encoding
        max_seq_len (int): Maximum sequence length of input
        T (int): Temperature parameter for positional encoding
        max_temporal_shift (int): Maximum temporal shift for positional encoding
    """
    def __init__(self, input_dim=15, num_classes=2, d_model=128, n_head=16, n_layers=4, d_inner=128,
                 activation="relu", dropout=0.2, max_len=366, max_seq_len=70, T=1000, max_temporal_shift=30):
        super(Transformer, self).__init__()
        self.modelname = self._get_name()
        self.max_seq_len = max_seq_len

        # Shared encoder (MLP for feature extraction)
        self.mlp_dim = [input_dim, 32, 64, d_model]
        layers = []
        for i in range(len(self.mlp_dim) - 1):
            layers.append(linlayer(self.mlp_dim[i], self.mlp_dim[i + 1]))
        self.mlp1 = nn.Sequential(*layers)

        self.inlayernorm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Positional encoding for temporal information
        self.position_enc = PositionalEncoding(d_model, max_len=max_len + 2 * max_temporal_shift, T=T)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(d_model, n_head, d_inner, dropout, activation)
        encoder_norm = nn.LayerNorm(d_model)
        self.transformerencoder = nn.TransformerEncoder(encoder_layer, n_layers, encoder_norm)

        # Classification decoder
        layers = []
        decoder = [d_model, 64, 32, num_classes]
        for i in range(len(decoder) - 1):
            layers.append(nn.Linear(decoder[i], decoder[i + 1]))
            if i < (len(decoder) - 2):
                layers.extend([
                    nn.BatchNorm1d(decoder[i + 1]),
                    nn.ReLU()
                ])
        layers.append(nn.Softmax(dim=-1))
        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, seq_len, input_dim)
            
        Returns:
            torch.Tensor: Output probabilities of shape (batch, num_classes)
        """
        b, s, c = x.shape

        # Feature extraction through MLP
        x = x.permute((0, 2, 1))
        x = self.mlp1(x)
        x = x.permute((0, 2, 1))

        # Layer normalization and positional encoding
        x = self.inlayernorm(x)
        src_pos = torch.arange(1, self.max_seq_len + 1, dtype=torch.long).expand(b, s).cuda().to(x.device)
        x = self.dropout(x + self.position_enc(src_pos))
        
        # Transformer processing (requires seq_len first)
        x = x.transpose(0, 1)  # N x T x D -> T x N x D
        x = self.transformerencoder(x)
        x = x.transpose(0, 1)  # T x N x D -> N x T x D

        # Temporal average pooling and classification
        x = ((x.permute((2, 0, 1))).sum(-1) / x.size(-1)).permute(1, 0)
        classification_logits = self.decoder(x)

        return classification_logits

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
        with torch.no_grad():
            for batch in loader:
                x = batch[0].to(device)
                outputs = self.forward(x)
                classificationResults.extend(outputs.cpu().numpy())

        return np.array(classificationResults)


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models.
    
    Args:
        d_model (int): Dimension of model embeddings
        max_len (int): Maximum sequence length
        T (int): Temperature parameter controlling frequency
    """
    def __init__(self, d_model: int, max_len: int = 5000, T: int = 10000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(T) / d_model))
        pe = torch.zeros(max_len + 1, d_model)
        pe[1:, 0::2] = torch.sin(position * div_term)  # Even indices
        pe[1:, 1::2] = torch.cos(position * div_term)  # Odd indices
        self.register_buffer('pe', pe)

    def forward(self, doy):
        """Add positional encoding to input.
        
        Args:
            doy: Tensor of shape [batch_size, seq_len] containing day-of-year indices
        """
        return self.pe[doy]


class linlayer(nn.Module):
    """Linear layer with batch norm and ReLU activation.
    
    Args:
        in_dim (int): Input dimension
        out_dim (int): Output dimension
    """
    def __init__(self, in_dim, out_dim):
        super(linlayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.lin = nn.Linear(in_dim, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, input):
        """Forward pass through the linear layer."""
        out = input.permute((0, 2, 1))  # Switch to channel last for linear layer
        out = self.lin(out)
        out = out.permute((0, 2, 1))  # Switch back to channel first for batch norm
        out = self.bn(out)
        out = F.relu(out)
        return out