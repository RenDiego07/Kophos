"""
BiLSTM Architecture with Attention Mechanism for ASL Recognition
Input: Tensor of shape (Batch_Size, Sequence_Length, Feature_Dim)
Output: Logits of shape (Batch_Size, Num_Classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Computes a weighted sum of the BiLSTM output sequence.
    Allows the model to focus on the most important frames of the sign.
    """
    def __init__(self, hidden_dim: int):
        super(Attention, self).__init__()
        # Multiplied by 2 because the LSTM is bidirectional
        self.attention = nn.Linear(hidden_dim * 2, 1)

    def forward(self, lstm_outputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # lstm_outputs shape: (Batch, Seq_Len, Hidden_Dim * 2)
        
        # Calculate attention scores
        attn_scores = self.attention(lstm_outputs) # Shape: (Batch, Seq_Len, 1)
        
        # Normalize scores using softmax to get probabilities that sum to 1
        attn_weights = F.softmax(attn_scores, dim=1) # Shape: (Batch, Seq_Len, 1)
        
        # Multiply weights by original outputs and sum across the sequence dimension
        # context_vector shape: (Batch, Hidden_Dim * 2)
        context_vector = torch.sum(attn_weights * lstm_outputs, dim=1)
        
        return context_vector, attn_weights


class BiLSTMSignModel(nn.Module):
    """
    Complete Neural Network Architecture for Sign Language Classification.
    """
    def __init__(
        self, 
        input_dim: int = 225, 
        hidden_dim: int = 128, 
        num_layers: int = 2, 
        num_classes: int = 100, 
        dropout: float = 0.3
    ):
        super(BiLSTMSignModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 1. Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True
        )

        # 2. Attention Layer
        self.attention = Attention(hidden_dim)

        # 3. Fully Connected Classifier Layers
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.
        
        Args:
            x: Tensor of shape (Batch, Seq_Len, Input_Dim)
               e.g., (32, 30, 225)
               
        Returns:
            out: Unnormalized log probabilities (logits) of shape (Batch, Num_Classes)
        """
        # Pass through BiLSTM
        # lstm_out shape: (Batch, Seq_Len, Hidden_Dim * 2)
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Pass through Attention Mechanism
        context_vector, attn_weights = self.attention(lstm_out)
        
        # Pass through Classifier
        out = self.fc1(context_vector)
        out = self.relu(out)
        out = self.dropout_layer(out)
        out = self.fc2(out)
        
        return out


# ============================================================
# TESTING ROUTINE
# ============================================================
if __name__ == "__main__":
    # Simulate a single batch of data: 4 videos, 30 frames, 225 features
    dummy_input = torch.randn(4, 30, 225)
    
    print("Initialize Model...")
    model = BiLSTMSignModel(input_dim=225, hidden_dim=128, num_classes=100)
    
    print("Running Forward Pass...")
    outputs = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape} -> (Batch, Seq_Len, Features)")
    print(f"Output Shape: {outputs.shape} -> (Batch, Num_Classes)")
    
    if outputs.shape == (4, 100):
        print("✅ Architecture test passed successfully!")