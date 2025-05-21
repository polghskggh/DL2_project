import torch.nn as nn
import torch.nn.functional as F

from einops.layers.torch import Rearrange
from .classification_module import ClassificationModule

class LSTM(ClassificationModule):
    def __init__(
            self,
            n_channels,
            lstm_size,
            hidden_size,
            n_classes,
            dropout=0.5,
            **kwargs
    ):
        # Create internal model
        model = LSTMCore(n_channels, lstm_size, hidden_size, n_classes, dropout)

        # Pass it to the base ClassificationModule
        super().__init__(model=model, n_classes=n_classes, **kwargs)


class LSTMCore(nn.Module):
    """
    Internal model used by LSTM wrapper that defines actual LSTM + FC layers.
    """

    def __init__(self, n_channels, lstm_size, hidden_size, n_classes, dropout=0.25):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=lstm_size, batch_first=True, bidirectional=True)

        self.rearrange_input = Rearrange("b c t -> b t c")
        self.dropout = nn.Dropout(dropout)
        
        self.bn1 = nn.BatchNorm1d(2 * lstm_size)
        self.fc1 = nn.Linear(2 * lstm_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)
        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # Input: (batch, n_input, time)

        x = self.rearrange_input(x)
        lstm_out, _ = self.lstm(x)  # h_n: (1, batch, lstm_size)
        series_embed = lstm_out.amax(dim=1)  # (batch, lstm_size)
        series_embed = self.bn1(series_embed)
        series_embed = self.dropout(series_embed)

        fc1_out = self.fc1(series_embed)
        fc1_out = self.bn2(fc1_out)
        fc1_out = F.softplus(fc1_out)
        fc1_out = self.dropout(fc1_out)

        logits = self.fc2(fc1_out)  # (batch, n_classes)
        return logits
