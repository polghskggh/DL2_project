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

    def __init__(self, n_channels, lstm_size, hidden_size, n_classes, dropout):
        super().__init__()
        self.lstm = nn.LSTM(input_size=n_channels, hidden_size=lstm_size, batch_first=True)

        self.rearrange_input = Rearrange("b c t -> b t c")
        self.fc1 = nn.Linear(lstm_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.fc2 = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # Input: (batch, n_input, time)

        x = self.rearrange_input(x)
        lstm_out, (h_n, _) = self.lstm(x)  # h_n: (1, batch, lstm_size)
        final_hidden = h_n[-1]  # (batch, lstm_size)

        fc1_out = self.fc1(final_hidden)
        fc1_out = self.bn1(fc1_out)
        fc1_out = F.softplus(fc1_out)
        fc1_out = self.dropout(fc1_out)

        logits = self.fc2(fc1_out)  # (batch, n_classes)
        return logits
