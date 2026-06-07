import torch.nn as nn


class LSTM_HAR(nn.Module):
    def __init__(self, input_size=6, hidden_size=128, num_classes=6):
        super().__init__() #runs parent's (nn.Module) __init__ code.

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.3
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.dropout(out)
        out = self.fc(out)
        return out