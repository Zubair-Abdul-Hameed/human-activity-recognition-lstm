import torch.nn as nn


class LSTM_HAR(nn.Module):
    def __init__(self, input_size=6, hidden_size=64, num_classes=6):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            batch_first=True
        )

        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = h_n[-1]
        out = self.dropout(out)
        out = self.fc(out)
        return out