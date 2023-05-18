import torch
import torch.nn as nn


class CnnLstmTwoHeadNN(nn.Module):
    def __init__(self, mlp_hidden_size, num_classes, device):
        super(CnnLstmTwoHeadNN, self).__init__()

        # Head 1
        self.kline_cnn = nn.Conv1d(in_channels=5, out_channels=64, kernel_size=3, padding=1)
        self.kline_lstm = nn.LSTM(input_size=64, hidden_size=64, num_layers=2, batch_first=True)

        # Head 2
        self.lob_cnn = nn.Conv1d(in_channels=110, out_channels=256, kernel_size=3, padding=1)
        self.lob_lstm = nn.LSTM(input_size=256, hidden_size=256, num_layers=2, batch_first=True)

        self.device = device

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(256 + 64, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, num_classes)
        )

    def forward(self, kline, lob):
        # Head 1
        cnn_out1 = self.kline_cnn(kline)
        cnn_out1 = cnn_out1.permute(0, 2, 1)  # Reshaping for LSTM
        _, (lstm_out1, _) = self.kline_lstm(cnn_out1)

        # Head 2
        cnn_out2 = self.lob_cnn(lob)
        cnn_out2 = cnn_out2.permute(0, 2, 1)  # Reshaping for LSTM
        _, (lstm_out2, _) = self.lob_lstm(cnn_out2)

        # Concatenate LSTM outputs
        lstm_out = torch.cat((lstm_out1[-1], lstm_out2[-1]), dim=-1)

        # MLP
        output = self.mlp(lstm_out)

        return output


class CnnLstmTwoHeadNNAgent(CnnLstmTwoHeadNN):
    def __init__(self, mlp_hidden_size, num_classes, device):
        super().__init__(mlp_hidden_size, num_classes, device)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, kline, lob):
        output = super().forward(kline, lob)
        return self.softmax(output)


class CnnLstmTwoHeadNNCritic(CnnLstmTwoHeadNN):
    def __init__(self, mlp_hidden_size, num_classes, device):
        super().__init__(mlp_hidden_size, num_classes, device)

        self.client_linear = nn.Linear(num_classes, 1)

    def forward(self, kline, lob):
        output = super().forward(kline, lob)
        return self.client_linear(output)
