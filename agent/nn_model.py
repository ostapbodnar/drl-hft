import torch
import torch.nn as nn


class CnnLstmTwoHeadNN(nn.Module):
    def __init__(self, num_classes, device):
        super(CnnLstmTwoHeadNN, self).__init__()

        # Head 1
        self.kline_cnn = nn.Sequential(
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )
        self.kline_lstm = nn.LSTM(input_size=1600, hidden_size=512, num_layers=3, batch_first=True)

        # Head 2
        self.lob_cnn = nn.Sequential(
            nn.Conv1d(in_channels=110, out_channels=128, kernel_size=5, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=5, stride=3),
        )
        self.lob_lstm = nn.LSTM(input_size=3968, hidden_size=512, num_layers=3, batch_first=True)

        self.device = device
        self.num_classes = num_classes

        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, observation):
        kline, lob = observation[:, :, :16], observation[:, :,16:]
        # Head 1
        cnn_out1 = self.kline_cnn(kline.permute(0, 2, 1))
        cnn_out1 = cnn_out1.reshape(cnn_out1.shape[0], 1, cnn_out1.shape[1] * cnn_out1.shape[2])
        _, (lstm_out1, _) = self.kline_lstm(cnn_out1)

        # Head 2
        cnn_out2 = self.lob_cnn(lob.permute(0, 2, 1))
        cnn_out2 = cnn_out2.reshape(cnn_out2.shape[0], 1, cnn_out2.shape[1] * cnn_out2.shape[2])
        _, (lstm_out2, _) = self.lob_lstm(cnn_out2)

        # Concatenate LSTM outputs
        lstm_out = torch.cat((lstm_out1[-1], lstm_out2[-1]), dim=-1)

        # MLP
        output = self.mlp(lstm_out)

        if self.num_classes != 1:
            output = self.softmax(output)

        return output
