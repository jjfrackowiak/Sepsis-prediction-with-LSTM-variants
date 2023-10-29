import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

BDLSTM2_lookback_kernel = {
    10: [1,1,1],
    20: [2,3,2],
    50: [5,9,5],
    100: [12,17,12],
}


class BDLSTM2(pl.LightningModule):

    def __init__(self, input_size=40, lookback=10, filters1=64, filters2=64, filters3=64, hidden_size=40, dropout=0.3, lr=1e-3):
        super().__init__()
        self.lr = lr
        self.hidden_size = hidden_size
        self.conv1d1 = nn.Conv1d(input_size, filters1, kernel_size=(BDLSTM2_lookback_kernel[lookback][0], ))
        self.maxpool1 = nn.MaxPool1d(kernel_size=(2, ))
        self.conv1d2 = nn.Conv1d(filters1, filters2, kernel_size=(BDLSTM2_lookback_kernel[lookback][1], ))
        self.maxpool2 = nn.MaxPool1d(kernel_size=(2, ))
        self.conv1d3 = nn.Conv1d(filters2, filters3, kernel_size=(BDLSTM2_lookback_kernel[lookback][2], ))
        self.maxpool3 = nn.MaxPool1d(kernel_size=(2, ))
        self.lstm1 = nn.LSTM(1, hidden_size, num_layers=1, batch_first=True,
                            bidirectional=True, dropout=dropout)
       #delete
        self.ff1 = nn.Linear(hidden_size, 1)

    def init_hidden(self, x, num_layers, batch_size, hidden_size):
        self.hidden = (torch.zeros(num_layers * (1 + 1 * True), batch_size, hidden_size).type_as(x),
                       torch.zeros(num_layers * (1 + 1 * True), batch_size, hidden_size).type_as(x))

    def forward(self, x):
        x = x.permute(0,2,1)
        self.init_hidden(x, 1, x.shape[0], self.hidden_size)

        x = self.conv1d1(x)  # 97 - 3 = 94
        x = self.maxpool1(x)  # 94 / 2 = 47
        x = self.conv1d2(x)
        x = self.maxpool2(x)
        x = self.conv1d3(x)
        x = self.maxpool3(x)

        lstm_out, self.hidden = self.lstm1(x, self.hidden)
        # lstm_out, self.hidden = self.lstm2(lstm_out, self.hidden)

        predictions = self.ff1(self.hidden[0][-1, :, :])

        return predictions, self.hidden

    def training_step(self, batch, batch_idx):
        x, y = batch

        output, hidden = self(x)
        pos_weight = torch.tensor(10)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(output, y)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def predict_step(self, batch, batch_idx):
        x, y = batch

        output, hidden = self(x)

        return self.trainer.datamodule.y_scaler.inverse_transform(output.cpu().numpy())
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        output, hidden = self(x)
        
        pos_weight = torch.tensor(10)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(output, y)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
