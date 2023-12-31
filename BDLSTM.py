import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchmetrics


class BDLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, stateless=True):
        super(BDLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.stateless = stateless
        self.lr = lr

        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def init_hidden(self, x, num_layers, batch_size, hidden_size):
        self.hidden = (Variable(torch.zeros(num_layers + 1, batch_size, hidden_size).type_as(x)),
                       Variable(torch.zeros(num_layers + 1, batch_size, hidden_size).type_as(x)))

    def forward(self, x):
        if self.stateless:
            self.init_hidden(x, 1, x.shape[0], self.hidden_size)

        self.hidden[0].detach_()
        self.hidden[1].detach_()
        hidden1 = self.hidden[0].detach()
        hidden2 = self.hidden[1].detach()

        lstm_out, self.hidden = self.lstm(x, (hidden1, hidden2))
        predictions = self.linear(self.hidden[0][-1, :, :])

        return predictions, self.hidden

    def training_step(self, batch, batch_idx):
        x, y = batch

        #transforming batch in case of NaN values

        if batch_idx == 0:
            self.init_hidden(x, 1, x.shape[0], self.hidden_size)

        output, hidden = self(x)
        pos_weight = torch.tensor(10)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        loss = criterion(output, y)
        
        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        return loss

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


