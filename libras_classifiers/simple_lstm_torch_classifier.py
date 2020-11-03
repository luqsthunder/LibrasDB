import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from libras_classifiers.librasdb_loaders import DBLoader2NPY
import torch.optim as optim

import numpy as np

torch.manual_seed(6)


class LSTMClassifier(nn.Module):
    hidden = None

    def __init__(self, amount_features, amount_timesteps, amount_classes):
        super(LSTMClassifier, self).__init__()

        self.amount_features = amount_features
        self.seq_len = amount_timesteps
        self.n_hidden = 80
        self.n_layers = 1
        self.lstm = nn.LSTM(
            input_size=amount_features,
            batch_first=True,
            hidden_size=self.n_hidden,
            num_layers=self.n_layers
        )
        self.dense = nn.Linear(self.n_hidden * self.seq_len, amount_classes)

    def init_hidden(self, batchsize):
        # even with batch_first = True this remains same as docs
        hidden_state = torch.zeros(self.n_layers, batchsize, self.n_hidden)
        cell_state = torch.zeros(self.n_layers, batchsize, self.n_hidden)
        self.hidden = (hidden_state, cell_state)

    def forward(self, input_data):
        x = input_data.view(8, self.seq_len, self.amount_features)
        batch_size, seq_len, _ = x.size()

        lstm_out, self.hidden = self.lstm(x, self.hidden)

        x = lstm_out.contiguous().view(batch_size, -1)
        return F.softmax(self.dense(x))


if __name__ == '__main__':
    batch_size = 8
    epochs = 20
    db = DBLoader2NPY('../libras-db-folders', batch_size=batch_size,
                      no_hands=False, angle_pose=False)
    db.fill_samples_absent_frames_with_na()
    input_dim = (len(db.joints_used()) - 1) * 2

    model = LSTMClassifier(input_dim, db.find_longest_sample(), db.amount_classes())
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    loss = None

    for epoch in range(20):  # again, normally you would NOT do 300 epochs, it is toy data
        for x_batch, y_batch, sample_weigth in db:
            # Step 1. Remember that Pytorch accumulates gradients.
            # We need to clear them out before each instance
            model.zero_grad()

            # Step 2. Get our inputs ready for the network, that is, turn them into
            # Tensors of word indices.

            # Step 3. Run our forward pass.
            x = torch.from_numpy(np.stack(x_batch, axis=0).reshape((8, 90, 61 * 2))).float()
            y = torch.from_numpy(np.array([np.argmax(y) for y in y_batch])).long()

            model.init_hidden(x.size(0))

            scores = model(x)

            # Step 4. Compute the loss, gradients, and update the parameters by
            #  calling optimizer.step()
            loss = loss_function(scores, y)
            loss.backward()
            optimizer.step()
        print('step : ', epoch, 'loss : ', loss.item())
