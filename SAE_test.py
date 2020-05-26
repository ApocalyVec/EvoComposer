import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

from utils.MIDI_utils import load_samples

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Encoder(nn.Module):
    ''' This the encoder part of VAE

    '''

    def __init__(self, input_len, hidden_dim, latent_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden_dim dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear1 = nn.Linear(input_len, hidden_dim)  # first fully connected layer
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

        self.activation = nn.Sigmoid()

    def forward(self, x):
        hidden1 = self.activation(self.linear1(x))  # encode with the first layer
        latent = self.activation(self.linear2(hidden1))  # encode with the second layer
        return latent


class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, latent_dim, hidden_dim, input_len):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden_dim dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear1 = nn.Linear(latent_dim, hidden_dim)  # start of the decoding
        self.out = nn.Linear(hidden_dim, input_len)  # output of the decoder

    def forward(self, x):
        hidden = F.relu(self.linear1(x))
        predicted = torch.sigmoid(self.out(hidden))

        return predicted


class VAE(nn.Module):
    ''' This the VAE, which takes a encoder and decoder.

    '''
    def __init__(self, enc, dec):
        super().__init__()

        self.enc = enc
        self.dec = dec

    def forward(self, x):
        # encode
        latent = self.enc(x)
        # decode
        predicted = self.dec(latent)
        return predicted


def train(X, batch_size, loss_func):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i in range(0, len(X), batch_size):
        # reshape the data into [batch_size, 784]
        x = torch.FloatTensor(X[i:i + batch_size])
        x = x.to(device)
        x_target = x.clone()
        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_pred = model(x)

        # reconstruction loss
        recon_loss = loss_func(x_pred, x_target)
        # recon_loss = F.binary_cross_entropy(x_pred, x_target, reduction='sum')
        # backward pass
        recon_loss.backward()
        train_loss += recon_loss.item()

        # update the weights
        optimizer.step()

    return train_loss

def test(X, batch_size, loss_func):
    # set the evaluation mode
    model.eval()

    # test loss for the data
    test_loss = 0

    # we don't need to track the gradients, since we are not updating the parameters during evaluation / testing
    with torch.no_grad():
        for i in range(0, len(X), batch_size):
            # reshape the data
            x = torch.FloatTensor(X[i:i + batch_size])
            x = x.to(device)
            x_target = x.clone()

            # forward pass
            x_pred = model(x)

            # reconstruction loss
            recon_loss = loss_func(x_pred, x_target)
            # recon_loss = F.binary_cross_entropy(x_pred, x_target, reduction='sum')

            test_loss += recon_loss.item()

    return test_loss

# Script body ##########################################################################################

data_dir = 'data/schubert'
timesteps = 32
f_threshold = 50

x_tr, x_val, y_tr, y_val, unique_x, unique_y = load_samples(data_dir, timesteps, f_threshold, _use_spark=True)

# a very good writeup on loss functions:
# https://medium.com/udacity-pytorch-challengers/a-brief-overview-of-loss-functions-in-pytorch-c0ddb78068f7
BATCH_SIZE = 64  # number of data points in each batch
N_EPOCHS = 10  # times to run the model on complete data
INPUT_DIM = timesteps  # size of each input
HIDDEN_DIM = 256  # hidden_dim dimension
LATENT_DIM = 128  # latent vector dimension
lr = 1e-2  # learning rate
criterion = nn.SmoothL1Loss()

# train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)
model = VAE(encoder, decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr)

best_test_loss = float('inf')

for e in range(N_EPOCHS):

    train_loss = train(X=x_tr, batch_size=BATCH_SIZE, loss_func=criterion)
    test_loss = test(X=x_val, batch_size=BATCH_SIZE, loss_func=criterion)

    train_loss /= len(x_tr)
    test_loss /= len(x_val)

    print(f'Epoch {e}, Train Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}')

    # if best_test_loss > test_loss:
    #     best_test_loss = test_loss
    #     patience_counter = 1
    # else:
    #     patience_counter += 1
    #
    # if patience_counter > 3:
    #     break