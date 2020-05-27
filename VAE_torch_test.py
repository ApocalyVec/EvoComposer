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

    def __init__(self, input_dim, hidden_dim, z_dim):
        '''
        Args:
            input_dim: A integer indicating the size of input (in case of MNIST 28 * 28).
            hidden_dim: A integer indicating the size of hidden_dim dimension.
            z_dim: A integer indicating the latent dimension.
        '''
        super().__init__()

        self.linear = nn.Linear(input_dim, hidden_dim)
        self.mu = nn.Linear(hidden_dim, z_dim)
        self.var = nn.Linear(hidden_dim, z_dim)

    def forward(self, x):
        # hidden_decoder is of shape [batch_size, input_dim]

        hidden = F.relu(self.linear(x))
        # hidden_dim is of shape [batch_size, hidden_dim]
        z_mu = self.mu(hidden)
        # z_mu is of shape [batch_size, latent_dim]
        z_var = self.var(hidden)
        # z_var is of shape [batch_size, latent_dim]

        return z_mu, z_var


class Decoder(nn.Module):
    ''' This the decoder part of VAE

    '''
    def __init__(self, z_dim, hidden_dim, output_dim):
        '''
        Args:
            z_dim: A integer indicating the latent size.
            hidden_dim: A integer indicating the size of hidden_dim dimension.
            output_dim: A integer indicating the output dimension (in case of MNIST it is 28 * 28)
        '''
        super().__init__()

        self.linear = nn.Linear(z_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # hidden_decoder is of shape [batch_size, latent_dim]

        hidden = F.relu(self.linear(x))
        # hidden_dim is of shape [batch_size, hidden_dim]

        predicted = torch.sigmoid(self.out(hidden))
        # predicted is of shape [batch_size, output_dim]

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
        z_mu, z_var = self.enc(x)

        # sample from the distribution having latent parameters z_mu, z_var
        # reparameterize
        std = torch.exp(z_var / 2)
        eps = torch.randn_like(std)
        x_sample = eps.mul(std).add_(z_mu)

        # decode
        predicted = self.dec(x_sample)
        return predicted, z_mu, z_var


def train(X, batch_size, loss_func):
    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for i in range(0, len(X), batch_size):
        # reshape the data into [batch_size, 784]
        x = torch.FloatTensor(X[i:i + batch_size])
        x = x.to(device)

        # update the gradients to zero
        optimizer.zero_grad()

        # forward pass
        x_sample, z_mu, z_var = model(x)

        # reconstruction loss
        recon_loss = loss_func(x_sample, x)

        # kl divergence loss
        kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

        # total loss
        loss = recon_loss + kl_loss

        # backward pass
        loss.backward()
        train_loss += loss.item()

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

            # forward pass
            x_sample, z_mu, z_var = model(x)

            # reconstruction loss
            recon_loss = loss_func(x_sample, x)

            # kl divergence loss
            kl_loss = 0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1.0 - z_var)

            # total loss
            loss = recon_loss + kl_loss
            test_loss += loss.item()

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
LATENT_DIM = 20  # latent vector dimension
lr = 1e-3  # learning rate
criterion = nn.SmoothL1Loss()

# train_iterator = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
# test_iterator = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# encoder
encoder = Encoder(INPUT_DIM, HIDDEN_DIM, LATENT_DIM)

# decoder
decoder = Decoder(LATENT_DIM, HIDDEN_DIM, INPUT_DIM)

# vae
model = VAE(encoder, decoder).to(device)

# optimizer
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

# # sample and generate a image
# z = torch.randn(1, LATENT_DIM).to(device)
#
# # run only the decoder
# reconstructed_img = model.dec(z)
# img = reconstructed_img.view(28, 28).data
#
# print(z.shape)
# print(img.shape)
#
# plt.imshow(img.cpu(), cmap='gray')
# plt.show()
