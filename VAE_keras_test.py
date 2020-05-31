from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K
import tensorflow as tf
from tensorflow.keras.losses import sparse_categorical_crossentropy, categorical_crossentropy, binary_crossentropy

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os

from tensorflow.python.keras.backend import mean
from tensorflow.python.keras.utils.vis_utils import plot_model

from utils.MIDI_utils import load_sample_unsupervised


def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = tf.random.normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

# script body ##########################################################################################################
data_dir = 'data/schubert'
input_timesteps = 64
f_threshold = 50

# TODO: to min-max normalize or not normalize, this is a problem
x_train, x_test, one_hot_encoder = load_sample_unsupervised(data_dir, input_timesteps, f_threshold, _use_spark=False)

melody_dim = x_train.shape[1:]
original_dim = np.prod(melody_dim)
num_classes = x_train.shape[-1]
intermediate_dim = 32
batch_size = 128

latent_steps = 4
latent_dim = 2
epochs = 50

# VAE model = encoder + decoder
# build encoder model
inputs = tf.keras.Input(shape=(input_timesteps, num_classes), name='encoder_input')
# x = tf.keras.layers.Reshape(target_shape=(input_timesteps, num_classes))(inputs)
x = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(intermediate_dim))(inputs)
z_mean = tf.keras.layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = tf.keras.layers.Dense(latent_dim, name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = tf.keras.layers.Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = tf.keras.Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()
plot_model(encoder, to_file='figs/vae_mlp_encoder.png', show_shapes=True)

# build decoder layers
latent_inputs = tf.keras.layers.Input(shape=(latent_dim, ), name='z_sampling')
# x = tf.keras.layers.Dense(input_timesteps, activation='relu')(latent_inputs)
# x = tf.keras.layers.Reshape(target_shape=(input_timesteps, 1))(x)
x = tf.keras.layers.RepeatVector(input_timesteps)(latent_inputs)

# A ################################################################
# x = tf.keras.layers.LSTM(intermediate_dim, return_sequences=True)(x)
# x = tf.keras.layers.Flatten()(x)
# x = tf.keras.layers.Dense(input_timesteps * num_classes, activation='softmax')(x)
# outputs = tf.keras.layers.Reshape(target_shape=(input_timesteps, num_classes))(x)
# B ################################################################
x = tf.keras.layers.LSTM(intermediate_dim, return_sequences=True)(x)
outputs = tf.keras.layers.LSTM(num_classes, return_sequences=True)(x)

# outputs = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation='softmax'))(x)
# C ################################################################
# x = tf.keras.layers.LSTM(intermediate_dim)(x)
# x = tf.keras.layers.Dense(input_timesteps * num_classes, activation='softmax')(x)
# outputs = tf.keras.layers.Reshape(target_shape=(input_timesteps, num_classes))(x)
# D ################################################################
# x = tf.keras.layers.LSTM(intermediate_dim)(x)
# outputs = tf.keras.layers.Dense(input_timesteps * num_classes, activation='softmax')(x)
####################################################################
# instantiate decoder model
decoder = tf.keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='figs/vae_mlp_decoder.png', show_shapes=True)

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])  # connect the decoder with the encoder
vae = tf.keras.Model(inputs, outputs, name='vae_mlp')

# training the model
reconstruction_loss = binary_crossentropy(inputs, outputs)
reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
vae.compile(optimizer='adam')
vae.summary()
plot_model(vae, to_file='figs/vae_mlp.png', show_shapes=True)
# vae.fit(x_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, None))