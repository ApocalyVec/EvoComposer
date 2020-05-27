import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL

from utils.MIDI_utils import load_sample_unsupervised

NUM_CLASSES = 167
# NUM_CLASSES = 11
TIMESTEPS = 64

class RVAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(RVAE, self).__init__()
        self.latent_dim = latent_dim
        self.inference_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(TIMESTEPS, NUM_CLASSES)),
                tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
                # No activation
                tf.keras.layers.Dense(latent_dim + latent_dim),
            ]
        )

        self.generative_net = tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                tf.keras.layers.Dense(units=64, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(64, 1)),
                tf.keras.layers.LSTM(TIMESTEPS, return_sequences=True),
                # No activation
                tf.keras.layers.LSTM(NUM_CLASSES, return_sequences=True, activation='softmax')
            ]
        )

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(100, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x):
        mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.generative_net(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs

        return logits


optimizer = tf.keras.optimizers.Adam(1e-3)


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(
        -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
        axis=raxis)


@tf.function
def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)

    cross_ent = tf.nn.softmax_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

data_dir = 'data/schubert'
input_timesteps = 64
f_threshold = 50
x_train, x_test, one_hot_encoder = load_sample_unsupervised(data_dir, input_timesteps, f_threshold)

TRAIN_BUF = len(x_train)
TEST_BUF = len(x_test)
BATCH_SIZE = 1024

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(x_test).shuffle(TEST_BUF).batch(BATCH_SIZE)

epochs = 100
latent_dim = 8
num_examples_to_generate = 16

# keeping the random vector constant for generation (prediction) so
# it will be easier to see the improvement.
random_vector_for_generation = tf.random.normal(
    shape=[num_examples_to_generate, latent_dim])
model = RVAE(latent_dim)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    train_loss = tf.keras.metrics.Mean()
    for train_x in train_dataset:
        loss = compute_apply_gradients(model, train_x, optimizer)
        train_loss(loss)
    end_time = time.time()

    if epoch % 1 == 0:
        test_loss = tf.keras.metrics.Mean()
        for test_x in test_dataset:
            test_loss(compute_loss(model, test_x))
        train_elbo = -train_loss.result()
        test_elbo = -test_loss.result()
        print('Epoch: {}, Train set ELBO: {}, Validation ELBO: {}'
              'time elapse for current epoch {}'.format(epoch,
                                                        train_elbo,
                                                        test_elbo,
                                                        end_time - start_time))
