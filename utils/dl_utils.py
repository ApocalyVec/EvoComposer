import numpy as np
import tensorflow as tf
from keras.layers import Activation

from utils.MIDI_utils import convert_to_midi


def lstm(n_vocab):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, return_sequences=True))
    model.add(tf.keras.layers.LSTM(128))
    model.add(tf.keras.layers.Dense(256))
    model.add(Activation('relu'))
    model.add(tf.keras.layers.Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def make_model(categories, output_len):
    print('Clearing session')
    model = tf.keras.Sequential()

    # embedding layer
    model.add(tf.keras.layers.Embedding(categories, 100, input_length=32, trainable=True))

    model.add(tf.keras.layers.Conv1D(64, 3, padding='causal', activation='relu'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool1D(2))

    model.add(tf.keras.layers.Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool1D(2))

    model.add(tf.keras.layers.Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.MaxPool1D(2))

    # model.add(Conv1D(256,5,activation='relu'))
    model.add(tf.keras.layers.GlobalMaxPool1D())

    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(output_len, activation='softmax'))

    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
    return model


def compose(model, unique_x, x_val, timesteps, fp):
    ind = np.random.randint(0, len(x_val) - 1)
    random_music = x_val[ind]
    predictions = []
    for i in range(10):
        random_music = random_music.reshape(1, timesteps)

        prob = model.predict(random_music)[0]
        y_pred = np.argmax(prob, axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0], len(random_music[0]), y_pred)
        random_music = random_music[1:]

    # convert the output features to back to notes
    x_int_to_note = dict((number, note_) for number, note_ in enumerate(unique_x))
    predicted_notes = [x_int_to_note[i] for i in predictions]

    # convert to playable MIDI format
    convert_to_midi(predicted_notes, fp)
