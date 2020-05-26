"""
Reference to
Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio." arXiv preprint arXiv:1609.03499 (2016).

ApocalyVec access on 5/9/20
"""

import os

# import findspark as findspark
import numpy as np
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.callbacks import *
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

from utils.dl_utils import make_model, compose
from utils.MIDI_utils import read_midi, _create_sc, create_filtered, prepare_xy, encode_seq, convert_to_midi, \
    load_samples

import random

if __name__ == '__main__':
    _train = False
    data_dir = 'data/schubert'
    timesteps = 32
    f_threshold = 50

    x_tr, x_val, y_tr, y_val, unique_x, unique_y = load_samples(data_dir, timesteps, f_threshold, _use_spark=True)

    if _train:
        # build wavenet model
        model = make_model(input_len=len(unique_x), output_len=len(unique_y))
        mc = ModelCheckpoint('models/best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
        history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=128, epochs=50,
                            validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])
        # summarize history
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    else:
        model = load_model('models/best_model.h5')

    # make music
    output_path = 'music/pred_3.mid'
    compose(model, unique_x, x_val, timesteps, fp=output_path)

