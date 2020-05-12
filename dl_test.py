"""
Reference to
Oord, Aaron van den, et al. "Wavenet: A generative model for raw audio." arXiv preprint arXiv:1609.03499 (2016).

ApocalyVec access on 5/9/20
"""

import os

import findspark as findspark
import numpy as np
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split
from keras.models import *
from keras.callbacks import *
import keras.backend as K
from keras.models import load_model
import matplotlib.pyplot as plt

from utils.dl_utils import make_model, compose
from utils.MIDI_utils import read_midi, _create_sc, create_filtered, prepare_xy, encode_seq, convert_to_midi

import random

if __name__ == '__main__':
    _train = False
    data_dir = '/Users/Leo/Documents/data/schubert'
    timesteps = 32
    f_threshold = 50

    spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7'  # Set your own
    java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
    os.environ['JAVA_HOME'] = java8_location
    findspark.init(spark_home=spark_location)
    sc = _create_sc(num_cores=16, driver_mem=12, max_result_mem=12)

    files = [x for x in os.listdir(data_dir) if x.split('.')[-1] == 'mid']  # read all the files end with mid
    files_rdd = sc.parallelize(files)
    notes_flat_rdd = files_rdd.flatMap(lambda x: read_midi(os.path.join(data_dir, x))).cache()
    notes_rdd = files_rdd.map(lambda x: read_midi(os.path.join(data_dir, x))).cache()

    notes_array = notes_rdd.collect()
    notes = notes_flat_rdd.collect()

    freq = OrderedDict(Counter(notes))

    # plt.plot([f for f in freq.values()])  # plot the frequencies

    # only keep the notes with frequency that are higher than 50
    frequent_notes = [n for n, f in freq.items() if f >= f_threshold]
    music_filtered = create_filtered(notes_array, frequent_notes)
    x, y = prepare_xy(music_filtered, timesteps=timesteps)

    # one-hot encode MIDI symbols
    # TODO use sklearn label encoder
    unique_x = list(set(x.ravel()))
    x_encoded = dict((note_, number) for number, note_ in enumerate(unique_x))
    x_seq = encode_seq(x, x_encoded)

    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
    y_seq = np.array([y_note_to_int[i] for i in y])

    # create train-test split
    x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

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

