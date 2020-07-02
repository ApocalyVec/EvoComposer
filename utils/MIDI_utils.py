from collections import OrderedDict
from collections import Counter, OrderedDict

import findspark
from music21 import converter, instrument, note, chord, stream
import numpy as np
from pyspark import SparkConf, SparkContext

import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


def generate_MIDI_representation(file, release_freq, rest_freq, beat_resolution=24):
    """
    # defining function to read MIDI files
    :param file:
    :return: np array of notes
    """
    print("Loading Music File:", file)

    notes = []
    notes_to_parse = None
    sampled_notes = []
    sampled_freq = []
    midi = converter.parse(file)

    # grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)
    if s2:
        for part in s2.parts:
            # select elements of only piano
            if 'Piano' in str(part):

                # notes_to_parse = part.recurse()
                notes_to_parse = part.recurse()

                # finding whether a particular element is note or a chord
                for element in notes_to_parse:
                    # note
                    if isinstance(element, note.Note):
                        duration = element.duration
                        notes.append(str(element.pitch))

                        replicate = [element.pitch] * int(duration.quarterLength * beat_resolution)
                        sampled_notes.extend(replicate)
                        if len(sampled_notes) != 0:
                            sampled_notes[-1] = 'Release'

                        note_freq = [element.pitch.frequency] * int(duration.quarterLength * beat_resolution)
                        sampled_freq.extend(note_freq)
                        if len(sampled_freq) != 0:
                            sampled_freq[-1] = release_freq
                    # chord
                    # elif isinstance(element, chord.Chord):
                        # duration = element.duration
                        # for pitch in element.pitches:
                        #     print(pitch.freq)
                        # if element.volume.velocity == 0:
                        #     release.append(element)
                        # if duration.fullName == 'Zero':
                        #     print(duration.quarterLength)
                        #     print("found2")
                        #     zeros.append(duration)
                        # notes.append('.'.join(str(n) for n in element.normalOrder))
                        # # durations.append(duration)
                        # replicate = ['.'.join(str(n) for n in element.normalOrder)] * int(duration.quarterLength * 24)
                        # sampled_notes.extend(replicate)
                        # sampled_notes[-1] = 'Release'
                        # sets[duration.fullName] = duration.quarterLength
                    elif isinstance(element, note.Rest):
                        duration = element.duration
                        replicate = ['Rest'] * int(duration.quarterLength * beat_resolution)
                        sampled_notes.extend(replicate)
                        note_freq = [rest_freq] * int(duration.quarterLength * beat_resolution)
                        sampled_freq.extend(note_freq)
    else:
        print('Ignoring MIDI file: ' + file + '\nfor it cannot be partitioned by instrument ')
    return np.array(notes), sampled_notes, sampled_freq


def _create_sc(num_cores: int, driver_mem: int, max_result_mem: int):
    conf = SparkConf(). \
        setMaster("local[" + str(num_cores) + "]"). \
        setAppName("Genex").set('spark.driver.memory', str(driver_mem) + 'G'). \
        set('spark.driver.maxResultSize', str(max_result_mem) + 'G')
    sc = SparkContext(conf=conf)

    return sc


def create_filtered(notes_array, frequent_notes):
    new_music = []
    for notes in notes_array:
        temp = []
        for note_ in notes:
            if note_ in frequent_notes:
                temp.append(note_)
        new_music.append(temp)
    return np.array(new_music)


def prepare_xy(music, window_size):
    x = []
    y = []

    for note_ in music:
        for i in range(0, len(note_) - window_size, 1):
            # preparing input and output sequences
            input_ = note_[i:i + window_size]
            output = note_[i + window_size]

            x.append(input_)
            y.append(output)

    return np.array(x), np.array(y)


def prepare_x(music_list, window_size, stride=1):
    x = []
    for note_array in music_list:
        for i in range(0, len(note_array) - window_size, stride):
            input_ = note_array[i:i + window_size]
            x.append(input_)
    return np.array(x)


def window_slice(data, window_size, stride):
    assert window_size <= len(data)
    assert stride > 0
    rtn = []
    for i in range(window_size, len(data), stride):
        rtn.append(data[i - window_size:i])
    return np.array(rtn)


def encode_seq(x, encode_dict):
    x_seq = []
    for i in x:
        temp = []
        for j in i:
            # assigning unique integer to every note
            temp.append(encode_dict[j])
        x_seq.append(temp)

    return np.array(x_seq)


def convert_to_midi(prediction_output, fp):
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:

        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                cn = int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)

            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)

        # pattern is a note
        else:

            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=fp)


def load_samples_repr(data_dir, timesteps, release_freq=0, rest_freq=1, rest_threshold=0.3, _use_spark=False):
    files = [x for x in os.listdir(data_dir) if x.split('.')[-1] == 'mid']  # read all the files end with mid
    files = files[:16]
    if _use_spark:
        sc = _create_sc(num_cores=16, driver_mem=12, max_result_mem=12)
        files_rdd = sc.parallelize(files)
        processed_rdd = files_rdd.map(lambda x: generate_MIDI_representation(os.path.join(data_dir, x), release_freq, rest_freq)).cache()
        processed = processed_rdd.collect()
    else:
        processed = np.array([generate_MIDI_representation(os.path.join(data_dir, x), release_freq, rest_freq) for x in files])
    freq_array_list = [np.array(x[2]) for x in processed]
    X = prepare_x(freq_array_list, window_size=timesteps)
    # TODO implement that filter_by_rest function
    X = filter_by_rests(X, rest_freq, threshold=rest_threshold)
    unique_x = np.unique(X.flatten())

    le = LabelEncoder().fit(unique_x)
    X = np.array([le.transform(x) for x in X])
    for x in X:
        le.inverse_transform(x)
    # create train-test split
    x_tr, x_val = train_test_split(X, test_size=0.2, random_state=0)

    return x_tr, x_val, unique_x, le

def filter_by_rests(samples, rest_freq, threshold):
    # TODO implement that filter_by_rest function
    rtn = []
    for s in samples:
        rests = [x for x in s if x == rest_freq]
        if len(rests) / len(s) <= threshold:
            rtn.append(s)
    return np.array(rtn)


def load_sample_unsupervised(data_dir, timesteps, f_threshold, _use_spark=False):
    files = [x for x in os.listdir(data_dir) if x.split('.')[-1] == 'mid']  # read all the files end with mid
    if _use_spark:
        # spark_location = '/Users/Leo/spark-2.4.3-bin-hadoop2.7'  # Set your own
        # java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_151.jdk/Contents/Home/jre'
        # os.environ['JAVA_HOME'] = java8_location
        # findspark.init(spark_home=spark_location)
        sc = _create_sc(num_cores=16, driver_mem=12, max_result_mem=12)
        files_rdd = sc.parallelize(files)
        notes_rdd = files_rdd.map(lambda x: generate_MIDI_representation(os.path.join(data_dir, x))).cache()
        notes_array = notes_rdd.collect()
    else:
        notes_array = np.array([generate_MIDI_representation(os.path.join(data_dir, x)) for x in files])
    notes_flat = [item for sublist in notes_array for item in sublist]
    freq = OrderedDict(Counter(notes_flat))

    # only keep the notes with frequency that are higher than 50
    frequent_notes = [n for n, f in freq.items() if f >= f_threshold]
    music_filtered = create_filtered(notes_array, frequent_notes)

    x, encoder = prepare_x(music_filtered, window_size=timesteps)

    # one-hot encode MIDI symbols
    # TODO use sklearn label encoder

    x_tr, x_val = train_test_split(x, test_size=0.2, random_state=0)

    return x_tr, x_val, encoder
