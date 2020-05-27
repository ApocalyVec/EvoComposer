"""
Reference to
https://gplearn.readthedocs.io/en/stable/index.html

ApocalyVec access on 5/3/20
"""
from gplearn.genetic import SymbolicRegressor
from gplearn.fitness import make_fitness
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
import graphviz
import os

import findspark as findspark
from collections import Counter, OrderedDict
from sklearn.model_selection import train_test_split

from utils.MIDI_utils import read_midi, _create_sc, create_filtered, prepare_xy, encode_seq
from utils.MIDI_utils import convert_to_midi

import pickle


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


if __name__ == '__main__':
    _train = False
    data_dir = 'music/resource/schubert'
    timesteps = 32
    f_threshold = 50

    spark_location = '/usr/local/lib/python3.7/site-packages/pyspark'  # Set your own
    java8_location = '/Library/Java/JavaVirtualMachines/jdk1.8.0_202.jdk/Contents/Home'
    os.environ['JAVA_HOME'] = java8_location
    findspark.init(spark_home=spark_location)
    sc = _create_sc(num_cores=8, driver_mem=12, max_result_mem=12)

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
    x, y = prepare_xy(music_filtered, window_size=timesteps)

    # one-hot encode MIDI symbols
    unique_x = list(set(x.ravel()))
    x_encoded = dict((note_, number) for number, note_ in enumerate(unique_x))
    x_seq = encode_seq(x, x_encoded)

    unique_y = list(set(y))
    y_note_to_int = dict((note_, number) for number, note_ in enumerate(unique_y))
    y_seq = np.array([y_note_to_int[i] for i in y])

    # create train-test split
    X_train, X_test, y_train, y_test = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

    # make music_list
    output_path = 'music/gp_music.mid'

    # edit_metric = make_fitness(calc_edit, greater_is_better=False, wrap=True)

    est_gp = SymbolicRegressor(population_size=1000,
                               generations=20, stopping_criteria=0.01,
                               p_crossover=0.7, p_subtree_mutation=0.1,
                               p_hoist_mutation=0.05, p_point_mutation=0.1,
                               max_samples=0.9, verbose=1,
                               function_set=('add', 'sub', 'mul', 'div'),
                               parsimony_coefficient=0.01, random_state=0,
                               )

    est_gp.fit(X_train, y_train)
    pickle_gp = pickle.dumps(est_gp)
    pickle.dump(est_gp, open("models/est_gp.p", "wb"))

    print(est_gp._program)

    est_tree = DecisionTreeRegressor()
    est_tree.fit(X_train, y_train)
    pickle.dump(est_tree, open("models/est_tree.p", "wb"))

    est_rf = RandomForestRegressor()
    est_rf.fit(X_train, y_train)
    pickle_rf = pickle.dumps(est_rf)
    pickle.dump(est_rf, open("models/est_rf.p", "wb"))

    y_gp = est_gp.predict(np.c_[X_train.ravel(), X_test.ravel()]).reshape(X_train.shape)
    score_gp = est_gp.score(X_test, y_test)
    y_tree = est_tree.predict(np.c_[X_train.ravel(), X_test.ravel()]).reshape(X_train.shape)
    score_tree = est_tree.score(X_test, y_test)
    y_rf = est_rf.predict(np.c_[X_train.ravel(), X_test.ravel()]).reshape(X_train.shape)
    score_rf = est_rf.score(X_test, y_test)

    fig = plt.figure(figsize=(12, 10))

    for i, (y, score, title) in enumerate([(y_gp, score_gp, "SymbolicRegressor"),
                                           (y_tree, score_tree, "DecisionTreeRegressor"),
                                           (y_rf, score_rf, "RandomForestRegressor")]):

        ax = fig.add_subplot(2, 2, i + 1, projection='3d')
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_xticks(np.arange(-1, 1.01, .5))
        ax.set_yticks(np.arange(-1, 1.01, .5))
        points = ax.scatter(X_train[:, 0], X_train[:, 1], y_train)
        if score is not None:
            score = ax.text(-.7, 1, .2, "$R^2 =\/ %.6f$" % score, 'hidden_decoder', fontsize=14)
        plt.title(title)

    plt.show()

    dot_data = est_gp._program.export_graphviz()
    graph = graphviz.Source(dot_data)
    graph.render('images/ex1_child', format='png', cleanup=True)
    graph
    compose(est_gp, unique_x, X_test, timesteps, fp=output_path)
