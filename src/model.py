# WHO WOULD WIN
# CREATED BY CAGE BULLARD 2025

from Team import Team
import re
from os import path
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np
import matplotlib.pyplot as plt
from keras import activations


TEAMS: dict[Team] = {}
EVAL_SIZE: int = -20


def readData():
    teams = TEAMS
    data: str = ""

    with open("src/kenpom.csv", "r+") as f:
        data = f.read()

    data=data.splitlines()[1:]

    for line in data:
        t: Team = Team()
        cells = line.split(",")
        t.rank = int(cells[0])
        wl = cells[3].split("-")
        t.wl = (int(wl[0]), int(wl[1]))

        t.team = re.match(r"([a-zA-Z.&' ]*)", cells[1])[0].strip()
        t.off = cells[5]
        t.dfs = cells[6]
        t.net = cells[4]
        teams[t.team] = (t)

    return teams

def tokenizeData():
    teams = TEAMS
    matchups = []
    results = []
    matrices = []

    with open("./src/matchups.csv", "r+") as f:
        matchups=f.read().splitlines()[1:]
    #   * tokenize and reduce to Team data structure 2x4 matrix
    for match in matchups:
        cells = match.split(",")
        team1 = teams[cells[0]].asList()
        team2 = teams[cells[1]].asList()
        matrices.append([team1, team2])
        results.append([int(cells[2]), int(cells[3])])

    return results, matrices


def trainModel(model: keras.Model, xtrain, ytrain, xval, yval, verbose: bool=False):
    history = model.fit(
        xtrain,
        ytrain,
        batch_size=1000,
        epochs=200,
        validation_data=(xval, yval)
    )
    if verbose:
        print(history.history)
    return history


@tf.keras.utils.register_keras_serializable("lossMietric")
def lossMetric(y_true, y_pred):
    """Return the square of the distance between the two points (x1, y1) and (ex1, ey1)"""
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), 1))


def makeModel():
    inputs = keras.Input(shape=(2,4,), dtype=tf.float32)
    layer1 = layers.Dense(4)(inputs) # gelu worked really well
    layer2 = layers.Conv1D(2, 1)(layer1)
    output = layers.Dense(1)(layer2)
    model = keras.Model(inputs=inputs, outputs=output, name="mmpredictor")
    model.compile(
        loss=lossMetric,
        optimizer='adam',
    )
    return model


def main(train=True):
    readData()

    teams = TEAMS

    # TODO:
    #   * read data in matchups.csv
    results, matrices = tokenizeData()

    xtrain=tf.Variable(np.asarray(matrices[:EVAL_SIZE]).astype("float32"), dtype="float32")
    xval=tf.Variable(np.asarray(matrices[EVAL_SIZE:]).astype("float32"), dtype="float32")
    ytrain=tf.Variable(np.asarray(results[:EVAL_SIZE]).astype("int32"), dtype="int32")
    yval=tf.Variable(np.asarray(results[EVAL_SIZE:]).astype("int32"), dtype="int32")

    if(path.exists(r"./src/mmpredictor.keras")):
        model = keras.models.load_model(r"./src/mmpredictor.keras")
    else:
        model: keras.Model = makeModel()

    if train:
        history = trainModel(model, xtrain, ytrain, xval, yval, verbose=True)
        plt.plot(history.history['loss'], label = "Training Loss")
        plt.show()

    model.save(r"./src/mmpredictor.keras")

    duke_v_houston = tf.constant(np.asarray([[teams["Auburn"].asList(), teams["Florida"].asList()]]), dtype="float32")
    print(model.predict(duke_v_houston))


    # results = model.evaluate(xval, yval, batch_size=128)
    # print(results)

if __name__ == "__main__":
    main()