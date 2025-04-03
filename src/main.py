# WHO WOULD WIN
# CREATED BY CAGE BULLARD 2025

from Team import Team
import re
import tensorflow as tf
from tensorflow import keras
from keras import layers
import numpy as np

TEAMS: dict[Team] = {}

def readData():
    teams = TEAMS
    data: str = ""

    with open("kenpom2025.csv", "r+") as f:
        data = f.read()

    data=data.splitlines()[1:]

    for line in data:
        t: Team = Team()
        cells = line.split(",")
        t.rank = int(cells[0])
        wl = cells[3].split("-")
        t.wl = (int(wl[0]), int(wl[1]))

        t.team = re.match(r"([a-zA-Z.& ]*)", cells[1])[0].strip()
        t.off = cells[5]
        t.dfs = cells[6]
        t.net = cells[4]
        teams[t.team] = (t)


def lossMetric(y_true, y_pred):
    """Return the square of the distance between the two points (x1, y1) and (ex1, ey1)"""
    return tf.sqrt(tf.reduce_sum(tf.square(y_true - y_pred), 1))

def main():
    readData()

    teams = TEAMS

    # TODO:
    #   * read data in matchups.csv
    matchups = []
    results = []
    matrices = []
    with open("matchups.csv", "r+") as f:
        matchups=f.read().splitlines()[1:]
    #   * tokenize and reduce to Team data structure 2x4 matrix
    for match in matchups:
        cells = match.split(",")
        team1 = teams[cells[0]].asList()
        team2 = teams[cells[1]].asList()
        matrices.append([team1, team2])
        results.append([int(cells[2]), int(cells[3])])

    #   * feed two team tensors into a model that produces a 2x1 tuple output
    inputs = keras.Input(shape=(2,4,), dtype=tf.float32)
    l = layers.Dense(2, dtype=tf.float32)(inputs)
    output = layers.Dense(1, dtype=tf.float32)(l)
    model = keras.Model(inputs=inputs, outputs=output, name="mmpredictor")

    print(model.summary())

    model.compile(
        loss=lossMetric,
        optimizer="adam",
    )

    xtrain=tf.Variable(np.asarray(matrices[:-10]).astype("float32"), dtype="float32")
    xval=tf.Variable(np.asarray(matrices[-10:]).astype("float32"), dtype="float32")
    # print(xval)
    ytrain=tf.Variable(np.asarray(results[:-10]).astype("float32"), dtype="float32")
    yval=tf.Variable(np.asarray(results[-10:]).astype("float32"), dtype="float32")
    # print(yval)

    history = model.fit(
        xtrain,
        ytrain,
        batch_size=10,
        epochs=9,
        validation_data=(xval, yval)
    )
    print(history.history)
    model.export(r"./model")

if __name__ == "__main__":
    main()