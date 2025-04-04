import tensorflow as tf
from tensorflow import keras
from model import readData
import numpy as np

TEAMS = []

TEAMS = readData()

def main():
    teams = TEAMS
    model = keras.models.load_model("./src/mmpredictor.keras")
    team1 = input("Input team name:")
    team2 = input("Input team name:")
    outcome = tf.constant(np.asarray([[teams[team1].asList(), teams[team2].asList()]]), dtype="float32")
    print(f"{team1}: {int(model.predict(outcome)[0][0])} - {team2}: {int(model.predict(outcome)[0][1])}")


if __name__ == "__main__":
    main()