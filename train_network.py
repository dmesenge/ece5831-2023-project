from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import sys
from numpy import genfromtxt


epochs = 30
batch_size = 128
nodes = 50

if(len(sys.argv) != 3):
    raise Exception("Incorrect Number of arguments")

inputs_file_path = sys.argv[1]
label_file_path = sys.argv[2]

#get the dut name
input_split = inputs_file_path.split("_")
dut = input_split[0]

# load the inputs from the CSV
inputs = np.array(genfromtxt(inputs_file_path, delimiter=','))
#inputs = inputs.reshape(100148, 20)

# load the labels from the CSV
labels = np.transpose(np.array([genfromtxt(label_file_path, delimiter=',')]))

#create the network
#model = keras.Sequential([
#        layers.Dense(100, activation="relu", input_shape=(20,)),
#        layers.Dense(50, activation="relu"),
#        layers.Dense(1, activation="linear")
#        ])
model = keras.Sequential([
        layers.Dense(nodes, activation="relu", input_shape=(20,)),
        layers.Dense(1, activation="linear")
        ])

model.compile(optimizer="rmsprop",
loss="mean_squared_error",
metrics=["accuracy"])

#train the network
model.fit(inputs, labels, epochs=epochs, batch_size=batch_size)

model.save(f"models/{dut}_{nodes}n_{epochs}e_{batch_size}b.h5")
