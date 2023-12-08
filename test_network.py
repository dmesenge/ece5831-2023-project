from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import sys
from numpy import genfromtxt
from keras.models import load_model

if(len(sys.argv) != 2):
    raise Exception("Incorrect Number of arguments")

model_file_path = sys.argv[1]

#get the dut name
input_split = model_file_path.split("_")
dut = input_split[0]

input_file_path = f"{dut}_test_inputs.csv"
output_file_path = f"{dut}_test_outputs.csv"

# load the inputs from the CSV
inputs = np.array(genfromtxt(input_file_path, delimiter=','))

#load the model
model = load_model(model_file_path)

predictions = model.predict(inputs)

#load the test outputs

test_outputs = np.array(genfromtxt(output_file_path, delimiter=','))

#calculate the errors
err_total = 0
err_max = 0
for idx in range(len(test_outputs)):
    error = np.abs(test_outputs[idx] - predictions[idx])
    err_total = err_total + error
    if error > err_max:
        err_max = error
    
err_norm = err_total/test_outputs.shape[0]

print(f"Average error in output: {err_norm}")
print(f"Maximum error at any point: {err_max}")

np.savetxt(f"{model_file_path}_preditions.csv", predictions, delimiter=",")

