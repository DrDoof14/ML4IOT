import tempfile
import argparse
from keras.metrics import mean_absolute_error as mae
import tensorflow_model_optimization as tfmot
import numpy as np
import os
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True,
                    help='choosing the output step')
args = parser.parse_args()

if args.version == 'a':
    output_steps = 3
elif args.version == 'b':
    output_steps = 9

zip_path = tf.keras.utils.get_file(
    origin='https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip',
    fname='jena_climate_2009_2016.csv.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')
csv_path, _ = os.path.splitext(zip_path)
df = pd.read_csv(csv_path)

column_indices = [2, 5]
columns = df.columns[column_indices]
data = df[columns].values.astype(np.float32)

n = len(data)
train_data = data[0:int(n * 0.7)]
test_data = data[int(n * 0.9):]


def window_gen(data, input_length, output_length, vectorize):
    x = []
    y = []
    for index in range(0, len(data) - input_length - output_length):
        x.append(data[index:index + input_length])
        output_loc = data[index + input_length:index + input_length + output_length]
        if vectorize == True:
            y.append(output_loc.reshape(-1))
        else:
            y.append(output_loc)
    x = np.array(x)  # to make it possible to run
    y = np.array(y)
    return x, y


def mlp_model(out_steps, alpha=1):
    mlp = keras.Sequential()
    mlp.add(keras.Input(shape=(input_steps, 2)))
    mlp.add(layers.Flatten())
    mlp.add(layers.Dense(units=128 * alpha, activation='relu'))
    mlp.add(layers.Dense(units=128 * alpha, activation='relu'))
    mlp.add(layers.Dense(units=out_steps * 2))
    mlp.compile(loss='mae', optimizer='adam')
    return mlp


def cnn_model(out_steps, alpha=1):
    cnn = tf.keras.Sequential([
        layers.Conv1D(filters=64 * alpha, kernel_size=(3,), activation='relu'),
        layers.Flatten(),
        layers.Dense(units=64 * alpha, activation='relu'),
        layers.Dense(units=out_steps * 2)
    ])
    cnn.compile(loss='mae', optimizer='adam')
    return cnn


def normalization(x, y):
    x_mean = np.mean(x, axis=0)
    y_mean = np.mean(y, axis=0)
    x_std = np.std(x, axis=0)
    y_std = np.std(y, axis=0)
    x_norm = (x - x_mean) / (x_std + 1.e-6)
    y_norm = (y - y_mean) / (y_std + 1.e-6)

    return x_norm, y_norm


input_steps = 6

x_train, y_train = window_gen(train_data, input_steps, output_steps, True)
x_test, y_test = window_gen(test_data, input_steps, output_steps, True)

x_train_norm, y_train_norm = normalization(x_train, y_train)
x_test_norm, y_test_norm = normalization(x_test, y_test)

model = cnn_model(out_steps=output_steps, alpha=0.25)
model.fit(x_train_norm, y_train_norm, batch_size=32, epochs=20)
y_test_predict = model.predict(x_test_norm, verbose=2)

y_test_temperature = y_test_norm[:, 0::2]
y_test_humidity = y_test_norm[:, 1::2]
y_test_temperature_predicted = y_test_predict[:, 0::2]
y_test_humidity_predicted = y_test_predict[:, 1::2]

test_temperature_mae = mae(y_test_temperature, y_test_temperature_predicted).numpy()
print("Test Temperature MAE(Before Pruning):", np.mean(test_temperature_mae))
test_humidity_mae = mae(y_test_humidity, y_test_humidity_predicted).numpy()
print("Test Humidity MAE(Before Pruning):", np.mean(test_humidity_mae))

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude
batch_size = 32
epochs = 20
validation_split = 0.2

num_train = 294370 * (1 - validation_split)
end_step = np.ceil(num_train / batch_size).astype(np.int32) * epochs
# Define model for pruning.
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0,
                                                             final_sparsity=0.9,
                                                             begin_step=0,
                                                             end_step=end_step)
}
model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam',
                          loss=tf.keras.losses.MeanAbsoluteError(),
                          metrics=['mae'])
logdir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(x_train_norm, y_train_norm,
                      batch_size=batch_size, epochs=epochs, callbacks=callbacks)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
pruned_tflite_model = converter.convert()
pruned_tflite_file = './pruned_model' + str(output_steps) + '.tflite'
with open(pruned_tflite_file, 'wb') as f:
    f.write(pruned_tflite_model)

import zlib, sys

filename_in = pruned_tflite_file
filename_out = "./compressed_model" + str(output_steps) + ".tflite.zlib"

with open(filename_in, mode="rb") as fin, open(filename_out, mode="wb") as fout:
    data = fin.read()
    compressed_data = zlib.compress(data, 9)
    print(f"Compressed size: {sys.getsizeof(compressed_data)}")
    fout.write(compressed_data)

# **********MAE for the pruned model************
y_test_predict_prune = model_for_pruning.predict(x_test_norm, verbose=2)
y_test_temperature_predicted_pruned = y_test_predict_prune[:, 0::2]
y_test_humidity_predicted_pruned = y_test_predict_prune[:, 1::2]
test_temperature_mae_pruned = tf.keras.metrics.mean_absolute_error(y_test_temperature,
                                                                   y_test_temperature_predicted_pruned)
print("Test Temperature MAE(After Pruning):", np.mean(test_temperature_mae_pruned))

test_humidity_mae_pruned = tf.keras.metrics.mean_absolute_error(y_test_humidity, y_test_humidity_predicted_pruned)
print("Test Humidity MAE(After Pruning):", np.mean(test_humidity_mae_pruned))
