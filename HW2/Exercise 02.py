import argparse
import numpy as np
import os
import tensorflow as tf
from keras.losses import SparseCategoricalCrossentropy
from tensorflow import keras
from keras import layers

parser = argparse.ArgumentParser()
parser.add_argument('--version', type=str, required=True,
                    help='choosing the output step')
args = parser.parse_args()
if args.version == 'a':
    mfcc = True
    alpha = 1
    momentom = 0.1
    num_filter = 512
    STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
    MFCC_OPTIONS = {'frame_length': 650, 'frame_step': 350, 'mfcc': True,
                    'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 30,
                    'num_coefficients': 9}
elif args.version == 'b':
    mfcc = True
    alpha = 1
    num_filter = 256
    momentom = 0.3
    STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
    MFCC_OPTIONS = {'frame_length': 650, 'frame_step': 350, 'mfcc': True,
                    'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 30,
                    'num_coefficients': 9}
elif args.version == 'c':
    mfcc = True
    alpha = 1
    num_filter = 128
    momentom = 0.6
    STFT_OPTIONS = {'frame_length': 256, 'frame_step': 128, 'mfcc': False}
    MFCC_OPTIONS = {'frame_length': 650, 'frame_step': 350, 'mfcc': True,
                    'lower_frequency': 20, 'upper_frequency': 4000, 'num_mel_bins': 30,
                    'num_coefficients': 9}

seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

data_dir = os.path.join('.', 'data', 'mini_speech_commands')
filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
num_samples = len(filenames)
LABELS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
total = 8000
train_files = open('./Dataset/kws_train_split(new).txt', 'r')
test_files = open('./Dataset/kws_test_split(new).txt', 'r')

list_train = []
y_train = []
for i in train_files:
    list_train.append(i)
    tmp = i.replace('.\\data\\mini_speech_commands\\', '')
    loc_slash = tmp.find('\\')
    y_train.append(LABELS.index(tmp[:loc_slash]))
train_files = tf.convert_to_tensor([s.rstrip() for s in list_train])

test_list = []
y_test = []
for i in test_files:
    test_list.append(i)
    tmp = i.replace('.\\data\\mini_speech_commands\\', '')
    loc_slash = tmp.find('\\')
    y_test.append(LABELS.index(tmp[:loc_slash]))
test_files = tf.convert_to_tensor([s.rstrip() for s in test_list])
y_test = np.array(y_test)


class SignalGenerator:
    def __init__(self, labels, sampling_rate, frame_length, frame_step,
                 num_mel_bins=None, lower_frequency=None, upper_frequency=None,
                 num_coefficients=None, mfcc=False):
        self.labels = labels
        self.sampling_rate = sampling_rate
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.num_mel_bins = num_mel_bins
        self.lower_frequency = lower_frequency
        self.upper_frequency = upper_frequency
        self.num_coefficients = num_coefficients
        num_spectrogram_bins = (frame_length) // 2 + 1

        if mfcc is True:
            self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
                self.num_mel_bins, num_spectrogram_bins, self.sampling_rate,
                self.lower_frequency, self.upper_frequency)
            self.preprocess = self.preprocess_with_mfcc
        else:
            self.preprocess = self.preprocess_with_stft

    def read(self, file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        label = parts[-2]
        label_id = tf.argmax(label == self.labels)
        audio_binary = tf.io.read_file(file_path)
        audio, _ = tf.audio.decode_wav(audio_binary)
        audio = tf.squeeze(audio, axis=1)

        return audio, label_id

    def pad(self, audio):
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])

        return audio

    def get_spectrogram(self, audio):
        stft = tf.signal.stft(audio, frame_length=self.frame_length,
                              frame_step=self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        return spectrogram

    def get_mfccs(self, spectrogram):
        mel_spectrogram = tf.tensordot(spectrogram,
                                       self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.num_coefficients]

        return mfccs

    def preprocess_with_stft(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        spectrogram = tf.image.resize(spectrogram, [32, 32])

        return spectrogram, label

    def preprocess_with_mfcc(self, file_path):
        audio, label = self.read(file_path)
        audio = self.pad(audio)
        spectrogram = self.get_spectrogram(audio)
        mfccs = self.get_mfccs(spectrogram)
        mfccs = tf.expand_dims(mfccs, -1)

        return mfccs, label

    def make_dataset(self, files):
        ds = tf.data.Dataset.from_tensor_slices(files)
        ds = ds.map(self.preprocess, num_parallel_calls=4)
        ds = ds.batch(32)
        ds = ds.cache()

        return ds


def mlp_model(num_filter):
    mlp = keras.Sequential()
    mlp.add(layers.Flatten())
    mlp.add(layers.Dense(units=num_filter, activation='relu'))
    mlp.add(layers.Dense(units=num_filter, activation='relu'))
    mlp.add(layers.Dense(units=num_filter, activation='relu'))
    mlp.add(layers.Dense(units=8))
    mlp.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam')
    return mlp


def cnn_model(num_filter, momentom, alpha=1):
    cnn = keras.Sequential()
    cnn.add(layers.Conv2D(filters=num_filter * alpha, kernel_size=[3, 3], strides=strides, use_bias=False,
                          activation='relu'))
    cnn.add(layers.BatchNormalization(momentum=momentom))
    cnn.add(layers.Conv2D(filters=num_filter * alpha, kernel_size=[3, 3], strides=[1, 1], use_bias=False,
                          activation='relu'))
    cnn.add(layers.BatchNormalization(momentum=momentom))
    cnn.add(layers.Conv2D(filters=num_filter * alpha, kernel_size=[3, 3], strides=[1, 1], use_bias=False,
                          activation='relu'))
    cnn.add(layers.BatchNormalization(momentum=momentom))
    cnn.add(layers.GlobalAveragePooling2D())
    cnn.add(layers.Dense(units=8))
    cnn.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam', metrics=['accuracy'])
    return cnn


def ds_cnn_model(alpha=1):
    ds_cnn = keras.Sequential()
    ds_cnn.add(
        layers.Conv2D(filters=256 * alpha, kernel_size=[3, 3], strides=strides, use_bias=False, activation='relu'))
    ds_cnn.add(layers.BatchNormalization(momentum=0.1))
    ds_cnn.add(layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False))
    ds_cnn.add(
        layers.Conv2D(filters=256 * alpha, kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'))
    ds_cnn.add(layers.BatchNormalization(momentum=0.1))
    ds_cnn.add(layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False))
    ds_cnn.add(
        layers.Conv2D(filters=256 * alpha, kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'))
    ds_cnn.add(layers.BatchNormalization(momentum=0.1))
    ds_cnn.add(layers.DepthwiseConv2D(kernel_size=[3, 3], strides=[1, 1], use_bias=False))
    ds_cnn.add(
        layers.Conv2D(filters=256 * alpha, kernel_size=[1, 1], strides=[1, 1], use_bias=False, activation='relu'))
    ds_cnn.add(layers.BatchNormalization(momentum=0.1))
    ds_cnn.add(layers.GlobalAveragePooling2D())
    ds_cnn.add(layers.Dense(units=8))
    ds_cnn.compile(loss=SparseCategoricalCrossentropy(from_logits=True), optimizer='adam')
    return ds_cnn


if mfcc is True:
    options = MFCC_OPTIONS
    strides = [2, 1]
else:
    options = STFT_OPTIONS
    strides = [2, 2]

generator = SignalGenerator(LABELS, 16000, **options)
train_ds = generator.make_dataset(train_files)
test_ds = generator.make_dataset(test_files)
units = 8
model = cnn_model(num_filter=num_filter, momentom=momentom, alpha=alpha)
model.fit(train_ds, verbose=1, epochs=30)
eval_result = np.argmax(model.predict(test_ds, verbose=2), axis=1)
eval_acc = sum(eval_result == y_test) / len(y_test)
print('Accuracy: ' + str(eval_acc))

import tensorflow_model_optimization as tfmot
import tempfile

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

batch_size = 1
epochs = 15
validation_split = 0
num_train = 6400 * (1 - validation_split)
end_step = np.ceil(num_train / batch_size).astype(np.int32) * epochs
pruning_params = {
    'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(initial_sparsity=0.3,
                                                             final_sparsity=0.8,
                                                             begin_step=6400 * 5,
                                                             end_step=6400 * 15)
}

model_for_pruning = prune_low_magnitude(model, **pruning_params)
model_for_pruning.compile(optimizer='adam',
                          loss=SparseCategoricalCrossentropy(from_logits=True),
                          metrics=['accuracy'])
logdir = tempfile.mkdtemp()
callbacks = [
    tfmot.sparsity.keras.UpdatePruningStep(),
    tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
]

model_for_pruning.fit(train_ds, batch_size=batch_size,
                      epochs=epochs, callbacks=callbacks)

model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)

converter = tf.lite.TFLiteConverter.from_keras_model(model_for_export)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
pruned_tflite_model = converter.convert()
pruned_tflite_file = './pruned_model_' + str(args.version) + '.tflite'
with open(pruned_tflite_file, 'wb') as f:
    f.write(pruned_tflite_model)

eval_result = np.argmax(model_for_pruning.predict(test_ds, verbose=1), axis=1)
eval_acc = sum(eval_result == y_test) / len(y_test)
print('Test accuracy (after Pruning): ' + str(eval_acc))

import zlib, sys

filename_out = "./compressed_model_" + str(args.version) + ".tflite.zlib"

with open(pruned_tflite_file, mode="rb") as fin, open(filename_out, mode="wb") as fout:
    data = fin.read()
    compressed_data = zlib.compress(data, 9)
    print(f"Original size: {sys.getsizeof(data)}")
    print(f"Compressed size: {sys.getsizeof(compressed_data)}")

    fout.write(compressed_data)
