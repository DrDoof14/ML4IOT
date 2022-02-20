import json
import sys
import requests
import tensorflow as tf
import numpy as np
from scipy.special import softmax
import time

url = 'http://127.0.0.1:8080/predict'
sampling_rate = 16000
frame_length = int(0.04 * sampling_rate)
frame_step = int(0.02 * sampling_rate)
num_mel_bins = 28
lower_freq = 20
upper_freq = 4000
coefficients = 10
num_spectrogram_bins = frame_length // 2 + 1
model_path = "../Prerequisite/kws_dscnn_True.tflite"
test_files = list(open('../Prerequisite/kws_test_split.txt', 'r'))
test_files = [s.rstrip() for s in test_files]
linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins, num_spectrogram_bins, sampling_rate, lower_freq, upper_freq)
LABELS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

actual_labels = []
for i in test_files:
    tmp = i.replace('./data/mini_speech_commands/', '')
    loc_slash = tmp.find('/')
    actual_labels.append(LABELS.index(tmp[:loc_slash]))
actual_labels = np.array(actual_labels)


def mfcc(tf_audio):
    audio = tf.squeeze(tf_audio, 1)
    if tf.shape(audio) != 16000:
        zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
        audio = tf.concat([audio, zero_padding], 0)
        audio.set_shape([sampling_rate])
    stft = tf.signal.stft(audio, frame_length, frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
    log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
    mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
    mfccs = mfccs[..., :coefficients]
    mfccs = tf.expand_dims(mfccs, -1)
    mfccs = tf.expand_dims(mfccs, 0)

    return mfccs


zip_path = tf.keras.utils.get_file(
    origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    fname='mini_speech_commands.zip',
    extract=True,
    cache_dir='.', cache_subdir='data')

interpreter = tf.lite.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
predicted_labels = []
CommunicationCost = 0
total_time = 0
for i in range(len(test_files)):
    audio = tf.io.read_file(test_files[i])
    tf_audio, _ = tf.audio.decode_wav(audio)
    start = time.time()
    interpreter.set_tensor(input_details[0]['index'], mfcc(tf_audio))
    interpreter.invoke()
    predict_result = interpreter.get_tensor(output_details[0]['index'])
    end = time.time()
    total_time += end - start
    predicted_label = np.argmax(predict_result)
    softmax_predict_result = softmax(predict_result[0])

    # option 1
    max_prediction = max(list(map(lambda x: float("{:.8f}".format(float(x * 100))), softmax_predict_result)))

    # option 2 for success checker (difference between two biggest probabilities)
    list_of_predictions = list(map(lambda x: float("{:.8f}".format(float(x * 100))), softmax_predict_result))
    sorted_predictions = sorted(list_of_predictions, reverse=True)
    difference = sorted_predictions[0] - sorted_predictions[1]

    if difference < 4 and max_prediction < 65:
        t = tf_audio.numpy().tolist()
        msg = {'Audio': t}
        CommunicationCost += sys.getsizeof(json.dumps(msg))
        try:
            req = requests.put(url, json=msg)
            if req.status_code == 200:
                body = req.json()
                predicted_labels.append(int(body.get('predicted_label')))
            else:
                print('Error:', req.text)
        except requests.exceptions.Timeout:
            print('Timeout !!')
        except requests.exceptions.TooManyRedirects:
            print('Bad URL!!!')
        except requests.exceptions.RequestException:
            print('Big Problem !!')

    else:
        predicted_labels.append(predicted_label)

print("CommunicationCost: {:.2f} MB".format(CommunicationCost * 0.000001))
predicted_labels = np.array(predicted_labels)
acc = tf.keras.metrics.Accuracy()
acc.update_state(predicted_labels, actual_labels)
print("Accuracy: {:.2f}".format(acc.result().numpy() * 100))
print("Total Inference time: {:.2f} ms".format(int(total_time * 1000) / len(test_files)))
