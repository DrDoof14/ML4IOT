import requests
import tensorflow as tf
import numpy as np
from scipy.special import softmax

url = 'http://127.0.0.1:8080'
sampling_rate = 16000
frame_length = int(0.04 * sampling_rate)
frame_step = int(0.02 * sampling_rate)
num_mel_bins = 40
lower_freq = 20
upper_freq = 4000
coefficients = 10
num_spectrogram_bins = frame_length // 2 + 1
model_path = "../Prerequisite/kws_dscnn_True.tflite"
test_files = list(open('../Prerequisite/kws_test_split.txt', 'r'))
test_files = [s.rstrip() for s in test_files]
LABELS = ['stop', 'up', 'yes', 'right', 'left', 'no', 'down', 'go']
seed = 42
tf.random.set_seed(seed)
np.random.seed(seed)

actual_label = []
for i in test_files:
    tmp = i.replace('./data/mini_speech_commands/', '')
    loc_slash = tmp.find('/')
    actual_label.append(LABELS.index(tmp[:loc_slash]))
actual_label = np.array(actual_label)


def mfcc(audio):
    tf_audio, rate = tf.audio.decode_wav(audio)
    tf_audio = tf.squeeze(tf_audio, 1)
    stft = tf.signal.stft(tf_audio, frame_length, frame_step, fft_length=frame_length)
    spectrogram = tf.abs(stft)
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins, num_spectrogram_bins, sampling_rate, lower_freq, upper_freq)
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
for i in range(len(test_files)):
    audio = tf.io.read_file(test_files[i])
    interpreter.set_tensor(input_details[0]['index'], mfcc(audio))
    interpreter.invoke()
    predict_result = interpreter.get_tensor(output_details[0]['index'])
    predicted_label = np.argmax(predict_result)
    print(predict_result[0])
    print(predicted_label)
    softmax_predict_result = softmax(predict_result[0])
    max_prediction = max(list(map(lambda x: float("{:.8f}".format(float(x * 100))), softmax_predict_result)))
    print(max_prediction)
    if max_prediction < 65:
        msg = {'Audio': audio}
        req = requests.post(url, json=msg)
        if req.status_code == 200:
            body = req.json()
            print(body)
        else:
            print('Error:', req.json())
    else:
        predicted_labels.append(predicted_label)
