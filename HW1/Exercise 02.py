from subprocess import Popen
import tensorflow as tf
import os
import time
import numpy as np
from numpy import linalg as LNG

Popen('sudo sh -c "echo performance >'
      '/sys/devices/system/cpu/cpufreq/policy0/scaling_governor"',
      shell=True).wait()


def SNR(mfcc_slow, mfcc_fast):
    numerator = LNG.norm(mfcc_slow)
    denominator = LNG.norm(mfcc_slow - mfcc_fast + 10 ** -6)
    snr_result = 20 * np.log(numerator / denominator)
    print('SNR = ' + str(snr_result) + ' dB')


def mfcc(bin, Audio, type):
    tot_mfcc = 0
    total_time = 0
    linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
        bin, 129, sampling_rate, lower_freq, upper_freq)
    for i in Audio:
        tf_audio, rate = tf.audio.decode_wav(i)
        tf_audio = tf.squeeze(tf_audio, 1)
        start = time.time()
        stft = tf.signal.stft(tf_audio, frame_length, frame_step, fft_length=frame_length)
        spectrogram = tf.abs(stft)
        num_spectrogram_bins = spectrogram.shape[-1]
        mel_spectrogram = tf.tensordot(spectrogram, linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :coefficients]
        # print(mfccs.sha)
        end = time.time()
        tot_mfcc = tot_mfcc + mfccs
        temp_time = end - start
        total_time = total_time + temp_time
    avg_time = total_time / len(audio)
    avg_mfcc = tot_mfcc / len(audio)
    print('MFCC ' + type + ' = ', avg_time * 1000, ' ms')
    return avg_mfcc



sampling_rate = 16000
lower_freq = 20
upper_freq = 4000
coefficients = 10
frame_length = int(0.016*sampling_rate)
frame_step = int(0.008*sampling_rate)






DATASET_PATH = './yes_no'
dir_list = os.listdir(DATASET_PATH)
audio = []
for i in dir_list:
    audio.append(tf.io.read_file(DATASET_PATH + '/' + i))

mfcc_slow = mfcc(bin=40, Audio=audio, type='slow')
mfcc_fast = mfcc(bin=32, Audio=audio, type='fast')

SNR(mfcc_slow=mfcc_slow, mfcc_fast=mfcc_fast)
