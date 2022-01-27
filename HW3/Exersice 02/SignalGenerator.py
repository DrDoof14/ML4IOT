import tensorflow as tf
import numpy as np



def mfcc(audio):
    #maybe its better to put variables some place else
    sampling_rate = 16000
    frame_length = int(0.04 * sampling_rate)
    frame_step = int(0.02 * sampling_rate)
    num_mel_bins = 40
    lower_freq = 20
    upper_freq = 4000
    coefficients = 10
    num_spectrogram_bins = frame_length // 2 + 1
    seed = 42
    tf.random.set_seed(seed)
    np.random.seed(seed)

    # tf_audio, rate = tf.audio.decode_wav(audio)
    tf_audio = tf.squeeze(audio, 1)
    # zero_padding = tf.zeros([sampling_rate] - tf.shape(audio), dtype=tf.float32)
    # audio = tf.concat([audio, zero_padding], 0)
    # audio.set_shape([sampling_rate])
    stft = tf.signal.stft(audio, frame_length, frame_step, fft_length=frame_length)
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