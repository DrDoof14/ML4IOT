import json
import cherrypy
import tensorflow as tf
import numpy as np


class SLowService:
    exposed = True

    def __init__(self):
        model_path = "../Prerequisite/kws_dscnn_True.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.sampling_rate = 16000
        self.frame_length = int(0.04 * self.sampling_rate)
        self.frame_step = int(0.02 * self.sampling_rate)
        self.num_mel_bins = 40
        self.lower_freq = 20
        self.upper_freq = 4000
        self.coefficients = 10
        self.num_spectrogram_bins = self.frame_length // 2 + 1
        self.linear_to_mel_weight_matrix = tf.signal.linear_to_mel_weight_matrix(
            self.num_mel_bins, self.num_spectrogram_bins, self.sampling_rate, self.lower_freq, self.upper_freq)
        seed = 42
        tf.random.set_seed(seed)
        np.random.seed(seed)

    def mfcc(self, audio):
        tf_audio = tf.squeeze(audio, 1)
        zero_padding = tf.zeros([self.sampling_rate] - tf.shape(tf_audio), dtype=tf.float32)
        audio = tf.concat([tf_audio, zero_padding], 0)
        audio.set_shape([self.sampling_rate])
        stft = tf.signal.stft(audio, self.frame_length, self.frame_step, fft_length=self.frame_length)
        spectrogram = tf.abs(stft)

        mel_spectrogram = tf.tensordot(spectrogram, self.linear_to_mel_weight_matrix, 1)
        log_mel_spectrogram = tf.math.log(mel_spectrogram + 1.e-6)
        mfccs = tf.signal.mfccs_from_log_mel_spectrograms(log_mel_spectrogram)
        mfccs = mfccs[..., :self.coefficients]
        mfccs = tf.expand_dims(mfccs, -1)
        mfccs = tf.expand_dims(mfccs, 0)

        return mfccs

    def PUT(self, **query):
        body = cherrypy.request.body.read()
        body = json.loads(body)
        audio = body.get('Audio')
        audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], self.mfcc(audio_tensor))
        self.interpreter.invoke()
        predict_result = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_label = np.argmax(predict_result)
        msg = {'predicted_label': str(predicted_label)}
        return json.dumps(msg)


if __name__ == '__main__':
    # conf probably needs modification
    #have to find the IP for the device you are using as the server 
    cherrypy.config.update({'server.socket_host': '192.168.1.127'})
    cherrypy.config.update({'server.socket_port': 8080})
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
        }
    }
    cherrypy.tree.mount(SLowService(), '/predict', conf)
    # To start cherrypy engine
    cherrypy.engine.start()
    cherrypy.engine.block()






