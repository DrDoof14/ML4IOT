import base64
import json
import cherrypy
import tensorflow as tf
from SignalGenerator import mfcc
import numpy as np


class SLowService:
    exposed = True

    def __init__(self):
        model_path = "../Prerequisite/kws_dscnn_True.tflite"
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def PUT(self, **query):
        body = cherrypy.request.body.read()
        body = json.loads(body)
        audio = body.get('Audio')
        audoio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        self.interpreter.set_tensor(self.input_details[0]['index'], mfcc(audoio_tensor))
        self.interpreter.invoke()
        predict_result = self.interpreter.get_tensor(self.output_details[0]['index'])
        predicted_label = np.argmax(predict_result)
        msg = {'predicted_label': str(predicted_label)}
        return json.dumps(msg)


if __name__ == '__main__':
    # conf probably needs modification
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
