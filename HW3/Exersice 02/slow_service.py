import base64
import json
import cherrypy
import tensorflow as tf
from SignalGenerator import mfcc
import numpy as np


class SLowService:
    exposed = True

    # @cherrypy.tools.json_in()
    # @cherrypy.tools.json_out()
    def PUT(self, **query):
        body = cherrypy.request.body.read()
        body = json.loads(body)
        audio = body.get('Audio')
        audoio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)
        print(audoio_tensor)

        # audio=base64.b64decode(audio.encode())
        # model_path = "../Prerequisite/kws_dscnn_True.tflite"
        # interpreter = tf.lite.Interpreter(model_path=model_path)
        # interpreter.allocate_tensors()
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # interpreter.set_tensor(input_details[0]['index'], mfcc(audio))
        # interpreter.invoke()
        # predict_result = interpreter.get_tensor(output_details[0]['index'])
        # predicted_label = np.argmax(predict_result)
        # msg = {'predicted_label': predicted_label}
        # return json.dumps(msg)
        # return audio


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
