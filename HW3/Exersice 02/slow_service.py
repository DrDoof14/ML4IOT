import json
import cherrypy
import tensorflow as tf
from SignalGenerator import mfcc
import numpy as np

class SLowService:
    def POST(self):
        body = cherrypy.request.body.read()
        body = json.loads(body)
        print(body)
        audio = body.get('Audio')
        model_path = "../Prerequisite/kws_dscnn_True.tflite"
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], mfcc(audio))
        interpreter.invoke()
        predict_result = interpreter.get_tensor(output_details[0]['index'])
        predicted_label = np.argmax(predict_result)
        msg={'predicted_label':predicted_label}
        return json.dumps(msg)

