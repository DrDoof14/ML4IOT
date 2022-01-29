import base64
from datetime import datetime
import json
import cherrypy
from os.path import isfile, join
from os import listdir
from DoSomething import DoSomething
import adafruit_dht
import tensorflow as tf
import numpy as np
import time


class AddModel:
    exposed = True  # Needed for exposing the Web Services

    def POST(self, **query):
        # if len(query) != 2:
        #     raise cherrypy.HTTPError(400, 'Wrong query')

        # if type(query.get('model')) != 'bytes':
        #     raise cherrypy.HTTPError(400, 'Model type is not bytes')
        #
        # if not query.get('name').endswith('.tflite'):
        #     raise cherrypy.HTTPError(400, 'The name extention should be tflite')

        body = cherrypy.request.body.read()
        body = json.loads(body)
        print(body)
        model = body.get('model')
        model_name = body.get('name')
        decoded_model = base64.b64decode(model)
        path = './model/' + str(model_name)
        with open(path, 'wb') as f:
            f.write(decoded_model)


class ListModels:
    exposed = True  # Needed for exposing the Web Services

    def GET():
        models_path = './model'
        onlyFiles = [f for f in listdir(models_path) if isfile(join(models_path, f))]
        for i in onlyFiles:
            if i.endswith('.tflite'):
                print(i)


class Predict:
    exposed = True  # Needed for exposing the Web Services

    def __init__(self):
        self.dht_device = adafruit_dht.DHT11(D4)
        self.publisher = DoSomething("publisher 1")

    def PUT(self, **query):
        self.publisher.run()
        model_name = query.get('model')
        tthres = float(query.get('tthres'))
        hthres = float(query.get('hthres'))
        models_path = './model/' + model_name
        self.publisher.run()
        interpreter = tf.lite.Interpreter(model_path=models_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        temperature_list = []
        humidity_list = []
        for i in range(6):
            temperature_list.append(self.dht_device.temperature)
            humidity_list.append(self.dht_device.humidity)
            if i != 5:
                time.sleep(1)
        while True:
            input_data = np.array([temperature_list, humidity_list], dtype=np.float32)
            input_data = input_data.reshape(1, 6, 2)
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            predict_result = interpreter.get_tensor(output_details[0]['index'])
            del temperature_list[0]
            del humidity_list[0]
            time.sleep(1)
            temperature_list.append(self.dht_device.temperature)
            humidity_list.append(self.dht_device.humidity)
            now = datetime.now()
            dt_string = now.strftime("(%d/%m/%Y %H:%M:%S)")
            if abs(predict_result[0][0] - temperature_list[5]) > tthres:
                msg = {'dateTime': dt_string, 'Quantity': 'Temperature', 'Predicted': predict_result[0][0],
                       'Actual': temperature_list[5]}
                self.publisher.myMqttClient.myPublish('ML4IOT', json.dumps(msg))

            if abs(predict_result[0][1] - humidity_list[5]) > hthres:
                msg = {'dateTime': dt_string, 'Quantity': 'Humidity', 'Predicted': predict_result[0][1],
                       'Actual': humidity_list[5]}
                self.publisher.myMqttClient.myPublish('ML4IOT', json.dumps(msg))


if __name__ == '__main__':
    # conf probably needs modification
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
        }
    }
    cherrypy.tree.mount(AddModel(), '/add', conf)
    cherrypy.tree.mount(ListModels, '/list', conf)
    cherrypy.tree.mount(Predict(), '/predict', conf)

    # To start cherrypy engine
    cherrypy.engine.start()
    cherrypy.engine.block()
