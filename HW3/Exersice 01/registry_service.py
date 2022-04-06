import base64
import os
from datetime import datetime
import json
import cherrypy
from os.path import isfile, join
from os import listdir
from DoSomething import DoSomething
import adafruit_dht
from board import D4
import tensorflow as tf
import numpy as np
import time


class AddModel:
    exposed = True  # Needed for exposing the Web Services

    def POST(self, **query):
        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Query detected! No query needed')
        body = cherrypy.request.body.read()
        body = json.loads(body)
        if len(body) > 2:
            raise cherrypy.HTTPError(400, 'Body is longer than needed')
        if not body.get('name').endswith('.tflite'):
            raise cherrypy.HTTPError(400, 'The name extention should be tflite')

        model = body.get('model')
        model_name = body.get('name')
        decoded_model = base64.b64decode(model)
        path = './model/' + str(model_name)
        with open(path, 'wb') as f:
            f.write(decoded_model)


class ListModels:
    exposed = True  # Needed for exposing the Web Services

    def GET(**query):
        if len(query) > 0:
            raise cherrypy.HTTPError(400, 'Query detected! No query needed')
        models_path = './model'
        onlyFiles = [f for f in listdir(models_path) if isfile(join(models_path, f))]
        tfModels = []
        for i in onlyFiles:
            if i.endswith('.tflite'):
                tfModels.append(i)
        return json.dumps(tfModels)


class Predict:
    exposed = True  # Needed for exposing the Web Services

    def __init__(self):
        self.dht_device = adafruit_dht.DHT11(D4)
        self.publisher = DoSomething("publisher 1")

    def GET(self, **query):
        self.publisher.run()
        if len(query) != 3:
            raise cherrypy.HTTPError(400, 'length of queries not long enough!')
        model_name = query.get('model')
        if type(model_name) != str:
            raise cherrypy.HTTPError(400, 'Chose the right type for the model name!')
        tthres = float(query.get('tthres'))
        hthres = float(query.get('hthres'))
        models_path = './model/' + model_name
        if os.path.isfile(models_path) == False:
            raise cherrypy.HTTPError(400, 'No model exists with this name!')
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
                # |msg = {'dateTime': dt_string, 'Quantity': 'Temperature', 'Predicted': predict_result[0][0],
                #                        'Actual': temperature_list[5]}
                msg = {
                    "bn": 'rpi_temp',
                    "e": [
                        {'n': 'Temperature', "u": "Cel",
                         't': dt_string,
                         'v': str(temperature_list[5])},
                        {'n': 'Temperature', "u": "Cel", 't': dt_string,
                         'v': str(predict_result[0][0])}]
                }

            self.publisher.myMqttClient.myPublish('ML4IOT/2022/289456/alert', json.dumps(msg))
            if abs(predict_result[0][1] - humidity_list[5]) > hthres:
                #                 msg = {'dateTime': dt_string, 'Quantity': 'Humidity', 'Predicted': predict_result[0][1],
                #                        'Actual': humidity_list[5]}
                msg = {
                    "bn": 'rpi_hum',
                    "e": [
                        {'n': 'Humidity', "u": "%",
                         't': dt_string,
                         'v': str(humidity_list[5])},
                        {'n': 'Humidity', "u": "%", 't': dt_string,
                         'v': str(predict_result[0][1])}]
                }
                self.publisher.myMqttClient.myPublish('ML4IOT/2022/289456/alert', json.dumps(msg))


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
    cherrypy.config.update({'server.socket_host': '192.168.1.145'})
    cherrypy.config.update({'server.socket_port': 8080})

    # To start cherrypy engine
    cherrypy.engine.start()
    cherrypy.engine.block()
