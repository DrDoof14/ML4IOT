import base64
from datetime import datetime
import json
import cherrypy
from os.path import isfile, join
from os import listdir
from MQTT.DoSomething import DoSomething
# import adafruit_dht
import tensorflow as tf
import numpy as np
# import test.mosquitto.org as broker
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

    def PUT(self, **query):
        model_name = query.get('model')
        tthres = float(query.get('tthres'))
        hthres = float(query.get('hthres'))
        models_path = './model/' + model_name
        publisher = DoSomething("publisher 1")
        publisher.run()
        # interpreter = tf.lite.Interpreter(model_path=models_path)
        # interpreter.allocate_tensors()
        # input_details = interpreter.get_input_details()
        # output_details = interpreter.get_output_details()
        # dht_device = adafruit_dht.DHT11(D4)
        temperature_list = []
        humidity_list = []
        # for i in range(6):
        #     temperature_list.append(dht_device.temperature)
        #     humidity_list.append(dht_device.humidity)
        #     if i != 5:
        #         time.sleep(1)
        while True:
            # input_data = np.array([temperature_list, humidity_list], dtype=np.float32)
            # input_data = input_data.reshape(1, 6, 2)
            # interpreter.set_tensor(input_details[0]['index'], input_data)
            # interpreter.invoke()
            # predict_result = interpreter.get_tensor(output_details[0]['index'])
            # print(predict_result)
            # del temperature_list[0]
            # del humidity_list[0]
            time.sleep(10)
            # temperature_list.append(dht_device.temperature)
            # humidity_list.append(dht_device.humidity)
            # send alert for Tempreture & Humidity
            # if abs(predict_result[0][0] - temperature_list[5]) > tthres:
            #     Alert(True, predicted=predict_result[0][0], actual=temperature_list[5])
            # if abs(predict_result[0][1] - humidity_list[5]) > hthres:
            #     Alert(False, predicted=predict_result[0][1], actual=humidity_list[5])
            # MQTT CODE
            now = datetime.now()
            dt_string = now.strftime("(%d/%m/%Y %H:%M:%S)")
            if abs(10.1 - 9) > tthres:
                msg = {'datTime': dt_string, 'Quantity': 'Temperature', 'Predicted': 10.1, 'Actual': 9}
                # Alert(type_alert='Temperature', predicted=10.1, actual=9)
            if abs(11 - 15) > hthres:
                msg = {'datTime': dt_string, 'Quantity': 'Humidity', 'Predicted': 11, 'Actual': 15}
                # Alert(type_alert='Humidity', predicted=11, actual=15)
            publisher.myMqttClient.myPublish('aaa', json.dumps(msg))


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
