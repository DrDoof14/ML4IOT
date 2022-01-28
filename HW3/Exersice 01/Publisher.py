import paho.mqtt.client as PahoMQTT
from MQTT.MyMQTT import MyMQTT
from MQTT.DoSomething import DoSomething
import json
from datetime import datetime

def Alert(type_alert, predicted, actual):
    now = datetime.now()
    dt_string = now.strftime("(%d/%m/%Y %H:%M:%S)")
    msg = {'datTime': dt_string, 'Quantity': type_alert, 'Predicted': predicted, 'Actual': actual}
    publisher = DoSomething("publisher 1")
    publisher.run()
    publisher.myMqttClient.myPublish('fdsfs', json.dumps(msg))
    publisher.end()






