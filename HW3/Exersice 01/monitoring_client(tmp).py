import json
import time
from DoSomething import DoSomething


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        if input_json['e'][0]['n'] == 'Temperature':
            print("{} {} Alert: Predicted={}°C Actual={}°C".format(input_json['e'][0]['t'], input_json['e'][0]['n'],
                                                                   input_json['e'][1]['v'], input_json['e'][0]['v']))
        elif input_json['e'][0]['n'] == 'Humidity':
            print("{} {} Alert: Predicted={}% Actual={}%".format(input_json['e'][0]['t'], input_json['e'][0]['n'],
                                                                 input_json['e'][1]['v'], input_json['e'][0]['v']))


if __name__ == "__main__":
    test = Subscriber("Monitoring Client")
    test.run()
    test.myMqttClient.mySubscribe("ML4IOT")
    while True:
        time.sleep(1)
