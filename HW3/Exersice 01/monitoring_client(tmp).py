import json
import time
from DoSomething import DoSomething


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        if input_json['Quantity'] == 'Temperature':
            print("{} {} Alert: Predicted={}°C Actual={}°C".format(input_json['dateTime'], input_json['Quantity'],
                                                                   input_json['Predicted'], input_json['Actual']))
        elif input_json['Quantity'] == 'Humidity':
            print("{} {} Alert: Predicted={}% Actual={}%".format(input_json['dateTime'], input_json['Quantity'],
                                                                 input_json['Predicted'], input_json['Actual']))


if __name__ == "__main__":
    test = Subscriber("Monitoring Client")
    test.run()
    test.myMqttClient.mySubscribe("ML4IOT")
    while True:
        time.sleep(1)
