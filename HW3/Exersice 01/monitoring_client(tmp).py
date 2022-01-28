import json
from MQTT.DoSomething import DoSomething


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        print("({}) {} Alert: Predicted={}% Actual={}%".format(input_json['dateTime'], input_json['Quantity'],
                                                               input_json['Predicted'], input_json['Actual']))


if __name__ == "__main__":
    test = Subscriber("Monitoring Client")
    test.run()
    test.myMqttClient.mySubscribe("ML4IOT")
