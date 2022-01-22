from MQTT.DoSomething import DoSomething
import time


def get_Alert():
    test = DoSomething("subscriber 1")
    test.run()
    test.myMqttClient.mySubscribe('sdfsdf')
    test.end()
