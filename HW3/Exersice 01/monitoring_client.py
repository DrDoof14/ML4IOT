import json
import time
from DoSomething import DoSomething


class Subscriber(DoSomething):
    def notify(self, topic, msg):
        input_json = json.loads(msg)
        print("{} {} Alert: Predicted={}{} Actual={}{}".format(input_json['e'][0]['t'], input_json['e'][0]['n'],
                                                               input_json['e'][1]['v'], input_json['e'][0]['u'],
                                                               input_json['e'][0]['v'], input_json['e'][0]['u']))
        

        
if __name__ == "__main__":
    test = Subscriber("Monitoring Client")
    test.run()
    test.myMqttClient.mySubscribe('ML4IOT/2022/289456/alert')
    while True:
        time.sleep(1)
