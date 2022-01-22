import requests
from MQTT.DoSomething import DoSomething

url = 'http://127.0.0.1:8080'

model_name = 'CNN.tflite'
tthres = 0.1
hthres = 0.2
url_predict = url + '/{}?model={}&tthres={}&hthres={}'.format('predict', model_name, tthres, hthres)
test = DoSomething("subscriber 1")
test.run()
req = requests.put(url_predict)
if req.status_code == 200:
    test.myMqttClient.mySubscribe('aaa')
    body = req.text
    print(body)
else:
    print('Error:', req.text)
