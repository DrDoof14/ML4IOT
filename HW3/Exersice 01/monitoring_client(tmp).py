import requests
from MQTT.subscriber import get_Alert

url = 'http://127.0.0.1:8080'

model_name = 'CNN.tflite'
tthres = 0.1
hthres = 0.2
url_predict = url + '/{}?model={}&tthres={}&hthres={}'.format('predict', model_name, tthres, hthres)
req = requests.get(url_predict)
if req.status_code == 200:
    get_Alert()
    body = req.json()
    print(body)
else:
    print('Error:', req.text)
