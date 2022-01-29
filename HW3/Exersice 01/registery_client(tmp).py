import base64
import json
import requests

url = 'http://127.0.0.1:8080'


def model_add(model_name='cnn.tflite'):
    url_add = url + '/add'
    with open(model_name, 'rb') as fp:
        model_string = fp.read()
        model_base64 = base64.b64encode(model_string)
    body_dict = {'model': model_base64.decode(), 'name': model_name}
    r = requests.post(url_add, json=body_dict)
    if r.status_code == 200:
        print(r.text)
    else:
        print('Error:', r.json())


def model_list():
    url_list = url + '/list'
    req = requests.get(url_list)
    if req.status_code == 200:
        print(req.text)
    else:
        print('Error:', req.text)


def predict(model_name='cnn.tflite'):
    tthres = 0.1
    hthres = 0.2
    url_predict = url + '/{}?model={}&tthres={}&hthres={}'.format('predict', model_name, tthres, hthres)
    req = requests.put(url_predict)
    if req.status_code == 200:
        body = req.json()
        print(body)
    else:
        print('Error:', req.text)


if __name__ == "__main__":
    model_add()
    model_list()
    predict()
