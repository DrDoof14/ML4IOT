import base64
import requests

url = 'http://192.168.1.145:8080'


def model_add(model_name):
    url_add = url + '/add'
    with open(model_name, 'rb') as fp:
        model_string = fp.read()
        model_base64 = base64.b64encode(model_string)
    body_dict = {'model': model_base64.decode(), 'name': model_name}
    r = requests.post(url_add, json=body_dict)
    if r.status_code != 200:
        print(400)
    else:
        print(200)


def model_list():
    url_list = url + '/list'
    req = requests.get(url_list)
    if req.status_code == 200:
        print(req.json())
    #         if len(req.json()) == 2:
    #             print("2 model exist")
    else:
        print('Error:', req.status_code)


def predict(model_name='cnn.tflite'):
    tthres = 0.1
    hthres = 0.2
    url_predict = url + '/{}?model={}&tthres={}&hthres={}'.format('predict', model_name, tthres, hthres)
    req = requests.get(url_predict)
    if req.status_code != 200:
        print('Error:', req.status_code)


if __name__ == "__main__":
    modelName = ['CNN.tflite', 'MLP.tflite']
    model_add(modelName[0])
    model_add(modelName[1])
    model_list()
    predict(model_name=modelName[0])
