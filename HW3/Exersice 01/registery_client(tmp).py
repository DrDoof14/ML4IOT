import base64
import json

import requests

url = 'http://127.0.0.1:8080'
url_add = url + '/add'
model_name = 'CNN.tflite'

with open(model_name, 'rb') as fp:
    model_string = fp.read()
    model_base64 = base64.b64encode(model_string)

body_dict = {'model': model_base64, 'name': 'cnn.tflite'}

json_body=json.dumps(list(body_dict))
r = requests.post(url_add, json=json_body)
if r.status_code == 200:
    body = r.json()
    print(body)
else:
    print('Error:', r.status_code)
