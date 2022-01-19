import base64

url = 'http://0.0.0.0:8080'
url_add = url + '/add'
model_name = '../Prerequisite/kws_dscnn_True.tflite'

with open(model_name, 'rb') as fp:
    model_string = fp.read()
    model_base64 = base64.b64encode(model_string)

body_dict = {'model': model_base64, 'name': 'cnn.tflite'}
