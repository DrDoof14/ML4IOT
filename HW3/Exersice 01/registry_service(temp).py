import base64
import cherrypy
from os.path import isfile, join
from os import listdir


class AddModel:
    exposed = True  # Needed for exposing the Web Services

    def POST(self, **query):
        model = query.get('model')
        model_name = query.get('name')
        decoded_model = base64.b64decode(model)
        path = './model/' + str(model_name)
        with open(path, 'wb') as f:
            f.write(decoded_model)


class ListModels:
    exposed = True  # Needed for exposing the Web Services

    def GET(self):
        models_path = './model'
        onlyFiles = [f for f in listdir(models_path) if isfile(join(models_path, f))]
        print(onlyFiles)


class Predict:
    exposed = True  # Needed for exposing the Web Services

    # print('Nothing')


if __name__ == '__main__':
    # conf probably needs modification
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
        }
    }
    cherrypy.tree.mount(AddModel(), '/add', conf)
    cherrypy.tree.mount(ListModels, '/list', conf)
    cherrypy.tree.mount(Predict, '/predict', conf)

    # To start cherrypy engine
    cherrypy.engine.start()
    cherrypy.engine.block()
