import cherrypy
from os.path import isfile, join
from os import listdir


class AddModel:
    exposed = True  # Needed for exposing the Web Services

    def GET(self, *path, **query):
        # if path[0] == 'add':
        #     save_path = './model' + self.model_name + '.tflite'
            # with open(save_path, 'wb') as f:
            #     f.write(self.tflite_model)
        print(path)
        # else:
        #     raise cherrypy.HTTPError(400, 'path is not correct')


class ListModels:
    exposed = True  # Needed for exposing the Web Services

    def GET(self):
        models_path = './model'
        onlyfiles = [f for f in listdir(models_path) if isfile(join(models_path, f))]
        print(onlyfiles)


class Predict:
    exposed = True  # Needed for exposing the Web Services

    print('Nothing')


if __name__ == '__main__':
    # conf probably needs modification
    conf = {
        '/': {
            'request.dispatch': cherrypy.dispatch.MethodDispatcher(),
            'tools.sessions.on': True,
        }
    }
    cherrypy.tree.mount(AddModel(), '/add',conf)
    cherrypy.tree.mount(ListModels, '/list')
    cherrypy.tree.mount(Predict, '/predict')

    # To start cherrypy engine
    cherrypy.engine.start()
    cherrypy.engine.block()
