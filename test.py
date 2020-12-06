from utils.config import Config
from framework.app.app import App

if __name__ == '__main__':
    test = Config.from_yaml('./a.yaml')
    test2 = Config.from_yaml('./b.yaml')
    from framework.app.app import App
    a = App.make_from_config_list(['./a.yaml'])
    print(a.config)
