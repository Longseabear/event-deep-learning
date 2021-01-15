from importlib import import_module
from framework.trainer.ModelController import ModelController
import sys

class CommandFactory(object):
    @staticmethod
    def make_command(name, config):
        current_module = sys.modules[__name__]
        return getattr(current_module, config.command)(name, config)

class CommandController(object):
    def __init__(self):
        self.commands = []

    def run(self, *args, **kwargs):
        for command in self.commands:
            command: Command
            # with command as cs:

class Command(object):
    def __init__(self, name, config):
        self.name = name
        self.config = config

        self.state, self.step = self.config.when.split(':')
        self.step = int(self.step)
        self.config.repeat = int(self.config.repeat)

    def update(self):
        if self.config.repeat > 0:
            self.config.repeat -= 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.config.ERROR= '{} {} {}'.format(exc_type, exc_val, exc_tb)
            print(self.config.ERROR)
        return '{} {} {}'.format(exc_type, exc_val, exc_tb)

    def __enter__(self):
        return self

    def valid(self, sample):
        model_info = ModelController.instance().model_variable
        if self.state != model_info['state'] or model_info['step'] % self.step != 0:
            return False
        return True

class PrintCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self, sample):
        if not super(PrintCommand, self).valid(sample):
            return
        print('PRINT MODULE: ', sep=' ')
        for name in self.config.required:
            print(name, sample[name]._data, sep=' ')
        return


if __name__ == '__main__':
    import sys
    from framework.dataloader.MetaData import ImageMeta
    from utils.config import Config
    a = CommandFactory.make_command('name', Config.from_dict({'command':'PrintCommand',
                                             'repeat':5,
                                             'args':{},
                                             'when':'INIT:100',
                                             'required': ['img','t']}))

    model_info = ModelController.instance().model_variable
    for i in range(300):
        model_info['step'] += 1
        try:
            with a as f:
                f.run({'img': ImageMeta('af', -1)})
                f.update()
        except Exception as e:
            print(e, 'a')
            pass