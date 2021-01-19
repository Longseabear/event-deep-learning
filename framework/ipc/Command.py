from importlib import import_module
from framework.trainer.ModelController import ModelController
from framework.dataloader.TensorTypes import TensorType
import matplotlib.pyplot as plt
import sys
import numpy as np
import torch
from framework.dataloader.Transform import TRANSFORM
from framework.dataloader.TensorTypes import *

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
        self.valid_type = []

        self.state, self.step = self.config.when.split(':')
        self.step = int(self.step)
        self.config.repeat = int(self.config.repeat)

    def update(self):
        if self.config.repeat > 0:
            self.config.repeat -= 1

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.config.ERROR = '{} {} {}'.format(exc_type, exc_val, exc_tb)
            print(self.config.ERROR)
        return '{} {} {}'.format(exc_type, exc_val, exc_tb)

    def __enter__(self):
        return self

    def valid(self, sample):
        model_info = ModelController.instance().state
        if self.state != model_info.state or model_info.step % self.step != 0:
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
            print(name, sample[name], sep=' ')
        return

class BatchedImageShow(Command):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name + "_output" for name in self.config.required], {'ALL': IMAGE()})

    def run(self, sample):
        if not super(BatchedImageShow, self).valid(sample):
            return

        temp_sample = {}
        for name in self.config.required:
            temp_sample[name + "_output"] = sample[name]
        temp_sample = self.numpy_trasnsform(temp_sample)

        for key in temp_sample.keys():
            plt.imshow(temp_sample[key][self.config.args.batch_number])
            plt.show()
        return

if __name__ == '__main__':
    import sys
    from utils.config import Config
    a = CommandFactory.make_command('name', Config.from_dict({'command':'BatchedImageShow',
                                             'repeat':5,
                                             'args':{'batch_number':1},
                                             'when':'INIT:100',
                                             'required': ['img','t']}))

    model_info = ModelController.instance().model_variable
    for i in range(300):
        model_info['step'] += 1
        try:
            with a as f:
                f.run({'img': torch.ones((3,3,8,8)),
                       't': torch.zeros((3,3,8,8))})
                f.update()
        except Exception as e:
            print(e, 'a')
            pass