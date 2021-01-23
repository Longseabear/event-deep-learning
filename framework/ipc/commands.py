from .Command import Command
from framework.dataloader.TensorTypes import TensorType
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from framework.app.app import App
from framework.app.Format import Formatter
from framework.dataloader.Transform import TRANSFORM
from framework.dataloader.TensorTypes import *
from framework.dataloader.DataLoader import DataLoaderController
import scipy.misc as misc
from framework.ipc.ThreadCommand import *
from utils.config import Config
import os

class RunCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, Config.from_dict({}).update(config))

    def run(self, controller):
        output = {}
        for name in self.config.required:
            if name in controller.all_callable:
                out = controller.__getattribute__(name)(**self.config.args)
                if out is not None:
                    output.update(out)
        return output

class PrintCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self, controller):
        print('PRINT MODULE: {}'.format(self.config.args.content), sep=' ')
        for name in self.config.required:
            print(name, controller.sample[name], sep=' ')

class PrintStateCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self, controller):
        if controller.get_state().tqdm is not None:
            controller.get_state().tqdm.write('[{}/{}]: {}'.format(controller.get_state().loader_name, controller.get_state().state_name,
                                       [(name,controller.get_state().__getattribute__(name)) for name in self.config.required]))

class BatchedImageShowCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name + "_output" for name in self.config.required], {'ALL': IMAGE()})

    def run(self, controller):
        temp_sample = {}
        for name in self.config.required:
            temp_sample[name + "_output"] = controller.sample[name]
        temp_sample = self.numpy_trasnsform(temp_sample)

        for key in temp_sample.keys():
            plt.imshow(temp_sample[key][self.config.args.batch_number])
            plt.show()

class PushModelStateCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, Config.from_dict({}).update(config))

    def run(self, controller):
        add_state_info = []
        for name in self.config.required:
            args = self.config.args[name] if name in self.config.args.keys() else None
            add_state_info.append((name, args))
        return {'push_state': add_state_info}

import queue
class BatchedImageSaveCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name for name in self.config.required], {'ALL': IMAGE()})
        self.type = IMAGE()

    def leave(self):
        if self.live or not MultipleProcessorController.instance().finished(self.__class__.__name__):
            return True
        else:
            MultipleProcessorController.instance().remove_process(self.__class__.__name__)
            return False

    def run(self, controller):
        args = []
        base_path = self.config.args.get('base_path', App.instance().get_base_path(), possible_none=False)
        folder = self.config.args.get('folder_name', 'visual/img', possible_none=False)
        fm = self.config.args.get('format', 'png', possible_none=False)

        App.instance().make_save_dir(folder, base_path)

        formatter = Formatter(controller, {'content': '', 'format': fm, 'batch':0})
        for name in self.config.required:
            formatter.contents['content'] = name
            imgs = self.numpy_trasnsform({name:controller.sample[name].clone()})

            b,_,_,_ = imgs[name].shape
            for i in range(b):
                formatter.contents['batch'] = str(i).zfill(4)
                path = os.path.join(base_path, folder,
                                    formatter.MainStateBasedFormatting('[$main:epoch:03]e_[$main:step:08]s_[$content]_[$batch].[$format]'))
                args.append((path, imgs[name][i]))

        import time
        def batched_image_save(queue):
            while True:
                sample = queue.get()
                if sample is None: break
                path, img = sample
                misc.imsave(path, img)
                time.sleep(0.001)
        MultipleProcessorController.instance().push_data(self.__class__.__name__, batched_image_save, args, num_worker=1)
