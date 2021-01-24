from .Command import Command
from framework.dataloader.TensorTypes import TensorType
import matplotlib.pyplot as plt
import numpy as np
import torch
import copy
from framework.app.app import App
from framework.app.Format import *
from framework.dataloader.Transform import TRANSFORM
from framework.dataloader.TensorTypes import *
from framework.dataloader.DataLoader import DataLoaderController
import scipy.misc as misc
from framework.ipc.ThreadCommand import *
from utils.config import Config
import os

def write_log(contents, controller, **kwargs):
    if controller.get_state().tqdm is None:
        print(contents, **kwargs)
    else:
        controller.get_state().tqdm.write(contents, **kwargs)

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
        write_log('PRINT MODULE: {}'.format(self.config.args.content), controller, end=' ')
        for name in self.config.required:
            write_log(name + " " + controller.sample[name], controller)

class PrintStateCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self, controller):
        if controller.get_state().tqdm is not None:
            write_log('[{}/{}]: {}'.format(controller.get_state().loader_name, controller.get_state().state_name,
                                       [(name,controller.get_state().__getattribute__(name)) for name in self.config.required]),
                      controller)

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
        super().__init__(name, config)

    def run(self, controller):
        add_state_info = []
        for name in self.config.required:
            args = copy.deepcopy(self.config.args[name]) if name in self.config.args.keys() else None
            add_state_info.append((name, args))
        return {'push_state': add_state_info}


def dir_path_parser(path):
    return os.path.join(*App.instance().variable_parsing(path, '/'))

class BatchedImageSaveCommand(Command):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.numpy_trasnsform = TRANSFORM['ToNumpy']([name for name in self.config.required], {'ALL': IMAGE()})
        self.type = IMAGE()

    def leave(self):
        if self.live or not MultipleProcessorController.instance().finished(self.__class__.__name__):
            return True
        else:
            return False

    def destroy(self):
        MultipleProcessorController.instance().remove_process(self.__class__.__name__)

    def run(self, controller):
        args = []
        dir_path = dir_path_parser(self.config.args.get('path', '$base/visual/img', possible_none=False))
        fm = self.config.args.get('format', 'png', possible_none=False)

        App.instance().make_save_dir(dir_path)
        formatter = MainStateBasedFormatter(controller, {'content': '', 'format': fm, 'batch':0},
                              format='[$main:epoch:03]e_[$main:step:08]s_[$content]_[$batch].[$format]')
        for name in self.config.required:
            formatter.contents['content'] = name
            imgs = self.numpy_trasnsform({name:controller.sample[name].clone()})

            b,_,_,_ = imgs[name].shape
            for i in range(b):
                formatter.contents['batch'] = str(i).zfill(4)
                path = os.path.join(dir_path, formatter.Formatting())
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

class ModuleLoadClass(Command):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self, controller):
        for name in self.config.required:
            args = copy.deepcopy(self.config.args[name])
            args['module_name'] = name
            args['controller'] = controller
            args['path'] = args.get('path', '$latest_{}'.format(name), possible_none=False)
            args['load_strict'] = args.get('load_strict', True)

            try:
                if args['path'] == '$latest':
                    args['path'] = args['path'] + '_{}'.format(name)
                args['path'] = dir_path_parser(args['path'])
            except KeyError as e:
                raise ValueError('App.variables load error. key:{}'.format(args['path']))

            if name not in dir(controller):
                write_log('[ERROR] Load fail: {} is not Modules.'.format(name), controller)
                continue

            module = controller.__getattribute__(name)
            if not isinstance(module, torch.nn.Module):
                write_log('[ERROR] Load fail: {} is not Modules'.format(name), controller)
                continue
            if not callable(getattr(module, 'load')):
                write_log('[ERROR] Load fail: {} must to have load method.'.format(name), controller)
                continue
            module.load(args)
            write_log('[INFO] {} Load scucesses [{}] '.format(self.config.required, args['path']), controller)

class ModuleSaveClass(Command):
    def __init__(self, name, config):
        super().__init__(name, config)

    def run(self, controller):
        for name in self.config.required:
            if name not in dir(controller):
                write_log('[ERROR] save fail: {} is not Modules.'.format(name), controller)
                continue

            args = copy.deepcopy(self.config.args[name])
            args['module_name'] = name
            args['controller'] = controller
            args['path'] = dir_path_parser(args.get('path', '$base/ckpt_{}'.format(name), possible_none=False))

            module = controller.__getattribute__(name)
            if not isinstance(module, torch.nn.Module):
                write_log('[ERROR] save fail: {} is not Modules'.format(name), controller)
                continue
            if not callable(getattr(module, 'save')):
                write_log('[ERROR] save fail: {} must to have save method.'.format(name), controller)
                continue

            module.save(args)
            write_log('[INFO] {} Save scucesses [{}] '.format(self.config.required, App.instance().get_variables('$latest_{}'.format(name))), controller)


