from framework.app.app import SingletoneInstance, App
from framework.trainer.optimizer import make_optimizer
from framework.dataloader.DataLoader import DataLoaderController
from framework.dataloader.Dataset import BaseDataset
from utils.runtime import get_class_object_from_name
from framework.trainer.losses import LossContainer
from framework.ipc.Command import CommandController
from tqdm import tqdm
import torch
import torch
import time
''': ModelState

ModelState: START -> EPOCH_START -> BATCH_START -> BATCH_END -> EPOCH_END -> TERMINATE
                          |-----------------------------------------|
'''
class ModelStateController(object):
    def __init__(self, config):
        self.current_state: ModelState = ModelState(loader_name='idle')
        self.config = config
        self.states = {}
        self.processing_stack = []

        #IDLE INIT
        self.processing_stack.append(('idle',None))
        self.states['idle'] = self.current_state

    def push_state(self, args):
        for loader_name, config in args:
            if (loader_name, config) not in self.processing_stack:
                self.processing_stack.append((loader_name, config))
                self.states[loader_name] = ModelState(loader_name=loader_name)
                DataLoaderController.instance().make_dataset(loader_name, config)
                self.current_state = self.states[loader_name]

    def pop_state(self):
        del self.states[self.processing_stack[-1][0]]
        self.processing_stack.pop()
        self.current_state = self.states[self.processing_stack[-1][0]] if len(self.processing_stack)>0 else None

    def iterator_reset(self):
        self.current_state.tqdm = tqdm(DataLoaderController.instance().dataloaders[self.current_state.loader_name],
                                       bar_format='{l_bar}{bar:10}{r_bar}', ascii=True)
        self.current_state.iter = iter(self.current_state.tqdm)


class ModelState(object):
    def __init__(self, name='START', step=1, epoch=0, loader_name='global'):
        self.state_name = name
        self.step = step
        self.epoch = epoch
        self.loader_name = loader_name
        self.iter = None
        self.tqdm: tqdm = None

class ModelController(SingletoneInstance):
    def __init__(self):
        self.config = App.instance().config.MODEL_CONTROLLER
        self.dataloader_controller = DataLoaderController.instance()

        self.MODEL: torch.nn.Module = None
        self.OPTIMIZER: torch.nn.Module = None
        self.LOSSES: torch.nn.Module = App.instance().set_gpu_device(LossContainer(self.config.LOSSES))

        self.COMMAND_CONTROLLER = CommandController(self.config.COMMAND_CONTROLLER.command_path, self)
        self.MODEL_STATE_CONTROLLER = ModelStateController(self.config.MODEL_STATE_CONTROLLER)
        self.sample = None

        self.all_callable = [method_name for method_name in dir(self) if callable(getattr(self, method_name))]

    @classmethod
    def controller_factory(cls, config=None):
        if config is None:
            config = App.instance().config.MODEL_CONTROLLER

        controller_module = config.MODULE_NAME
        controller_class = config.CLASS_NAME
        controller_module: ModelController = get_class_object_from_name(controller_module, controller_class)
        return controller_module.instance()

    def get_current_loader(self):
        return self.dataloader_controller.dataloaders[self.get_state().loader_name]

    def update_state(self, changed):
        for key in changed.keys():
            if hasattr(ModelStateController, key) and callable(getattr(ModelStateController, key)):
                self.MODEL_STATE_CONTROLLER.__getattribute__(key)(changed[key])
            else:
                self.get_state().__setattr__(key, changed[key])

    def get_state(self) -> ModelState:
        return self.MODEL_STATE_CONTROLLER.current_state

    def get_main_state(self) -> ModelState:
        return self.MODEL_STATE_CONTROLLER.states[self.MODEL_STATE_CONTROLLER.processing_stack[1][0]]

    # def reserve_state_change(self, state_name):
    #     if self.MODEL_STATE_CONTROLLER.current_state.reserve_state is not None:
    #         self.MODEL_STATE_CONTROLLER.current_state.reserve_state = state_name
    #     else:
    #         raise ValueError('double change')

    def change_state_name(self, state_name):
        self.MODEL_STATE_CONTROLLER.current_state.state_name = state_name

    def epoch_start(self):
        self.MODEL_STATE_CONTROLLER.iterator_reset()

    def step_finish(self):
        self.get_state().step += 1

    def epoch_finish(self):
        while self.get_state().state_name == "EPOCH_END":
            self.get_state().epoch += 1
            if self.get_current_loader().config.mode == 'train':
                self.OPTIMIZER.schedule()
            if self.get_state().epoch < self.get_current_loader().config.obj_epoch:
                self.change_state_name("EPOCH_START")
            else:
                self.MODEL_STATE_CONTROLLER.pop_state()

    def batch_run(self):
        self.epoch_start()
        self.get_state().tqdm.desc = "[{}({}/{})] ".format(self.get_state().loader_name, self.get_state().epoch,
                                                          self.get_current_loader().config.obj_epoch)
        self.change_state_name('BATCH_START')
        while True:
            changed = self.COMMAND_CONTROLLER.run()
            self.update_state(changed)
            state_name = self.get_state().state_name

            if state_name == 'BATCH_START': # end
                self.step_finish()
                self.change_state_name('BATCH_END')
            elif state_name == 'BATCH_END':
                self.change_state_name('BATCH_START')
            elif state_name == "EPOCH_END":
                return
            elif state_name == "START":
                return
            else:
                raise NotImplementedError

    def run(self):
        temp_bar = loading()
        while True:
            changed = self.COMMAND_CONTROLLER.run()
            self.update_state(changed)

            if self.get_state().loader_name == 'idle':
                print('\ridle' + next(temp_bar), end='')
                time.sleep(0.5)
                continue

            state_name = self.get_state().state_name
            if state_name == 'START':
                self.change_state_name('EPOCH_START')
            elif state_name == 'EPOCH_START' or state_name.split('_')[0] == 'BATCH':
                self.batch_run()
            elif state_name == 'EPOCH_END':
                self.epoch_finish()
            else:
                raise NotImplementedError

def loading():
    while True:
        for i in range(4):
            yield '.' * i
