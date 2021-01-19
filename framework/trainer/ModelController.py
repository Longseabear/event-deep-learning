from framework.app.app import SingletoneInstance, App
from framework.trainer.optimizer import make_optimizer
from framework.dataloader.DataLoader import DataLoaderController
from framework.dataloader.Dataset import BaseDataset
from utils.runtime import get_class_object_from_name
from .losses import LossContainer
from tqdm import tqdm
import torch

class ModelState(object):
    def __init__(self, state='INIT', step=0, epoch=0, loader_name=''):
        self.state = state
        self.step = step
        self.epoch = epoch
        self.loader_name = loader_name

class ModelController(SingletoneInstance):
    @classmethod
    def controller_factory(cls, config=None):
        if config is None:
            config = App.instance().config.MODEL_CONTROLLER

        controller_module = config.MODULE_NAME
        controller_class = config.CLASS_NAME
        controller_module: ModelController = get_class_object_from_name(controller_module, controller_class)
        return controller_module.instance()

    def __init__(self):
        self.state = None
        self.config = App.instance().config.MODEL_CONTROLLER
        self.dataloader_controller = DataLoaderController.instance()
        self.model = None
        self.optimizer = None
        self.loss = App.instance().set_gpu_device(LossContainer(self.config.LOSSES))

    def set_state(self, state):
        self.state = state

    def state_init_run(self):
        self.state.state = 'INIT'
        self.state.epoch = self.optimizer.get_last_epoch()+1
        self.loader = self.dataloader_controller.dataloaders[self.state.loader_name]

    def state_step_run(self):
        self.state.state = 'STEP'

    def state_end_run(self):
        self.state.state = 'END'
        self.optimizer.schedule()

    # INIT / STEP / END
    def train(self):
        self.state_init_run()
        self.state_step_run()
        self.state_end_run()

    def test(self):
        with torch.no_grad():
            loader = self.dataloader_controller.dataloaders[self.state.loader_name]

            for samples in tqdm(loader):
                samples = self.model(samples)

                #optimization
                self.optimizer.zero_grad()
                loss = self.loss_step(samples)
