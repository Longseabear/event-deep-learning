from framework.app.app import SingletoneInstance, App
from framework.trainer.optimizer import make_optimizer
from framework.dataloader.DataLoader import DataLoaderController
from .losses import LossContainer
from tqdm import tqdm
import torch

class ModelController(SingletoneInstance):
    def __init__(self):
        self.config = App.instance().config.MODEL_CONTROLLER
        self.dataloader_controller = DataLoaderController.instance()
        self.model = None

        if self.model is not None:
            self.optimizer = make_optimizer(self.config.OPTIMIZER, self.model)
        self.loss = LossContainer(self.config.LOSSES)

        self.model_variable = {
            'state': 'INIT',
            'step': 1,
        }

    # INIT / STEP / END
    def train(self, info):
        self.model_variable['epoch'] = self.optimizer.get_last_epoch()+1
        self.model_variable['state'] = 'INIT'
        loader = self.dataloader_controller.dataloaders[info.loader_name]

        # Learning Process
        self.model_variable['state'] = 'STEP'
        for samples in tqdm(loader):
            samples = self.model(samples)

            self.optimizer.zero_grad()
            loss = self.loss_step(samples)
            loss.backward()
            self.optimizer.step()

        self.optimizer.schedule()
        self.model_variable['state'] = 'END'


    def test(self, info):
        with torch.no_grad():
            loader = self.dataloader_controller.dataloaders[info.loader_name]

            for samples in tqdm(loader):
                samples = self.model(samples)

                #optimization
                self.optimizer.zero_grad()
                loss = self.loss_step(samples)
