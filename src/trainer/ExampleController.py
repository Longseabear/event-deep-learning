from framework.app.app import SingletoneInstance, App
from framework.trainer.optimizer import make_optimizer
from framework.dataloader.DataLoader import DataLoaderController
from framework.trainer.ModelController import ModelController
from framework.model.BaseModel import BaseModel
from framework.app.app import App
from tqdm import tqdm
import torch
from framework.app.Exceptions import *

class ExampleContoller(ModelController):
    def __init__(self):
        super(ExampleContoller, self).__init__()

        self.MODEL = BaseModel.model_factory(self.config.MODEL.MODEL_CONFIG_PATH)
        self.MODEL = App.instance().set_gpu_device(self.MODEL)

        print(self.MODEL.description())
        if self.MODEL is not None:
            self.OPTIMIZER = make_optimizer(self.config.OPTIMIZER, self.MODEL)

        self.all_callable = [method_name for method_name in dir(self) if callable(getattr(self, method_name))]
        self.sample = None

    def step(self, sample):
        self.sample = sample
        self.sample = self.MODEL(self.sample)

    def train(self):
        try:
            main_graph = self.COMMAND_CONTROLLER.get_current_main_module()
            self.step(next(main_graph.iter))
            self.OPTIMIZER.zero_grad()
            loss = self.LOSSES(self.sample)
            loss.backward()
            self.OPTIMIZER.step()
            main_graph.tqdm.set_postfix_str("loss: {}".format(loss))
            main_graph.total_step += 1
        except StopIteration:
            raise MainGraphStepInterrupt
        return

    def test(self):
        try:
            with torch.no_grad():
                main_graph = self.COMMAND_CONTROLLER.get_current_main_module()
                self.step(next(main_graph.iter))
                loss = self.LOSSES(self.sample)
                main_graph.tqdm.set_postfix_str("loss: {}".format(loss))
                main_graph.total_step += 1
        except StopIteration:
            raise MainGraphStepInterrupt
        return