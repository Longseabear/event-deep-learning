from framework.app.app import SingletoneInstance, App
from framework.trainer.optimizer import make_optimizer
from framework.dataloader.DataLoader import DataLoaderController
from framework.trainer.ModelController import ModelController
from framework.model.BaseModel import BaseModel
from framework.app.app import App
from tqdm import tqdm
import torch

class ExampleContoller(ModelController):
    def __init__(self):
        super(ExampleContoller, self).__init__()

        self.MODEL = BaseModel.model_factory(self.config.MODEL.MODEL_CONFIG_PATH)
        self.MODEL = App.instance().set_gpu_device(self.MODEL)

        print(self.MODEL.description())
        if self.MODEL is not None:
            self.OPTIMIZER = make_optimizer(self.config.OPTIMIZER, self.MODEL)

        self.all_callable = [method_name for method_name in dir(self) if callable(getattr(self, method_name))]

    def train(self):
        try:
            model_state = self.get_state()
            self.sample = next(model_state.iter)
            self.sample = self.MODEL(self.sample)
            self.OPTIMIZER.zero_grad()
            loss = self.LOSSES(self.sample)
            model_state.tqdm.set_postfix_str("loss: {}".format(loss))
            loss.backward()
            self.OPTIMIZER.step()
        except StopIteration:
            return {'state_name': 'EPOCH_END'}
        return

    def test(self):
        try:
            with torch.no_grad():
                model_state = self.get_state()
                self.sample = next(model_state.iter)
                self.sample = self.MODEL(self.sample)
                loss = self.LOSSES(self.sample)
                model_state.tqdm.set_postfix_str("loss: {}".format(loss))
        except StopIteration:
            return {'state_name': 'EPOCH_END'}
        return