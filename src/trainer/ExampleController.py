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

        self.model = BaseModel.model_factory(self.config.MODEL.MODEL_CONFIG_PATH)
        self.model = App.instance().set_gpu_device(self.model)

        print(self.model.description())
        if self.model is not None:
            self.optimizer = make_optimizer(self.config.OPTIMIZER, self.model)

    def state_init_run(self):
        super(ExampleContoller, self).state_init_run()

    def state_step_run(self):
        super(ExampleContoller, self).state_step_run()

        # Learning Process 화장실
        pbar = tqdm(self.loader)
        for samples in pbar:
            samples = self.model(samples)
            self.optimizer.zero_grad()
            loss = self.loss(samples)
            pbar.set_postfix_str("loss: {}".format(loss))
            loss.backward()
            self.optimizer.step()

    def state_end_run(self):
        super(ExampleContoller, self).state_end_run()
