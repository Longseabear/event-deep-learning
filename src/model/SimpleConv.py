import torch
from framework.model.BaseModel import *
from framework.app.app import App


class SimpleLayer(BaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.inputs_name = ['IMAGE']
        self.outputs_name = ['OUTPUT']

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,3,1,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,3,1,1,1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, 3, 1, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 32, 3, 1, 1, 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32,3,3,1,1,1),
            torch.nn.ReLU()
        )

    def forward(self, sample):
        sample = super(SimpleLayer, self).forward(sample)
        sample['OUTPUT'] = self.layers(sample['IMAGE'])
        return sample

