import torch
import torch.nn as nn
from typing import *
from abc import abstractmethod
from framework.dataloader.MetaData import *

class LossFactory(object):
    @staticmethod
    def instantiate_loss(name, config):
        if config.loss_type == 'L1':
            return L1Loss(name, config)
        else:
            raise NotImplementedError(name)

class LossContainer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.loss_dict = nn.ModuleDict()
        for loss_name in config.keys():
            self.loss_dict[loss_name] = LossFactory.instantiate_loss(loss_name, config[loss_name])

    def forward(self, sample):
        total_losses = []
        for key in self.loss_dict.keys():
            inputs = []
            for input_name in self.loss_dict[key].required.keys():
                inputs.append(sample[input_name]._data)

            loss = self.loss_dict[key](*inputs)
            loss = self.loss_dict[key].weight * loss
            total_losses.append(loss)

        total_loss = sum(total_losses)
        return total_loss

class BaseLoss(nn.Module):
    def __init__(self, name=None, config=None):
        super(BaseLoss, self).__init__()
        self.config = config

        self.name = name
        self.weight = config.weight
        self.required = config.data
        self.visual_data = {}

        self.sum = 0
        self.count = 0

    @abstractmethod
    def set(self):
        self.cumsum = 0
        self.count = 0

    @abstractmethod
    def get_data(self):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError

    @abstractmethod
    def update_state(self, loss):
        raise NotImplementedError

class L1Loss(BaseLoss):
    def __init__(self, name, config):
        super().__init__(name, config)
        self.loss = nn.L1Loss()
        self.visual_data = LossMeta(self.name, {})

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        self.update_state(loss)
        self.visual_data._data['VALUE'] = loss.detach()
        return loss

    def update_state(self, loss):
        self.sum += loss.item()

    # def set(self):
    #     pass

    def __str__(self):
        return str(self.sum/self.count)
