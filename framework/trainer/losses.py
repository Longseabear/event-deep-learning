import torch
import torch.nn as nn
from abc import abstractmethod
from framework.utils.config import Config
from framework.app.app import App
import os


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
        self.total_loss = LossEmpty('TotalLoss')
        self.device = App.instance().get_device()

    def forward(self, sample):
        total_losses = []
        for key in self.loss_dict.keys():
            inputs = []
            for name in self.loss_dict[key].required:
                inputs.append(sample[name].to(self.device))
            loss = self.loss_dict[key](*inputs)
            loss = self.loss_dict[key].weight * loss
            total_losses.append(loss)

        total_loss = sum(total_losses)
        self.total_loss(total_loss)
        return total_loss

    def save(self, info):
        state = info['state']
        contents = App.instance().current_time_format() + " {} epoch, {} step\n".format(state.epoch, state.step)
        contents += "|{}: {}\t".format('Total', str(self.total_loss))
        for key in self.loss_dict.keys():
            contents += "|{}: {}\t".format(key, str(self.loss_dict[key]))
        App.instance().smart_write(contents + '\n', info['path'], 'a+')

    def load(self):
        pass

    def get_dir(self, dir_path):
        return os.path.join(dir_path, 'loss.txt')

class BaseLoss(nn.Module):
    def __init__(self, name=None, config=None):
        super(BaseLoss, self).__init__()
        self.config = config

        self.name = name
        self.weight = config.weight
        self.required = config.inputs

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

        self.register_buffer('sum', torch.zeros(1))
        self.register_buffer('count', torch.zeros(1))

    def forward(self, pred, target):
        loss = self.loss(pred, target)
        self.update_state(loss)
        return loss

    def update_state(self, loss):
        self.sum += loss.item()
        self.count += 1

    def __str__(self):
        if self.count==0:
            return "0"
        return str(self.sum.item()/self.count.item())

class LossEmpty(BaseLoss):
    def __init__(self, name, config=None):
        super().__init__(name, Config.from_dict({'weight':1, 'inputs':None}))

        self.register_buffer('sum', torch.zeros(1))
        self.register_buffer('count', torch.zeros(1))

    def forward(self, loss):
        self.update_state(loss)

    def update_state(self, loss):
        self.sum += loss.item()
        self.count += 1

    def __str__(self):
        if self.count==0:
            return "0"
        return str(self.sum.item()/self.count.item())

if __name__ == '__main__':
    config = Config.from_yaml('../../reso   urce/configs/trainer/ExampleController.yaml')
    print(config.MODEL_CONTROLLER.LOSSES)
    config.MODEL_CONTROLLER.LOSSES['image_loss_2'] = Config.from_dict({
        'inputs': ['OUTPUT','GT'], 'loss_type': 'L1', 'weight': 2
    })
    loss = LossContainer(config.MODEL_CONTROLLER.LOSSES)
    a = torch.tensor([4.0,1.0]).view(1,1,1,2).float()
    b = torch.tensor([7.0,3.0]).view(1, 1, 1, 2).float()
    sample = {'OUTPUT':a, 'GT':b}
    loss(sample)
    a = torch.tensor([4.0,1.0]).view(1,1,1,2).float()
    b = torch.tensor([7.0,4.0]).view(1, 1, 1, 2).float()
    sample = {'OUTPUT':a, 'GT':b}
    print(loss.total_loss)
    loss(sample)
    print(loss.total_loss)
    print(loss.loss_dict['image_loss_l1'])