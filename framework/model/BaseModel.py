import torch
from utils.runtime import get_instance_from_name, get_class_object_from_name
from utils.config import Config
from framework.app.app import App

class BaseModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.inputs_name = []
        self.outputs_name = []

    def description(self):
        return str(self) + "=> IN:{}, OUT:{} ".format(
            self.inputs_name, self.outputs_name
        ) + "device: {}:{}\n".format(App.instance().get_device(), App.instance().get_gpu_ids())

    @classmethod
    def model_factory(cls, config_path):
        config = Config.from_yaml(config_path)

        dataset_module = config.module_name
        dataset_class = config.class_name
        dataset_module: BaseModel = get_class_object_from_name(dataset_module, dataset_class)
        return dataset_module(config)

    def forward(self, samples):
        for name in self.inputs_name:
            samples[name] = samples[name].to(device=App.instance().get_device()).float()
        return samples
