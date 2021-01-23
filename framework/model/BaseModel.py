import torch
from utils.runtime import get_instance_from_name, get_class_object_from_name
from utils.config import Config
from framework.app.app import App
import os


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

    def save(self, info):
        dst = info['path']
        file_names = os.listdir(dst)
        previous_file_name = None
        for name in file_names:
            _, file_extension = os.path.splitext(name)
            if file_extension == 'model':
                epoch = name.split('_')[-2]
                if int(epoch) == info['state'].epoch:
                    previous_file_name = name

        torch.save(self.state_dict(), os.path.join(dst, self.get_save_name(info['state'])))
        if previous_file_name is not None:
            os.remove(os.path.join(dst, previous_file_name))

    def load(self, info):
        self.load_state_dict(torch.load(info['path']))

    def get_save_name(self, state):
        return App.instance().name_format(App.instance().name) + "_{}_{}.model".format(state.epoch, state.step)

    def get_dir(self, state, dir_path):
        pass