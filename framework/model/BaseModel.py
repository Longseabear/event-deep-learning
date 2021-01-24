import torch
from utils.runtime import get_instance_from_name, get_class_object_from_name
from utils.config import Config
from framework.app.app import App
from framework.app.Format import *
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

    def get_name_format(self, controller):
        f = MainStateBasedFormatter(controller, {'model_name': App.instance().name, 'time': App.instance().time_format()},'[$time]_[$model_name]_[$main:epoch:03]_[$main:step:08].model')
        return f.Formatting()

    def save(self, info):
        dst = info['path']
        App.instance().make_save_dir(dst)

        file_names = os.listdir(dst)
        previous_file_name = None
        for name in file_names:
            _, file_extension = os.path.splitext(name)
            if file_extension == 'model':
                epoch = name.split('_')[-2]
                if int(epoch) == info['controller'].get_main_state().epoch:
                    previous_file_name = name

        saved_data = {'data': self.state_dict(), 'config': Config.extraction_dictionary(self.config)}
        saved_path = os.path.join(dst, self.get_name_format(info['controller']))
        torch.save(saved_data, saved_path)
        if previous_file_name is not None:
            os.remove(os.path.join(dst, previous_file_name))

        App.instance().set_variables('$latest_{}'.format(info['module_name']), saved_path)

    def load(self, info):
        path = info['path']
        state_dict = torch.load(path)
        if info.get('config_load', False):
            del self.config
            self.config = Config.from_dict(state_dict['config'])
        self.load_state_dict(state_dict['data'], strict=info['load_strict'])
