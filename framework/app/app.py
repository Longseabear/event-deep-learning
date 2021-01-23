from utils.config import Config
import torch
import numpy as np
import logging
import logging.config
import datetime
import sys
import torch
import os


DEFAULT_CONFIG_PATH = "{}/default_config.yaml".format(os.path.dirname(os.path.abspath( __file__ )))
class SingletoneInstance:
    __instance = None

    @classmethod
    def __getInstance(cls):
        return cls.__instance

    @classmethod
    def instance(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)
        cls.instance = cls.__getInstance # Override
        return cls.__instance

    @classmethod
    def reinitialize(cls, *args, **kargs):
        cls.__instance = cls(*args, **kargs)

    @classmethod
    def assign_instance(cls, instance):
        cls.__instance = instance

class App(SingletoneInstance):
    @classmethod
    def make_from_config_list(cls, config_paths):
        ''':key
            config_paths are order dependent. The higher the index, the more overwritten the value.
        '''
        config = Config.get_empty()

        for path in config_paths:
            config.update(config.from_yaml(path))
        return cls.instance(config, update=True)

    def __init__(self, config=None, update=False):
        self.config = Config.from_yaml(DEFAULT_CONFIG_PATH)
        if config is not None:
            self.config.update(config)

        if update:
            self.update(self.config)

        self.name = self.config.App.NAME
        self.mode = self.config.App.mode

        base = self.config.App.save_dir
        model_name = self.name
        experiment_name = self.config.App.EXPERIMENT_NAME
        if self.mode == 'finetuning':
            if os.path.exists(os.path.join(base, model_name, experiment_name)):
                experiment_name = self.name_format(experiment_name)
            base_path = os.path.join(base, model_name, experiment_name)
            self.config.App.base_path = base_path
            print('[INFO] Now App mode is {}, Since the directory already exists, a new folder {} is created.'.format(self.mode,
                                                                                                                      self.config.App.base_path))
        elif self.mode == 'new':
            if os.path.exists(os.path.join(base, model_name, experiment_name)):
                experiment_name = self.name_format(experiment_name)
            base_path = os.path.join(base, model_name, experiment_name)
            self.config.App.base_path = base_path
            print('[INFO] Now App mode is {}, Since the directory already exists, a new folder {} is created.'.format(self.mode,
                                                                                                                      self.config.App.base_path))
        elif self.mode == 'resume':
            if not os.path.exists(self.config.App.base_path):
                raise FileExistsError
        else:
            raise NotImplementedError

    def update(self, config):
        self.system = config.SYSTEM

        logging.config.dictConfig(Config.extraction_dictionary(self.system.LOGGER))
        self.logger = logging.getLogger('DEFAULT')
        self.setSeed(self.system.SEED)

    def setSeed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)

        if self.system.REPRODUCIBILITY:
            self.logger.info('REPRODUCIBILITY MODE TRUE')
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def name_format(self, name):
        now = datetime.datetime.now()
        return now.strftime('{}_%y%m%d_%Hh%Mm'.format(name))

    def current_time_format(self):
        now = datetime.datetime.now()
        return now.strftime('[%y%m%d_%Hh%Mm]'.format())

    def get_device(self):
        device_name = self.config.App.device
        return device_name

    def get_gpu_ids(self):
        gpu_ids = self.config.App.gpu_ids
        return gpu_ids

    def set_gpu_device(self, module: torch.nn.Module):
        device_name = self.config.App.device
        gpu_ids = self.config.App.gpu_ids
        parallel = self.config.App.parallel

        device = torch.device('cpu' if device_name is 'cpu' else 'cuda')

        if device_name is not 'cpu' and len(gpu_ids) > 1 and parallel is "data":
            return torch.nn.DataParallel(module, gpu_ids).to(device)
        return module.to(device)

    def get_base_path(self):
        return self.config.App.base_path

    def make_save_dir(self, name, path=None):
        if path == None:
            path = self.config.App.base_path
        os.makedirs(os.path.join(path, name), exist_ok=True)

    def smart_write(self, contents, dst, mode=None):
        if isinstance(dst, str):
            if dst == 'stdout':
                sys.stdout.write(contents)
            else:
                with open(dst, mode) as f:
                    f.write(contents)
        else:
            with open(dst, mode) as f:
                f.write(contents)
