from utils.config import Config
import torch
import numpy as np
import logging
import logging.config

DEFAULT_CONFIG_PATH = "./framework/app/default_config.yaml"
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