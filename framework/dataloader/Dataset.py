import torch
import torch.utils.data as datalodaer
from framework.app.app import *
from framework.dataloader.Transform import TRANSFORM
from utils.runtime import get_instance_from_name, get_class_object_from_name
from abc import *
import struct

class BaseDataset(datalodaer.Dataset, metaclass=ABCMeta):
    def __init__(self, example, required, meta, config):
        self._examples: list = example
        self._required: list = required
        self._meta = meta
        self.config = config
        self._transformers = []
        self.make_transforms()

    def make_transforms(self):
        for transforms_info in self.config.transforms:
            transforms_info = transforms_info.split(':')
            transform_name = transforms_info[0]
            required_inputs = set(self._required)
            if len(transforms_info) > 1:
                required_inputs = required_inputs.intersection(set(transforms_info[1].split('-')))
            self._transformers.append(TRANSFORM[transform_name](list(required_inputs), self._meta))

    @classmethod
    @abstractmethod
    def datasetFactory(self, config):
        raise NotImplementedError

    def __getitem__(self, item):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def _set_transform(self, transforms):
        self._transformers = transforms

