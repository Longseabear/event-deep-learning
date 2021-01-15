import torch
import torch.utils.data as datalodaer
from framework.dataloader.MetaData import MetaData
from framework.app.app import *
from utils.runtime import get_instance_from_name, get_class_object_from_name
from abc import *
import struct

class BaseDataset(datalodaer.Dataset, metaclass=ABCMeta):
    @abstractmethod
    def __init__(self, examples=None, transform_lists=[], required=[], config=None):
        self._examples: list = examples
        self._name2metaClass: dict = None
        self._transformers: list = transform_lists
        self._required: list = required
        self.config = config

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

